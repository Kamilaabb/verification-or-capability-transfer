"""vLLM same-forward logit summaries for c014.

This module is loaded inside the pinned vLLM engine process via ``--logits-processors``.
It returns the input tensor unchanged. Full-vocabulary logits never leave the GPU and
are never written to disk; only registered scalar reductions and top-20 values cross to
CPU. Requests opt in through ``vllm_xargs.c014_capture_id``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


class C014LogitCapture(LogitsProcessor):
    """Capture summaries from the exact raw-logit tensor used for greedy sampling."""

    @classmethod
    def validate_params(cls, sampling_params: Any) -> None:
        extra = sampling_params.extra_args or {}
        capture_id = extra.get("c014_capture_id")
        if capture_id is not None and (not isinstance(capture_id, str) or not capture_id):
            raise ValueError("c014_capture_id must be a nonempty string")

    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool) -> None:
        del vllm_config, device, is_pin_memory
        configured = os.environ.get("C014_CAPTURE_PATH")
        if not configured:
            raise RuntimeError("C014_CAPTURE_PATH is required by C014LogitCapture")
        self.path = Path(configured)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.states: dict[int, dict[str, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        # vLLM 0.29 does not invoke processors in its argmax_invariant list when
        # every request is greedy. Register in the always-applied list instead;
        # apply() still returns the tensor byte-for-byte unchanged.
        return False

    @staticmethod
    def _finalize_committed_tokens(state: dict[str, Any]) -> list[int]:
        # The V1 runner keeps a trailing -1 slot for the next token. It can also
        # invoke the processor once after a stop token before discarding that step.
        # Pair only the final committed, non-negative prefix at request removal.
        output_ids: list[int] = []
        for value in state["output_token_ids"]:
            token_id = int(value)
            if token_id < 0:
                break
            output_ids.append(token_id)
        positions = state["positions"]
        # At the max_tokens boundary vLLM removes the request before its live output
        # list receives the final greedy token. The final forward is nevertheless in
        # positions, and its top-1 is the sampled token by the registered temperature-0
        # decoding. Recover exactly that one boundary token; do not do this for EOS/stop
        # termination, where max_tokens has not been reached.
        max_tokens = state.get("max_tokens")
        if (isinstance(max_tokens, int)
                and len(positions) == max_tokens
                and len(output_ids) + 1 == max_tokens):
            output_ids.append(int(positions[-1]["top20_token_ids"][0]))
        if len(positions) < len(output_ids):
            raise RuntimeError("c014 capture has fewer forward steps than committed tokens")
        del positions[len(output_ids):]
        for sampled_id, position in zip(output_ids, positions, strict=True):
            position["sampled_token_id"] = sampled_id
            top_ids = position["top20_token_ids"]
            top_logits = position["top20_raw_logits"]
            try:
                sampled_at = top_ids.index(sampled_id)
            except ValueError:
                position["sampled_raw_logit"] = None
            else:
                position["sampled_raw_logit"] = top_logits[sampled_at]
        return output_ids

    @staticmethod
    def _is_writer() -> bool:
        # Under tensor parallelism every rank may run this processor on the same logits; only
        # rank 0 writes, so each capture appears once. Without tensor parallelism the rank is 0.
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            return get_tensor_model_parallel_rank() == 0
        except (ImportError, AssertionError, RuntimeError):
            return True

    def _flush(self, state: dict[str, Any]) -> None:
        if not self._is_writer():
            return
        output_ids = self._finalize_committed_tokens(state)
        payload = {
            "capture_id": state["capture_id"],
            "positions": state["positions"],
            "output_token_ids": output_ids,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not batch_update:
            return

        for index in batch_update.removed:
            state = self.states.pop(index, None)
            if state is not None:
                self._flush(state)

        for index, params, _, output_token_ids in batch_update.added:
            extra = params.extra_args or {}
            capture_id = extra.get("c014_capture_id")
            if capture_id:
                self.states[index] = {
                    "capture_id": capture_id,
                    "max_tokens": params.max_tokens,
                    "output_token_ids": output_token_ids,
                    "positions": [],
                }
            else:
                self.states.pop(index, None)

        for source, destination, direction in batch_update.moved:
            source_state = self.states.pop(source, None)
            destination_state = self.states.pop(destination, None)
            if source_state is not None:
                self.states[destination] = source_state
            if direction == MoveDirectionality.SWAP and destination_state is not None:
                self.states[source] = destination_state

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, state in self.states.items():
            if index >= logits.shape[0]:
                continue
            # One record per generation step. A pending record means the engine called
            # apply twice without committing the prior sample, which must never be
            # silently duplicated.
            if len(state["positions"]) != len(state["output_token_ids"]):
                raise RuntimeError("c014 capture state is not aligned with generated tokens")
            row = logits[index].float()
            logsumexp = torch.logsumexp(row, dim=-1)
            log_probs = row - logsumexp
            probabilities = torch.exp(log_probs)
            # Entropy is mathematically non-negative. Clamp only possible round-off
            # below zero so the persisted scalar obeys that invariant.
            entropy = (-(probabilities * log_probs).sum()).clamp_min(0.0)
            top_values, top_ids = torch.topk(row, k=20, dim=-1)
            values = [float(value) for value in top_values.detach().cpu().tolist()]
            ids = [int(value) for value in top_ids.detach().cpu().tolist()]
            scalar_values = [float(logsumexp), float(entropy)]
            if not all(math.isfinite(value) for value in values + scalar_values):
                raise RuntimeError("c014 capture produced a non-finite logit summary")
            state["positions"].append({
                "index": len(state["positions"]),
                "sampled_token_id": None,
                "sampled_raw_logit": None,
                "max_raw_logit": values[0],
                "logsumexp": scalar_values[0],
                "entropy": scalar_values[1],
                "top1_top2_margin": values[0] - values[1],
                "top20_token_ids": ids,
                "top20_raw_logits": values,
            })
        return logits
