"""Scalar evidence for the MMMU-Pro vision study (feature contract vision_mc_scalar_v1).

Computed from one model output's token trace and same-forward native trace, before any
grading. The statistics mirror scripts/selector_dev1000.py so the two domains describe an
output the same way: "out" covers every content token, "ans" the single token carrying the
registered answer letter. Visual-embedding statistics come from the offline HF pass and
live in a separate evidence file.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from authority_allocation.math_expression_spans import content_token_indices
from authority_allocation.mmmu_pro import locate_answer_letter

FEATURE_CONTRACT = "vision_mc_scalar_v1"
LOW_CONFIDENCE = math.log(0.5)

PRE_OUTPUT_FEATURES = ("n_options", "n_images", "question_char_count", "num_visual_tokens",
                       "prompt_tokens")
OUTPUT_FEATURES = ("answer_parseable", "output_token_count", "truncated",
                   "out_lp_mean", "out_lp_min", "out_lp_p10", "out_lp_std", "out_ent_mean",
                   "out_ent_p90", "out_ent_max", "out_margin_mean", "out_margin_p10",
                   "out_max_logit_mean", "out_lowconf_frac", "out_lowconf_run",
                   "ans_lp", "ans_ent", "ans_margin", "ans_max_logit")
FEATURES = PRE_OUTPUT_FEATURES + OUTPUT_FEATURES


def answer_token_index(text: str, trace: Mapping[str, Any], n_options: int) -> int | None:
    """Index of the one token covering the registered answer letter, if exactly one does."""
    located = locate_answer_letter(text, n_options)
    if located is None or trace.get("status") != "ok":
        return None
    _, char_start = located
    offset = int(trace.get("content_start_byte") or 0)
    start = offset + len(text[:char_start].encode("utf-8"))
    end = start + 1
    covering = [int(t["index"]) for t in trace.get("tokens", [])
                if t["end_byte"] > start and t["start_byte"] < end]
    return covering[0] if len(covering) == 1 else None


def _output_stats(positions: list[Mapping[str, Any]]) -> dict[str, float | None]:
    names = ("out_lp_mean", "out_lp_min", "out_lp_p10", "out_lp_std", "out_ent_mean",
             "out_ent_p90", "out_ent_max", "out_margin_mean", "out_margin_p10",
             "out_max_logit_mean", "out_lowconf_frac", "out_lowconf_run")
    if not positions:
        return {name: None for name in names}
    lp = np.array([p["sampled_logprob"] for p in positions], dtype=float)
    ent = np.array([p["entropy"] for p in positions], dtype=float)
    margin = np.array([p["top1_top2_margin"] for p in positions], dtype=float)
    max_logit = np.array([p["max_raw_logit"] for p in positions], dtype=float)
    low = lp < LOW_CONFIDENCE
    run = best = 0
    for flag in low:
        run = run + 1 if flag else 0
        best = max(best, run)
    return {
        "out_lp_mean": float(lp.mean()), "out_lp_min": float(lp.min()),
        "out_lp_p10": float(np.percentile(lp, 10)), "out_lp_std": float(lp.std()),
        "out_ent_mean": float(ent.mean()), "out_ent_p90": float(np.percentile(ent, 90)),
        "out_ent_max": float(ent.max()), "out_margin_mean": float(margin.mean()),
        "out_margin_p10": float(np.percentile(margin, 10)),
        "out_max_logit_mean": float(max_logit.mean()),
        "out_lowconf_frac": float(low.mean()), "out_lowconf_run": float(best),
    }


def derive_features(*, output: Mapping[str, Any], native: Mapping[str, Any],
                    n_options: int, n_images: int, question_text: str,
                    num_visual_tokens: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(features, audit) for one output. A feature is None when its tokens cannot all be
    paired with the native trace, never a partial aggregate."""
    text = output.get("raw_output") or ""
    trace = output.get("token_logprobs") or {}
    usage = output.get("usage") or {}
    by_index = {int(p["index"]): p for p in native.get("positions", [])}
    content = content_token_indices(trace)
    positions = [by_index[i] for i in content if i in by_index]
    if len(positions) != len(content):
        positions = []
    answer_index = answer_token_index(text, trace, n_options)
    answer = by_index.get(answer_index) if answer_index is not None else None
    features: dict[str, Any] = {
        "n_options": n_options,
        "n_images": n_images,
        "question_char_count": len(question_text),
        "num_visual_tokens": num_visual_tokens,
        "prompt_tokens": usage.get("input_tokens"),
        "answer_parseable": int(locate_answer_letter(text, n_options) is not None),
        "output_token_count": usage.get("output_tokens"),
        "truncated": int(usage.get("finish_reason") == "length"),
        **_output_stats(positions),
        "ans_lp": None if answer is None else float(answer["sampled_logprob"]),
        "ans_ent": None if answer is None else float(answer["entropy"]),
        "ans_margin": None if answer is None else float(answer["top1_top2_margin"]),
        "ans_max_logit": None if answer is None else float(answer["max_raw_logit"]),
    }
    if tuple(features) != FEATURES:
        raise AssertionError("feature order drifted from the contract")
    audit = {"content_token_count": len(content), "paired_token_count": len(positions),
             "answer_token_index": answer_index}
    return features, audit
