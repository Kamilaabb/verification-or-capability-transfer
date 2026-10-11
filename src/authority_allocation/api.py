"""OpenAI-compatible chat calls against a locally served model.

Domain-neutral: the caller supplies the decoding block and any constrained schema, so
the same client serves a math verifier and a code verifier without branching.

A format-repair request rejected with HTTP 400 is a repair failure, not a crash — the
caller applies its registered fallback. Letting that exception escape once killed a run
after every model stage had already completed.

Some chat templates read the wall clock. Llama 3.2's writes "Today Date: <day of the
call>" into every system turn unless ``date_string`` is passed, so the same row would be
a different prompt on a different day, including across a resumed run. The condition's
model block pins such inputs as ``chat_template_kwargs``; ``chat_template_kwargs_for``
refuses to call a clock-reading template without a pinned date.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from urllib import request
from urllib.error import HTTPError

from authority_allocation.generated_token_logprobs import build_generated_token_trace
from authority_allocation.serialization import canonical_json


def usage_from_response(
    response_json: Mapping[str, Any], latency_ms: float, retries: int
) -> dict[str, Any]:
    usage = response_json.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "latency_ms": latency_ms,
        "finish_reason": (response_json.get("choices") or [{}])[0].get("finish_reason"),
        "retries": retries,
    }


CLOCK_READING_TEMPLATE_MARKER = "strftime_now"


def chat_template_kwargs_for(model: Mapping[str, Any]) -> dict[str, Any] | None:
    """The registered extra chat-template inputs of a condition's model block.

    When the served snapshot is present, its template is read: a template that calls
    ``strftime_now`` must have ``date_string`` pinned, or the call is refused.
    """
    kwargs = dict(model.get("chat_template_kwargs") or {})
    if "enable_thinking" in kwargs:
        raise SystemExit("enable_thinking comes from model.thinking, not chat_template_kwargs")
    server_path = model.get("server_path")
    config = Path(server_path) / "tokenizer_config.json" if server_path else None
    if config is not None and config.is_file():
        template = json.loads(config.read_text(encoding="utf-8")).get("chat_template") or ""
        if CLOCK_READING_TEMPLATE_MARKER in template and not kwargs.get("date_string"):
            raise SystemExit(
                f"the chat template in {config} writes the call date into the prompt; "
                "pin model.chat_template_kwargs.date_string in the condition")
    return kwargs or None


def api_call(
    *,
    api_base: str,
    served_model: str,
    messages: list[dict[str, str]],
    decoding: Mapping[str, Any],
    seed: int,
    thinking: bool,
    schema: Mapping[str, Any] | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    timeout_seconds: int = 900,
    api_key: str = "local",
) -> tuple[str, str | None, dict[str, Any]]:
    template_kwargs: dict[str, Any] = {"enable_thinking": thinking}
    if chat_template_kwargs:
        if "enable_thinking" in chat_template_kwargs:
            raise ValueError("enable_thinking is set by the thinking argument")
        template_kwargs.update(chat_template_kwargs)
    body: dict[str, Any] = {
        "model": served_model,
        "messages": messages,
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "max_tokens": decoding["max_new_tokens"],
        "seed": seed,
        "chat_template_kwargs": template_kwargs,
    }
    logprob_config = decoding.get("output_logprobs") or {}
    if logprob_config.get("enabled"):
        body["logprobs"] = True
        body["top_logprobs"] = int(logprob_config.get("top_logprobs", 0))
        body["return_token_ids"] = bool(logprob_config.get("return_token_ids", True))
    if decoding.get("vllm_xargs"):
        body["vllm_xargs"] = dict(decoding["vllm_xargs"])
    # vLLM fills any parameter a request omits from the model's generation_config.json, so
    # every registered penalty is sent. Qwen2.5-3B-Instruct's config carries
    # repetition_penalty 1.05, which would silently change greedy output from the
    # registered 1.0; Qwen3's and Llama 3.2's configs carry no penalty, so their runs
    # before this line decoded with 1.0 as registered.
    for key in ("top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty"):
        if decoding.get(key) is not None:
            body[key] = decoding[key]
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "registered_output", "strict": True, "schema": schema},
        }
    url = api_base.rstrip("/") + "/chat/completions"
    http_request = request.Request(
        url,
        data=canonical_json(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(http_request, timeout=timeout_seconds) as response:
        response_json = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000
    choice = response_json["choices"][0]
    message = choice["message"]
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if not isinstance(content, str):
        content = ""
    usage_record = usage_from_response(response_json, latency_ms, retries=0)
    if logprob_config.get("enabled"):
        usage_record["token_logprobs"] = build_generated_token_trace(
            response_json,
            expected_text=content,
            logprobs_mode=str(logprob_config.get("mode", "raw_logprobs")),
        )
    if logprob_config.get("return_prompt_token_ids"):
        # The vision study counts <|image_pad|> in the served prompt to prove the registered
        # pixel limits took effect; other callers never ask, so their usage is unchanged.
        usage_record["prompt_token_ids"] = response_json.get("prompt_token_ids")
    return content, reasoning, usage_record


def format_repair_call(**kwargs: Any) -> tuple[str, str | None, dict[str, Any]] | None:
    """Attempt the registered format repair; an HTTP rejection means repair failed."""

    try:
        return api_call(**kwargs)
    except HTTPError as exc:
        print(
            f"Format repair request failed with HTTP {exc.code}; applying registered fallback.",
            file=sys.stderr,
        )
        return None
