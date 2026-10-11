"""Auditable helpers for generated-token logprob feature extraction.

The helpers operate on UTF-8 byte spans because OpenAI-compatible chat
responses expose token bytes.  A field feature includes every generated token
whose byte span overlaps the registered source span; this rule is deterministic
even when a tokenizer merges punctuation with field contents.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_generated_token_trace(
    response_json: Mapping[str, Any],
    *,
    expected_text: str,
    logprobs_mode: str,
) -> dict[str, Any]:
    """Normalize sampled-token logprobs and locate the exact visible content."""

    choices = response_json.get("choices") or []
    choice = choices[0] if choices else {}
    logprobs = choice.get("logprobs") or {}
    entries = logprobs.get("content")
    if not isinstance(entries, list):
        return {
            "status": "missing",
            "logprobs_mode": logprobs_mode,
            "tokens": [],
            "reconstructed_text_sha256": None,
            "content_start_byte": None,
            "content_end_byte": None,
        }

    choice_token_ids = choice.get("token_ids")
    if not isinstance(choice_token_ids, list) or len(choice_token_ids) != len(entries):
        choice_token_ids = [None] * len(entries)

    tokens: list[dict[str, Any]] = []
    reconstructed = bytearray()
    for index, (entry, fallback_token_id) in enumerate(
        zip(entries, choice_token_ids, strict=True)
    ):
        if not isinstance(entry, Mapping):
            raise ValueError("logprob content entries must be objects")
        byte_values = entry.get("bytes")
        if isinstance(byte_values, list) and all(
            isinstance(value, int) and 0 <= value <= 255 for value in byte_values
        ):
            token_bytes = bytes(byte_values)
        else:
            token = entry.get("token")
            if not isinstance(token, str):
                raise ValueError("logprob entry lacks token bytes and text")
            token_bytes = token.encode("utf-8")
        logprob = entry.get("logprob")
        if (
            not isinstance(logprob, (int, float))
            or not math.isfinite(logprob)
            or logprob > 1e-12
        ):
            raise ValueError("sampled-token logprob must be finite and <= 0")
        start = len(reconstructed)
        reconstructed.extend(token_bytes)
        token_id = entry.get("token_id", fallback_token_id)
        if token_id is not None and not isinstance(token_id, int):
            token_id = None
        tokens.append(
            {
                "index": index,
                "token_id": token_id,
                "token": entry.get("token") if isinstance(entry.get("token"), str) else None,
                "bytes": list(token_bytes),
                "start_byte": start,
                "end_byte": len(reconstructed),
                "logprob": float(logprob),
            }
        )

    reconstructed_bytes = bytes(reconstructed)
    content_bytes = expected_text.encode("utf-8")
    content_start: int | None = None
    if content_bytes and reconstructed_bytes.endswith(content_bytes):
        content_start = len(reconstructed_bytes) - len(content_bytes)
    elif content_bytes and reconstructed_bytes.count(content_bytes) == 1:
        content_start = reconstructed_bytes.index(content_bytes)
    status = "ok" if content_start is not None else "text_mismatch"
    return {
        "status": status,
        "logprobs_mode": logprobs_mode,
        "tokens": tokens,
        "reconstructed_text_sha256": _sha256(reconstructed_bytes),
        "content_start_byte": content_start,
        "content_end_byte": (
            None if content_start is None else content_start + len(content_bytes)
        ),
    }


def _char_span_to_bytes(text: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    if not 0 <= start <= end <= len(text):
        raise ValueError("character span is outside the source text")
    return len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))


def _overlapping_tokens(
    trace: Mapping[str, Any], byte_span: tuple[int, int]
) -> list[Mapping[str, Any]]:
    if trace.get("status") != "ok":
        return []
    start, end = byte_span
    return [
        token
        for token in trace.get("tokens", [])
        if token["end_byte"] > start and token["start_byte"] < end
    ]


def mean_logprob(
    trace: Mapping[str, Any],
    *,
    text: str | None = None,
    char_span: tuple[int, int] | None = None,
) -> float | None:
    """Mean sampled-token natural logprob over all tokens or one text span."""

    if trace.get("status") != "ok":
        return None
    content_start = trace.get("content_start_byte")
    content_end = trace.get("content_end_byte")
    if not isinstance(content_start, int) or not isinstance(content_end, int):
        return None
    if char_span is None:
        tokens = _overlapping_tokens(trace, (content_start, content_end))
    else:
        if text is None:
            raise ValueError("text is required with char_span")
        local_start, local_end = _char_span_to_bytes(text, char_span)
        tokens = _overlapping_tokens(
            trace, (content_start + local_start, content_start + local_end)
        )
    if not tokens:
        return None
    return sum(float(token["logprob"]) for token in tokens) / len(tokens)


def token_count_for_span(
    trace: Mapping[str, Any], text: str, char_span: tuple[int, int]
) -> int | None:
    if trace.get("status") != "ok":
        return None
    content_start = trace.get("content_start_byte")
    if not isinstance(content_start, int):
        return None
    local_start, local_end = _char_span_to_bytes(text, char_span)
    tokens = _overlapping_tokens(
        trace, (content_start + local_start, content_start + local_end)
    )
    return len(tokens) if tokens else None


def boxed_payload_span(text: str, *, within: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """Find the payload of the last balanced ``\\boxed{...}`` occurrence."""

    lower, upper = within if within is not None else (0, len(text))
    marker = "\\boxed{"
    marker_start = text.rfind(marker, lower, upper)
    if marker_start < 0:
        return None
    payload_start = marker_start + len(marker)
    depth = 1
    for index in range(payload_start, upper):
        character = text[index]
        if character == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif character == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return payload_start, index
    return None


def json_string_value_span(text: str, key: str) -> tuple[int, int] | None:
    """Locate one JSON string value as a raw encoded character span."""

    matches = list(re.finditer(rf'"{re.escape(key)}"\s*:\s*"', text))
    if len(matches) != 1:
        return None
    match = matches[0]
    start = match.end()
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            # Confirm that the raw slice is a valid JSON string, so malformed
            # output never creates a fabricated field span.
            try:
                json.loads('"' + text[start:index] + '"')
            except json.JSONDecodeError:
                return None
            return start, index
    return None


def derive_scalar_logprob_features(
    *,
    g_text: str,
    g_trace: Mapping[str, Any],
    v_text: str,
    v_trace: Mapping[str, Any],
    v_report_parse_status: str,
) -> dict[str, float | None]:
    """Compute the six logprob/length features registered by scalar_single_v2."""

    g_box = boxed_payload_span(g_text)
    result: dict[str, float | None] = {
        "g_output_mean_logprob": mean_logprob(g_trace),
        "g_final_answer_mean_logprob": (
            None if g_box is None else mean_logprob(g_trace, text=g_text, char_span=g_box)
        ),
        "v_verdict_mean_logprob": None,
        "v_request_mean_logprob": None,
        "v_candidate_final_answer_mean_logprob": None,
        "v_critique_token_length_log": None,
    }
    # A format-repair call has a different prompt-conditioned distribution and
    # is deliberately not mixed with first-pass V likelihoods.
    if v_report_parse_status != "ok":
        return result

    verdict_span = json_string_value_span(v_text, "verdict")
    request_span = json_string_value_span(v_text, "requested_action")
    critique_span = json_string_value_span(v_text, "critique")
    candidate_span = json_string_value_span(v_text, "candidate_answer")
    candidate_box = (
        None if candidate_span is None else boxed_payload_span(v_text, within=candidate_span)
    )
    if verdict_span is not None:
        result["v_verdict_mean_logprob"] = mean_logprob(
            v_trace, text=v_text, char_span=verdict_span
        )
    if request_span is not None:
        result["v_request_mean_logprob"] = mean_logprob(
            v_trace, text=v_text, char_span=request_span
        )
    if candidate_box is not None:
        result["v_candidate_final_answer_mean_logprob"] = mean_logprob(
            v_trace, text=v_text, char_span=candidate_box
        )
    if critique_span is not None:
        count = token_count_for_span(v_trace, v_text, critique_span)
        result["v_critique_token_length_log"] = (
            None if count is None else math.log1p(count)
        )
    return result


def derive_scalar_logprob_features_v3(
    *,
    g_text: str,
    g_trace: Mapping[str, Any],
    v_text: str,
    v_trace: Mapping[str, Any],
    v_report_parse_status: str,
) -> dict[str, float | None]:
    """Compute scalar_single_v3 using the complete candidate-answer field.

    V2 attempted to isolate a ``\\boxed{...}`` payload inside candidate_answer.
    Real c003 outputs did not provide a registered boxed source span, so v3
    replaces that feature with the mean over the complete raw JSON string value.
    All other v2 formulas and retry/fallback semantics remain unchanged.
    """

    result = derive_scalar_logprob_features(
        g_text=g_text,
        g_trace=g_trace,
        v_text=v_text,
        v_trace=v_trace,
        v_report_parse_status=v_report_parse_status,
    )
    result.pop("v_candidate_final_answer_mean_logprob")
    result["v_candidate_answer_mean_logprob"] = None
    if v_report_parse_status != "ok":
        return result
    candidate_span = json_string_value_span(v_text, "candidate_answer")
    if candidate_span is not None:
        result["v_candidate_answer_mean_logprob"] = mean_logprob(
            v_trace, text=v_text, char_span=candidate_span
        )
    return result
