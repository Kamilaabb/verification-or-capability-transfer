"""Parse c018 verifier reports and derive auditable scalar features."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from authority_allocation.math_expression_spans import (
    MathSpan,
    content_token_indices,
    overlapping_token_indices,
    parse_math_spans,
)


VERDICTS = ("CORRECT", "INCORRECT", "UNCERTAIN")
ACTIONS = ("KEEP", "CRITIQUE_REFINE")

# Every parsed report is classified the same way whatever its verdict: whether V quoted
# a location, and if so how the quotation matches the G output. The verdict features
# already carry the verdict, so a controller can learn any interaction between the two.
LOCATION_OUTCOMES = ("exact", "whitespace_match", "not_found", "ambiguous")
LOCATION_STATUSES = ("not_given", *LOCATION_OUTCOMES, "report_fallback")
LOCATION_FEATURES = (
    "v_error_location_exact",
    "v_error_location_given",
    "v_error_location_whitespace_match",
    "v_error_location_not_found",
    "v_error_location_ambiguous",
)

FEATURE_NAMES = (
    "v_verdict_correct",
    "v_verdict_incorrect",
    "v_verdict_uncertain",
    "v_action_keep",
    "v_action_critique_refine",
    "v_candidate_answer_parseable",
    "gv_final_answer_equivalent",
    "v_verdict_correct_mean_logprob",
    "v_verdict_incorrect_mean_logprob",
    "v_verdict_uncertain_mean_logprob",
    "v_action_mean_logprob",
    "v_math_expression_mean_logprob",
    "v_critique_token_length_log",
    "v_verdict_correct_mean_max_logit",
    "v_verdict_incorrect_mean_max_logit",
    "v_verdict_uncertain_mean_max_logit",
    "v_action_mean_max_logit",
    "v_math_expression_mean_max_logit",
    *LOCATION_FEATURES,
    "v_error_location_mean_logprob",
    "v_error_location_mean_max_logit",
    "g_error_location_mean_logprob",
    "g_error_location_mean_max_logit",
    "v_output_mean_logprob",
    "v_output_mean_max_logit",
)

# Section headers must appear in order, each on its own line, with VERDICT and ACTION
# values on their header lines. Whitespace-only deviations are tolerated: leading or
# trailing whitespace, blank lines between sections, and a field value written on the
# same line as its header. A field's END marker may be omitted when the next section
# header, or the end of output for FINAL ANSWER, already delimits the field. Field text
# may not contain its own END marker or run into the next header, so an END marker
# written inline, or prose after END FINAL ANSWER, fails rather than being absorbed.
# Anything else that adds or drops a token (Markdown, code fences, surrounding prose,
# echoed choice bars, a missing header, truncated output) still fails.
_FIELD_HEADERS = ("ERROR LOCATION", "CRITIQUE", "CORRECTED CONTENT", "FINAL ANSWER")


def _field_pattern(header: str, group: str, next_header: str | None) -> str:
    if next_header is None:
        return (rf"{header}:(?P<{group}>(?:(?!END {header}).)*?)"
                rf"(?:\n[ \t]*END {header})?\s*\Z")
    return (rf"{header}:(?P<{group}>(?:(?!END {header}|\n[ \t]*{next_header}:).)*?)"
            rf"(?:\n[ \t]*END {header}[ \t]*)?\n\s*(?={next_header}:)")


REPORT_PATTERN = re.compile(
    r"\A\s*"
    r"VERDICT:[ \t]*(?P<verdict>CORRECT|INCORRECT|UNCERTAIN)[ \t]*\n\s*"
    r"ACTION:[ \t]*(?P<action>KEEP|CRITIQUE_REFINE)[ \t]*\n\s*"
    + "".join(
        _field_pattern(
            header,
            header.lower().replace(" ", "_"),
            _FIELD_HEADERS[index + 1] if index + 1 < len(_FIELD_HEADERS) else None,
        )
        for index, header in enumerate(_FIELD_HEADERS)
    ),
    re.DOTALL,
)

_WHITESPACE = re.compile(r"\s+")


def _trimmed_group(match: re.Match[str], name: str) -> tuple[str, tuple[int, int]]:
    raw = match.group(name)
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    start = match.start(name) + left
    end = match.start(name) + right
    return match.string[start:end], (start, end)


def parse_verifier_report(text: str) -> tuple[dict[str, Any], dict[str, list[int]]] | None:
    """Parse the tagged first-pass format without repairing model output."""

    match = REPORT_PATTERN.fullmatch(text)
    if match is None:
        return None
    fields: dict[str, str] = {}
    spans: dict[str, list[int]] = {
        "verdict": [match.start("verdict"), match.end("verdict")],
        "action": [match.start("action"), match.end("action")],
    }
    for name in ("error_location", "critique", "corrected_content", "final_answer"):
        value, span = _trimmed_group(match, name)
        fields[name] = value
        spans[name] = [span[0], span[1]]
    if not fields["critique"] or not fields["final_answer"]:
        return None
    report: dict[str, Any] = {
        "verdict": match.group("verdict"),
        "action": match.group("action"),
        "error_location": (
            None if fields["error_location"].upper() == "NONE"
            else fields["error_location"]
        ),
        "critique": fields["critique"],
        "corrected_content": (
            None if fields["corrected_content"].upper() == "NONE"
            else fields["corrected_content"]
        ),
        "final_answer": fields["final_answer"],
    }
    return report, spans


def _span_indices(
    trace: Mapping[str, Any], text: str, span: tuple[int, int] | list[int]
) -> list[int]:
    return overlapping_token_indices(
        trace, text, [MathSpan(int(span[0]), int(span[1]), "field")]
    )


def _mean_logprob(trace: Mapping[str, Any], indices: list[int]) -> float | None:
    by_index = {int(token["index"]): token for token in trace.get("tokens", [])}
    values = [float(by_index[index]["logprob"]) for index in indices if index in by_index]
    return None if not values else sum(values) / len(values)


def _mean_max_logit(native: Mapping[str, Any], indices: list[int]) -> float | None:
    if native.get("status") != "ok":
        return None
    by_index = {int(position["index"]): position for position in native.get("positions", [])}
    values = [float(by_index[index]["max_raw_logit"])
              for index in indices if index in by_index]
    return None if not values else sum(values) / len(values)


def _occurrences(needle: str, haystack: str) -> list[int]:
    """Start offsets of every occurrence of needle, overlapping ones included."""

    starts: list[int] = []
    at = haystack.find(needle)
    while at >= 0:
        starts.append(at)
        at = haystack.find(needle, at + 1)
    return starts


def _location(
    g_text: str, report: Mapping[str, Any] | None
) -> tuple[str, tuple[int, int] | None]:
    """Classify the quoted error location. Only a unique exact match yields a span."""

    if report is None:
        return "report_fallback", None
    quote = report.get("error_location")
    if not isinstance(quote, str) or not quote:
        return "not_given", None
    exact = _occurrences(quote, g_text)
    if len(exact) == 1:
        return "exact", (exact[0], exact[0] + len(quote))
    if exact:
        return "ambiguous", None
    # Diagnostic only: a whitespace-insensitive hit is counted but yields no span, so the
    # token-level location features stay tied to an exact quotation.
    relaxed = _occurrences(_WHITESPACE.sub("", quote), _WHITESPACE.sub("", g_text))
    if len(relaxed) == 1:
        return "whitespace_match", None
    if relaxed:
        return "ambiguous", None
    return "not_found", None


def _location_features(status: str) -> dict[str, int | None]:
    """Encode a location status: given is 0/1, and when 1 exactly one outcome cell is 1."""

    if status == "report_fallback":
        return {name: None for name in LOCATION_FEATURES}
    values: dict[str, int | None] = {"v_error_location_given": int(status != "not_given")}
    for outcome in LOCATION_OUTCOMES:
        values[f"v_error_location_{outcome}"] = int(status == outcome)
    return values


def derive_verifier_scalar_evidence(
    *,
    g_text: str,
    g_trace: Mapping[str, Any],
    g_native: Mapping[str, Any],
    g_answer: str | None,
    v_text: str,
    v_trace: Mapping[str, Any],
    v_native: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    field_spans: Mapping[str, list[int]],
    v_answer: str | None,
    answers_equivalent: bool | None,
) -> dict[str, Any]:
    """Return every registered scalar plus token/span audit data."""

    field_indices = {
        name: _span_indices(v_trace, v_text, span)
        for name, span in field_spans.items()
    }
    v_math_spans = parse_math_spans(v_text)
    v_math_indices = overlapping_token_indices(v_trace, v_text, v_math_spans)
    v_output_indices = content_token_indices(v_trace)

    location_status, g_location_span = _location(g_text, report)
    located = g_location_span is not None
    g_location_indices = (
        _span_indices(g_trace, g_text, g_location_span) if located else []
    )
    v_location_indices = field_indices.get("error_location", []) if located else []

    features: dict[str, Any] = {name: None for name in FEATURE_NAMES}
    if report is not None:
        verdict = str(report["verdict"])
        action = str(report["action"])
        for label in VERDICTS:
            features[f"v_verdict_{label.lower()}"] = int(verdict == label)
        features["v_action_keep"] = int(action == "KEEP")
        features["v_action_critique_refine"] = int(action == "CRITIQUE_REFINE")
        features["v_candidate_answer_parseable"] = int(v_answer is not None)
        features["gv_final_answer_equivalent"] = (
            answers_equivalent if g_answer is not None and v_answer is not None else None
        )
        verdict_indices = field_indices.get("verdict", [])
        verdict_key = verdict.lower()
        features[f"v_verdict_{verdict_key}_mean_logprob"] = _mean_logprob(
            v_trace, verdict_indices
        )
        features[f"v_verdict_{verdict_key}_mean_max_logit"] = _mean_max_logit(
            v_native, verdict_indices
        )
        action_indices = field_indices.get("action", [])
        features["v_action_mean_logprob"] = _mean_logprob(v_trace, action_indices)
        features["v_action_mean_max_logit"] = _mean_max_logit(v_native, action_indices)
        critique_count = len(field_indices.get("critique", []))
        features["v_critique_token_length_log"] = (
            math.log1p(critique_count) if critique_count else None
        )
    features.update(_location_features(location_status))

    features["v_math_expression_mean_logprob"] = _mean_logprob(v_trace, v_math_indices)
    features["v_math_expression_mean_max_logit"] = _mean_max_logit(
        v_native, v_math_indices
    )
    features["v_output_mean_logprob"] = _mean_logprob(v_trace, v_output_indices)
    features["v_output_mean_max_logit"] = _mean_max_logit(v_native, v_output_indices)
    if located:
        features["v_error_location_mean_logprob"] = _mean_logprob(
            v_trace, v_location_indices
        )
        features["v_error_location_mean_max_logit"] = _mean_max_logit(
            v_native, v_location_indices
        )
        features["g_error_location_mean_logprob"] = _mean_logprob(
            g_trace, g_location_indices
        )
        features["g_error_location_mean_max_logit"] = _mean_max_logit(
            g_native, g_location_indices
        )

    return {
        "features": features,
        "missingness": {name: features[name] is None for name in FEATURE_NAMES},
        "v_math_spans": [
            {"start": span.start, "end": span.end, "kind": span.kind}
            for span in v_math_spans
        ],
        "v_math_token_indices": v_math_indices,
        "v_field_token_indices": field_indices,
        "location_match_status": location_status,
        "g_error_location_char_span": (
            None if g_location_span is None else list(g_location_span)
        ),
        "g_error_location_token_indices": g_location_indices,
    }
