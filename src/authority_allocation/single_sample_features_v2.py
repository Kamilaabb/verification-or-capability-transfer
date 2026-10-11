"""Sixteen registered pre-action features for scalar_single_v2."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .single_sample_features import FEATURE_NAMES as F0_FEATURE_NAMES
from .single_sample_features import build_single_sample_payload


ADDED_FEATURE_NAMES = (
    "g_output_mean_logprob",
    "g_final_answer_mean_logprob",
    "v_verdict_mean_logprob",
    "v_request_mean_logprob",
    "v_candidate_final_answer_mean_logprob",
    "v_critique_token_length_log",
    "gv_answer_type_match",
)
FEATURE_NAMES = F0_FEATURE_NAMES + ADDED_FEATURE_NAMES
LOGPROB_FEATURE_NAMES = ADDED_FEATURE_NAMES[:5]


def _validate_optional_number(name: str, value: Any, *, nonnegative: bool) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or null")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if not nonnegative and value > 1e-12:
        raise ValueError(f"{name} must be a natural log probability <= 0")


def build_single_sample_v2_payload(
    *,
    y0_extractable: bool,
    v_report: Mapping[str, Any] | None,
    candidate_extractable: bool | None,
    answers_equivalent: bool | None,
    g_output_mean_logprob: float | None,
    g_final_answer_mean_logprob: float | None,
    v_verdict_mean_logprob: float | None,
    v_request_mean_logprob: float | None,
    v_candidate_final_answer_mean_logprob: float | None,
    v_critique_token_length_log: float | None,
    gv_answer_type_match: bool | None,
) -> dict[str, dict[str, Any]]:
    """Build the exact v2 Controller payload without identifiers or labels."""

    base = build_single_sample_payload(
        y0_extractable=y0_extractable,
        v_report=v_report,
        candidate_extractable=candidate_extractable,
        answers_equivalent=answers_equivalent,
    )
    for name, value in (
        ("g_output_mean_logprob", g_output_mean_logprob),
        ("g_final_answer_mean_logprob", g_final_answer_mean_logprob),
        ("v_verdict_mean_logprob", v_verdict_mean_logprob),
        ("v_request_mean_logprob", v_request_mean_logprob),
        ("v_candidate_final_answer_mean_logprob", v_candidate_final_answer_mean_logprob),
    ):
        _validate_optional_number(name, value, nonnegative=False)
    _validate_optional_number(
        "v_critique_token_length_log", v_critique_token_length_log, nonnegative=True
    )
    if gv_answer_type_match is not None and not isinstance(gv_answer_type_match, bool):
        raise ValueError("gv_answer_type_match must be boolean or null")

    if not y0_extractable:
        g_final_answer_mean_logprob = None
        gv_answer_type_match = None
    if v_report is None:
        v_verdict_mean_logprob = None
        v_request_mean_logprob = None
        v_candidate_final_answer_mean_logprob = None
        v_critique_token_length_log = None
        gv_answer_type_match = None
    elif not candidate_extractable:
        v_candidate_final_answer_mean_logprob = None
        gv_answer_type_match = None

    added = {
        "g_output_mean_logprob": g_output_mean_logprob,
        "g_final_answer_mean_logprob": g_final_answer_mean_logprob,
        "v_verdict_mean_logprob": v_verdict_mean_logprob,
        "v_request_mean_logprob": v_request_mean_logprob,
        "v_candidate_final_answer_mean_logprob": v_candidate_final_answer_mean_logprob,
        "v_critique_token_length_log": v_critique_token_length_log,
        "gv_answer_type_match": gv_answer_type_match,
    }
    features = {**base["features"], **added}
    missingness = {name: features[name] is None for name in FEATURE_NAMES}
    return {"features": features, "missingness": missingness}


def mathematical_answer_type(parsed_value: Any) -> str | None:
    """Return a conservative, implementation-stable mathematical type."""

    if parsed_value is None:
        return None
    class_name = type(parsed_value).__name__.lower()
    module_name = type(parsed_value).__module__.lower()
    if "matrix" in class_name:
        return "matrix"
    if "interval" in class_name:
        return "interval"
    if "set" in class_name:
        return "set"
    if "tuple" in class_name or isinstance(parsed_value, tuple):
        return "tuple"
    if module_name.startswith("sympy"):
        if bool(getattr(parsed_value, "is_Integer", False)):
            return "integer"
        if bool(getattr(parsed_value, "is_Rational", False)):
            return "rational"
        if bool(getattr(parsed_value, "is_number", False)):
            return "scalar_number"
        if bool(getattr(parsed_value, "is_Relational", False)):
            return "relation"
        if hasattr(parsed_value, "free_symbols"):
            return "symbolic_expression"
    return None
