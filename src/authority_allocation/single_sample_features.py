"""Nine pre-action scalar features for the single-G/single-V condition."""

from __future__ import annotations

from typing import Any, Mapping


FEATURE_NAMES = (
    "g_answer_extractable",
    "v_verdict_correct",
    "v_verdict_incorrect",
    "v_verdict_uncertain",
    "v_request_keep",
    "v_request_revise",
    "v_request_override",
    "v_candidate_extractable",
    "gv_final_answer_equivalent",
)

VERDICTS = ("correct", "incorrect", "uncertain")
REQUESTS = ("KEEP", "CRITIQUE_REVISE", "OVERRIDE")


def build_single_sample_payload(
    *,
    y0_extractable: bool,
    v_report: Mapping[str, Any] | None,
    candidate_extractable: bool | None,
    answers_equivalent: bool | None,
) -> dict[str, dict[str, Any]]:
    """Build the exact Controller payload without identifiers or outcome data.

    ``v_report=None`` represents the registered V fallback.  That fallback
    requests KEEP, while V-derived semantic fields remain explicitly missing.
    """

    if v_report is None:
        verdict = None
        request = "KEEP"
        candidate_extractable = None
        answers_equivalent = None
    else:
        verdict = v_report.get("verdict")
        request = v_report.get("requested_action")
        if verdict not in VERDICTS:
            raise ValueError(f"invalid verifier verdict: {verdict!r}")
        if request not in REQUESTS:
            raise ValueError(f"invalid verifier request: {request!r}")
        if candidate_extractable is None:
            raise ValueError("candidate_extractable is required for a valid V report")
        if not y0_extractable or not candidate_extractable:
            answers_equivalent = None
        elif answers_equivalent is None:
            raise ValueError("answers_equivalent is required when both answers are extractable")

    features: dict[str, Any] = {
        "g_answer_extractable": int(y0_extractable),
        "v_verdict_correct": None if verdict is None else int(verdict == "correct"),
        "v_verdict_incorrect": None if verdict is None else int(verdict == "incorrect"),
        "v_verdict_uncertain": None if verdict is None else int(verdict == "uncertain"),
        "v_request_keep": int(request == "KEEP"),
        "v_request_revise": int(request == "CRITIQUE_REVISE"),
        "v_request_override": int(request == "OVERRIDE"),
        "v_candidate_extractable": (
            None if candidate_extractable is None else int(candidate_extractable)
        ),
        "gv_final_answer_equivalent": answers_equivalent,
    }
    missingness = {name: features[name] is None for name in FEATURE_NAMES}
    return {"features": features, "missingness": missingness}
