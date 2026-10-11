"""Scalar evidence for the BigCodeBench code study (feature contract code_solve_scalar_v1).

Written after the model call and before grading. The output statistics are the ones the
vision and knowledge studies compute (vision_mc_features._output_stats, reused rather than
copied so every domain describes an output identically); the answer-token block is replaced
by a count of fenced Python blocks, because a program has no single answer token.
"""

from __future__ import annotations

from typing import Any, Mapping

from authority_allocation.bigcodebench_study import python_block_count
from authority_allocation.math_expression_spans import content_token_indices
from authority_allocation.vision_mc_features import _output_stats

FEATURE_CONTRACT = "code_solve_scalar_v1"
FEATURES = ("problem_char_count", "prompt_tokens", "output_token_count", "truncated",
            "python_block_count",
            "out_lp_mean", "out_lp_min", "out_lp_p10", "out_lp_std", "out_ent_mean",
            "out_ent_p90", "out_ent_max", "out_margin_mean", "out_margin_p10",
            "out_max_logit_mean", "out_lowconf_frac", "out_lowconf_run")


def derive_features(*, output: Mapping[str, Any], native: Mapping[str, Any],
                    problem_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(features, audit) for one output. An output statistic is None when its tokens cannot
    all be paired with the native trace, never a partial aggregate."""
    text = output.get("raw_output") or ""
    trace = output.get("token_logprobs") or {}
    usage = output.get("usage") or {}
    by_index = {int(p["index"]): p for p in native.get("positions", [])}
    content = content_token_indices(trace)
    positions = [by_index[i] for i in content if i in by_index]
    if len(positions) != len(content):
        positions = []
    features: dict[str, Any] = {
        "problem_char_count": len(problem_text),
        "prompt_tokens": usage.get("input_tokens"),
        "output_token_count": usage.get("output_tokens"),
        "truncated": int(usage.get("finish_reason") == "length"),
        "python_block_count": python_block_count(text),
        **_output_stats(positions),
    }
    if tuple(features) != FEATURES:
        raise AssertionError("feature order drifted from the contract")
    audit = {"content_token_count": len(content), "paired_token_count": len(positions)}
    return features, audit
