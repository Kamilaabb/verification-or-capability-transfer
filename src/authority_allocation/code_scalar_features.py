"""Registered pre-action scalar features for the BigCodeBench controller.

The Controller receives only the returned numbers and missingness flags.  Raw problem,
program, and critique text stay in source records and are never placed in its payload.
No generated code is executed while computing these features.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Mapping

from .generated_token_logprobs import derive_scalar_logprob_features_v3


S0_SELF_REPORT = (
    "v_verdict_correct",
    "v_verdict_incorrect",
    "v_verdict_uncertain",
    "v_request_keep",
    "v_request_revise",
    "v_request_override",
)
S1_CONFIDENCE_AND_DIFFICULTY = (
    "g_output_mean_logprob",
    "v_verdict_mean_logprob",
    "v_request_mean_logprob",
    "v_candidate_answer_mean_logprob",
    "problem_chars_log",
    "g_code_lexemes_log",
    "critique_words_log",
)
S2_PROGRAM_AND_CRITIQUE_CONTENT = (
    "gv_unique_call_names_log_abs_delta",
    "gv_control_flow_log_abs_delta",
    "gv_literals_log_abs_delta",
    "critique_category_interface",
    "critique_category_edge_case",
    "critique_category_api_library",
    "critique_category_logic_algorithm",
    "critique_category_robustness",
)
FEATURE_NAMES = (
    S0_SELF_REPORT + S1_CONFIDENCE_AND_DIFFICULTY + S2_PROGRAM_AND_CRITIQUE_CONTENT
)

FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
LEXEME = re.compile(
    r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|:=|\*\*|//|->|"
    r"[-+*/%<>=]|[()[\]{},.:]"
)
WORD = re.compile(r"[A-Za-z_]\w*")

CRITIQUE_CATEGORIES = {
    "interface": (
        "signature", "parameter", "argument", "return", "entry point", "format",
        "function name", "type hint",
    ),
    "edge_case": (
        "edge case", "empty", "boundary", "zero", "negative", "duplicate", "none",
        "null", "missing", "single element",
    ),
    "api_library": (
        "api", "library", "import", "module", "method", "function call", "attribute",
        "package",
    ),
    "logic_algorithm": (
        "logic", "algorithm", "condition", "incorrect", "wrong", "calculation",
        "index", "iteration", "loop", "order",
    ),
    "robustness": (
        "exception", "error handling", "try", "except", "validation", "invalid input",
    ),
}


def _log1p(value: int | float) -> float:
    return math.log1p(max(float(value), 0.0))


def _source(text: str | None) -> str:
    if not text:
        return ""
    blocks = FENCE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def _tree(text: str) -> ast.AST | None:
    if not text:
        return None
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        return None


def _program_counts(text: str) -> tuple[int, int, int]:
    tree = _tree(_source(text))
    if tree is None:
        return 0, 0, 0
    nodes = list(ast.walk(tree))
    control_flow = sum(
        isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.Try,
                ast.With,
                ast.AsyncWith,
                ast.Match,
                ast.comprehension,
            ),
        )
        for node in nodes
    )
    call_names = set()
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            call_names.add(node.func.attr)
    literals = sum(isinstance(node, ast.Constant) for node in nodes)
    return len(call_names), control_flow, literals


def build_code_scalar_payload(
    *,
    problem_text: str,
    g_text: str,
    g_trace: Mapping[str, Any],
    v_text: str,
    v_trace: Mapping[str, Any],
    v_report: Mapping[str, Any] | None,
    v_report_parse_status: str,
    registered_request: str,
) -> dict[str, dict[str, Any]]:
    """Return the exact code_scalar_v1 feature and missingness objects."""

    derived = derive_scalar_logprob_features_v3(
        g_text=g_text,
        g_trace=g_trace,
        v_text=v_text,
        v_trace=v_trace,
        v_report_parse_status=v_report_parse_status,
    )
    report_missing = v_report is None
    verdict = None if report_missing else str(v_report["verdict"])
    critique = None if report_missing else str(v_report["critique"])
    candidate = None if report_missing else str(v_report["candidate_answer"])

    features: dict[str, Any] = {
        "v_verdict_correct": None if verdict is None else int(verdict == "correct"),
        "v_verdict_incorrect": None if verdict is None else int(verdict == "incorrect"),
        "v_verdict_uncertain": None if verdict is None else int(verdict == "uncertain"),
        "v_request_keep": int(registered_request == "KEEP"),
        "v_request_revise": int(registered_request == "CRITIQUE_REVISE"),
        "v_request_override": int(registered_request == "OVERRIDE"),
        "g_output_mean_logprob": derived["g_output_mean_logprob"],
        "v_verdict_mean_logprob": derived["v_verdict_mean_logprob"],
        "v_request_mean_logprob": derived["v_request_mean_logprob"],
        "v_candidate_answer_mean_logprob": derived["v_candidate_answer_mean_logprob"],
        "problem_chars_log": _log1p(len(problem_text)),
        "g_code_lexemes_log": _log1p(len(LEXEME.findall(_source(g_text)))),
        "critique_words_log": None if critique is None else _log1p(len(WORD.findall(critique))),
    }

    if candidate is None or critique is None:
        features.update(
            {name: None for name in S2_PROGRAM_AND_CRITIQUE_CONTENT}
        )
    else:
        g_calls, g_control, g_literals = _program_counts(g_text)
        v_calls, v_control, v_literals = _program_counts(candidate)
        lower = critique.lower()
        features.update(
            {
                "gv_unique_call_names_log_abs_delta": abs(
                    _log1p(g_calls) - _log1p(v_calls)
                ),
                "gv_control_flow_log_abs_delta": abs(
                    _log1p(g_control) - _log1p(v_control)
                ),
                "gv_literals_log_abs_delta": abs(
                    _log1p(g_literals) - _log1p(v_literals)
                ),
                **{
                    f"critique_category_{category}": int(
                        any(phrase in lower for phrase in phrases)
                    )
                    for category, phrases in CRITIQUE_CATEGORIES.items()
                },
            }
        )

    if tuple(features) != FEATURE_NAMES:
        raise AssertionError("code_scalar_v1 feature membership or order drifted")
    return {
        "features": features,
        "missingness": {name: features[name] is None for name in FEATURE_NAMES},
    }


__all__ = [
    "CRITIQUE_CATEGORIES",
    "FEATURE_NAMES",
    "S0_SELF_REPORT",
    "S1_CONFIDENCE_AND_DIFFICULTY",
    "S2_PROGRAM_AND_CRITIQUE_CONTENT",
    "build_code_scalar_payload",
]
