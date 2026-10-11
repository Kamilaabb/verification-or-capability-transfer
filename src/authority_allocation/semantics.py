"""Small, dependency-free implementation of the frozen protocol semantics.

This module intentionally does not contain model-serving code.  Its job is to
make the few rules that must never drift executable and testable.
"""

from __future__ import annotations

from enum import IntEnum
import hashlib
from typing import Any, Iterable, Mapping, Optional


class Action(IntEnum):
    KEEP = 0
    CRITIQUE_REVISE = 1
    OVERRIDE = 2


class AuthorityCap(IntEnum):
    M0 = 0
    M1 = 1
    M2 = 2


def effective_action(cap: AuthorityCap | str, request: Action | str) -> Action:
    """Return the only registered execution rule: min(cap, request)."""

    parsed_cap = cap if isinstance(cap, AuthorityCap) else AuthorityCap[cap]
    parsed_request = request if isinstance(request, Action) else Action[request]
    return Action(min(int(parsed_cap), int(parsed_request)))


def select_y0(outputs: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select sample index 0, even when it is incomplete or unparseable.

    Raising on a missing or duplicate index 0 is safer than silently changing
    the estimand by selecting the first usable completion.
    """

    matches = [row for row in outputs if row.get("sample_index") == 0]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one sample_index=0, found {len(matches)}")
    return matches[0]


def counterfactual_outcomes(
    request: Action | str,
    *,
    u0: Optional[int],
    u1: Optional[int],
    u2: Optional[int],
    rewrite_status: str,
) -> dict[str, Optional[int]]:
    """Build z0/z1/z2 under the protocol's failure policy.

    A failed/not-run rewrite is recorded as ``u1=None`` but an attempted M1
    path falls back to KEEP, so its deployable outcome is ``u0``.
    """

    parsed_request = request if isinstance(request, Action) else Action[request]
    outcomes: dict[str, Optional[int]] = {}
    for cap in AuthorityCap:
        action = effective_action(cap, parsed_request)
        if action is Action.KEEP:
            value = u0
        elif action is Action.CRITIQUE_REVISE:
            value = u1 if rewrite_status == "ok" else u0
        else:
            value = u2
        outcomes[f"z{int(cap)}"] = value
    return outcomes


def derive_call_seed(
    replicate_plan_id: str,
    scope: str,
    replicate_id: str,
    base_seed: int,
    cell_id: str,
    problem_id: str,
    sample_index: int,
) -> int:
    """Derive a stable 31-bit call seed from the registered ordered fields.

    The controller menu is deliberately not an input, which pairs C2 and C3.
    """

    parts = (
        replicate_plan_id,
        scope,
        replicate_id,
        str(base_seed),
        cell_id,
        problem_id,
        str(sample_index),
    )
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def paired_policy_metrics(baseline: Iterable[int], policy: Iterable[int]) -> dict[str, float | int]:
    """Compute the primary paired metrics on one common, fully graded set."""

    baseline_values = list(baseline)
    policy_values = list(policy)
    if len(baseline_values) != len(policy_values) or not baseline_values:
        raise ValueError("baseline and policy must be non-empty and have equal length")
    if any(value not in (0, 1) for value in baseline_values + policy_values):
        raise ValueError("main paired metrics require only fully graded binary outcomes")

    n = len(baseline_values)
    accuracy_m0 = sum(baseline_values) / n
    accuracy_policy = sum(policy_values) / n
    repair = sum(b == 0 and p == 1 for b, p in zip(baseline_values, policy_values)) / n
    harm = sum(b == 1 and p == 0 for b, p in zip(baseline_values, policy_values)) / n
    delta_accuracy = accuracy_policy - accuracy_m0
    if abs(delta_accuracy - (repair - harm)) > 1e-12:
        raise AssertionError("paired repair-harm identity violated")
    return {
        "n": n,
        "accuracy_m0": accuracy_m0,
        "accuracy_policy": accuracy_policy,
        "delta_accuracy": delta_accuracy,
        "repair": repair,
        "harm": harm,
    }
