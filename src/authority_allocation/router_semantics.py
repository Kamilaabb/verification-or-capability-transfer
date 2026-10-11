"""Executable `router_v1` semantics: two arms, no authority cap, cost as an outcome.

`authority_v6`'s `semantics.effective_action` is `min(cap, request)`, which lets the
controller only veto the verifier. That is deliberately absent here. A row has exactly
two deployable outcomes, `a0` (never self-check) and `a1` (always self-check), and a
router is any rule choosing between them per row.

The functions below are the ones that must never drift: the pass-2 failure fallback,
and the joint accuracy/cost value of an arbitrary routing decision. Everything a policy
comparison needs is arithmetic over an archived counterfactual file, so a new router is
scored without rerunning a single model call.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

FAILED_SELF_CHECK = frozenset({"model_invalid", "call_error", "fallback"})


def deployed_outcomes(
    *, u0: Optional[int], u_self: Optional[int], self_check_status: str
) -> dict[str, Optional[int]]:
    """Return `{a0, a1}` under the registered failure policy.

    A failed pass 2 keeps `u_self` null, because nothing was graded, but the deployed
    answer falls back to y0, so `a1` is `u0` there. Recording the failure as a wrong
    answer instead would charge the always-self-check arm for an outage rather than for
    a judgement, and would make the harm pool depend on serving reliability.
    """
    if u0 is not None and u0 not in (0, 1):
        raise ValueError(f"u0 must be 0, 1 or None, got {u0!r}")
    if u_self is not None and u_self not in (0, 1):
        raise ValueError(f"u_self must be 0, 1 or None, got {u_self!r}")

    if self_check_status in FAILED_SELF_CHECK:
        return {"a0": u0, "a1": u0}
    if u_self is None:
        # Graded-but-null cannot happen for a successful call; the runner must not emit it.
        raise ValueError("u_self is null but self_check_status reports a successful call")
    return {"a0": u0, "a1": u_self}


def pool_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Repair and harm pools for A1 against A0, plus the rows a router can act on.

    These are the quantities Gate 0 exists to measure. `harm` bounds everything a router
    can ever win on accuracy: skipping a row forgoes its repair and avoids its harm, so
    a perfect router scores `a0 + harm` and nothing beats that.
    """
    repair = harm = both_right = both_wrong = ungraded = 0
    for row in rows:
        a0, a1 = row["a0"], row["a1"]
        if a0 is None or a1 is None:
            ungraded += 1
        elif a0 == 0 and a1 == 1:
            repair += 1
        elif a0 == 1 and a1 == 0:
            harm += 1
        elif a0 == 1:
            both_right += 1
        else:
            both_wrong += 1
    return {
        "repair": repair,
        "harm": harm,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "ungraded": ungraded,
        "decision_relevant": repair + harm,
    }


def best_fixed_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which fixed arm is best, and by how much. A router must be reported against it.

    Beating the weaker arm is not a result: A0 and A1 are both free to choose, so the
    bar is `max(A0, A1)`.
    """
    arms = fixed_arm_values(rows)
    a0, a1 = arms["A0_never_self_check"], arms["A1_always_self_check"]
    name = "A1_always_self_check" if a1["correct"] > a0["correct"] else (
        "A0_never_self_check" if a0["correct"] > a1["correct"] else "tie")
    return {"name": name, "correct": max(a0["correct"], a1["correct"]),
            "a0_correct": a0["correct"], "a1_correct": a1["correct"]}


def oracle_headroom(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Accuracy a perfect router can add over the *best* fixed arm: `min(R, H)`.

    A perfect router takes A1 on every repair row and A0 on every harm row, scoring
    `both_right + R + H`, while the best fixed arm scores `both_right + max(R, H)`.
    The difference is `min(R, H)`, not `H`: `H` is only the answer when A1 already wins,
    and when A0 wins the router's entire headroom is the repairs it can pick up instead.

    The balanced case is the important one. At `R == H` the two fixed arms tie, so the
    average effect of self-checking is zero, yet the headroom is at its largest for that
    pool size. An average effect near zero is therefore evidence that routing may be
    worth the most, never evidence that it is worth nothing: what a router exploits is
    heterogeneity across rows, not the mean.
    """
    counts = pool_counts(rows)
    repair, harm = counts["repair"], counts["harm"]
    return {
        "repair": repair,
        "harm": harm,
        "decision_relevant": repair + harm,
        "headroom_over_best_fixed": min(repair, harm),
        "headroom_over_a1": harm,
        "headroom_over_a0": repair,
    }


def policy_value(
    rows: Sequence[Mapping[str, Any]], called: Iterable[str]
) -> dict[str, Any]:
    """Score an arbitrary routing decision on accuracy and on tokens.

    `called` names the rows the router spends pass 2 on. Pass 1 is unconditional, so its
    tokens are paid on every row; pass 2's tokens are paid only where the router called.
    This is the function that makes cost a first-class outcome: it needs no model, no
    fit and no rerun, and a token count carries no sampling error.
    """
    called_set = set(called)
    known = {row["problem_id"] for row in rows}
    unknown = called_set - known
    if unknown:
        raise KeyError(f"router called {len(unknown)} rows absent from the archive: {sorted(unknown)[:5]}")

    correct = tokens = 0
    ungraded = 0
    for row in rows:
        on = row["problem_id"] in called_set
        outcome = row["a1"] if on else row["a0"]
        if outcome is None:
            ungraded += 1
        else:
            correct += outcome
        tokens += row["tokens_pass1"] or 0
        if on:
            tokens += row["tokens_pass2"] or 0
    n = len(rows)
    return {
        "rows": n,
        "called": len(called_set),
        "call_fraction": len(called_set) / n if n else 0.0,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "ungraded": ungraded,
        "output_tokens": tokens,
        "tokens_per_correct": tokens / correct if correct else None,
    }


def fixed_arm_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The two fixed baselines a router must be reported against."""
    return {
        "A0_never_self_check": policy_value(rows, []),
        "A1_always_self_check": policy_value(rows, [r["problem_id"] for r in rows]),
    }


def logit_identity_error(positions: Iterable[Mapping[str, float]]) -> float:
    """Max `|sampled_logprob - (max_logit - logsumexp)|` over a teacher-forced trace.

    Under greedy decoding the sampled token is the argmax, so the identity holds exactly
    up to numerical error. A large value means the captured distribution is not the one
    that produced the archived output, which invalidates every R2 feature built from it.
    """
    worst = 0.0
    for position in positions:
        error = abs(
            position["sampled_logprob"] - (position["max_logit"] - position["logsumexp"])
        )
        worst = max(worst, error)
    return worst


def bootstrap_policy_ci(
    rows: Sequence[Mapping[str, Any]],
    called: Iterable[str],
    *,
    baseline: Optional[Iterable[str]] = None,
    seed: int,
    repetitions: int = 10000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Problem-level bootstrap CIs for a router's accuracy AND token differences.

    Cost is measured, not estimated, so for *these* rows the token total carries no
    estimation error. A claim about a population of problems is a different matter: a
    fresh draw changes which rows the router calls and how long the answers run, so the
    saving needs an interval like any other statistic. Resampling problems with
    replacement gives both.

    Cost is nonetheless the better-powered outcome here, and the reason is structural
    rather than a claim of exactness: the accuracy difference is carried only by the
    `R + H` rows where the arms disagree, while every row contributes its pass-2 tokens
    to the cost difference. The effective sample size is the pool for one and the whole
    manifest for the other.
    """
    import random

    called_set = set(called)
    baseline_set = set(baseline) if baseline is not None else set()
    index = {row["problem_id"]: row for row in rows}
    if not index:
        raise ValueError("no rows to bootstrap")

    def score(sample: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
        policy = policy_value(sample, [p for p in called_set if p in {r["problem_id"] for r in sample}])
        base = policy_value(sample, [p for p in baseline_set if p in {r["problem_id"] for r in sample}])
        return (policy["accuracy"] - base["accuracy"],
                policy["output_tokens"] - base["output_tokens"])

    point = score(list(rows))
    rng = random.Random(seed)
    ids = [row["problem_id"] for row in rows]
    acc_draws, tok_draws = [], []
    for _ in range(repetitions):
        sample = [index[rng.choice(ids)] for _ in ids]
        a, t = score(sample)
        acc_draws.append(a)
        tok_draws.append(t)
    acc_draws.sort()
    tok_draws.sort()
    lo_i = int(alpha / 2 * repetitions)
    hi_i = min(repetitions - 1, int((1 - alpha / 2) * repetitions))
    return {
        "repetitions": repetitions,
        "accuracy_difference": point[0],
        "accuracy_ci": [acc_draws[lo_i], acc_draws[hi_i]],
        "token_difference": point[1],
        "token_ci": [tok_draws[lo_i], tok_draws[hi_i]],
    }
