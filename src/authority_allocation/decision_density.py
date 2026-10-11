"""Decision-relevant density and paired power arithmetic over counterfactual archives.

Every function here is exact arithmetic on archived `(z0, z1, z2)` outcomes. Nothing
fits a model, selects a threshold or touches a manifest, so results computed with this
module are of the same kind as the repair/harm tallies already reported in
`docs/analysis/oversight_cost_accounting.md`: they stand or fall on the archive alone.

The module exists because c013 made one limit concrete. A condition's nominal row count
is not its sample size for an authority question. Rows where all three caps agree carry
no information about which cap to grant, and the M1-vs-M0 contrast is decided entirely
by the rows where those two disagree. On c013 that is 25 of 300, which bounds what the
run could have detected no matter what the rewriter did.
"""

from __future__ import annotations

from collections import Counter
from math import comb, sqrt
from typing import Any, Iterable, Mapping

CAP_KEYS = ("z0", "z1", "z2")


def load_cap_outcomes(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[int, int, int]]:
    """Index counterfactual records by `problem_id` as `(z0, z1, z2)` triples.

    A null cap outcome means the row was never resolved into a labelled counterfactual,
    so it cannot enter a cap comparison and is rejected rather than silently dropped:
    a validated archive has none, and one that does is a different object than the
    analyses below assume.
    """
    outcomes: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        problem_id = row["problem_id"]
        if problem_id in outcomes:
            raise ValueError(f"duplicate problem_id in counterfactual archive: {problem_id}")
        triple = tuple(row[key] for key in CAP_KEYS)
        if any(value not in (0, 1) for value in triple):
            raise ValueError(f"unresolved cap outcome for {problem_id}: {triple}")
        outcomes[problem_id] = triple  # type: ignore[assignment]
    return outcomes


def restrict(
    outcomes: Mapping[str, tuple[int, int, int]],
    problem_ids: Iterable[str],
) -> dict[str, tuple[int, int, int]]:
    """Keep exactly `problem_ids`, requiring every one of them to be present.

    The caller supplies the row set it wants compared. Quietly intersecting instead
    would let a partially overlapping archive produce a table that reads like a paired
    comparison without being one.
    """
    wanted = list(problem_ids)
    missing = [pid for pid in wanted if pid not in outcomes]
    if missing:
        raise KeyError(f"{len(missing)} requested rows absent from archive: {missing[:5]}")
    return {pid: outcomes[pid] for pid in wanted}


def pattern_counts(outcomes: Mapping[str, tuple[int, int, int]]) -> dict[str, int]:
    """Count all eight `(z0,z1,z2)` patterns, including the zeros."""
    counts = Counter("".join(str(value) for value in triple) for triple in outcomes.values())
    return {
        f"{a}{b}{c}": counts.get(f"{a}{b}{c}", 0)
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    }


def cap_correct(outcomes: Mapping[str, tuple[int, int, int]], cap: str) -> int:
    index = CAP_KEYS.index(cap)
    return sum(1 for triple in outcomes.values() if triple[index] == 1)


def margin(
    outcomes: Mapping[str, tuple[int, int, int]], lower: str, upper: str
) -> tuple[int, int]:
    """Return `(repair, harm)` for granting `upper` instead of `lower`.

    Repair is a row the lower cap gets wrong and the upper gets right; harm is the
    reverse. Both are only definable where the two caps disagree, which is the whole
    point: their sum is the contrast's real sample size.
    """
    low = CAP_KEYS.index(lower)
    high = CAP_KEYS.index(upper)
    repair = sum(1 for t in outcomes.values() if t[low] == 0 and t[high] == 1)
    harm = sum(1 for t in outcomes.values() if t[low] == 1 and t[high] == 0)
    return repair, harm


def decision_relevant_rows(outcomes: Mapping[str, tuple[int, int, int]]) -> int:
    """Rows where the three caps do not all agree.

    Rows where they agree are structurally void for an authority question: no policy
    over them can change the score, so they inflate the nominal N without adding power.
    """
    return sum(1 for triple in outcomes.values() if len(set(triple)) > 1)


def mcnemar_exact_p(repair: int, harm: int) -> float:
    """Two-sided exact McNemar p-value for a paired binary contrast.

    Conditional on the discordant count, repairs are Binomial(n, 0.5) under the null
    that the extra authority changes nothing. The exact test is used rather than the
    chi-square approximation because these counts are routinely under 30.
    """
    total = repair + harm
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(repair, harm) + 1))
    return min(1.0, 2 * tail / 2**total)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which stays inside [0,1] at the small counts used here."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (centre - half, centre + half)


def accuracy_difference_interval(
    repair: int, harm: int, rows: int, z: float = 1.96
) -> tuple[float, float]:
    """95% CI for the paired accuracy difference `(repair - harm) / rows`.

    The discordant pairs carry all the uncertainty, so the interval is the Wilson
    interval on the repair share mapped through `(2p - 1) * n_discordant / rows`.
    """
    discordant = repair + harm
    if rows == 0:
        raise ValueError("rows must be positive")
    if discordant == 0:
        return (0.0, 0.0)
    low, high = wilson_interval(repair, discordant, z=z)
    scale = discordant / rows
    return ((2 * low - 1) * scale, (2 * high - 1) * scale)


def newcombe_paired_interval(
    both: int, a_only: int, b_only: int, neither: int, z: float = 1.96
) -> tuple[float, float]:
    """Newcombe (1998) method 10 for the unconditional paired difference `(a_only - b_only) / n`.

    Unlike `accuracy_difference_interval`, which holds the discordant count fixed, this
    interval also carries the uncertainty in how many questions are discordant, so it keeps
    its coverage when the difference is large (statistics audit, 2026-10-05).
    """
    n = both + a_only + b_only + neither
    if n == 0:
        raise ValueError("no questions")
    p1, p2 = (both + a_only) / n, (both + b_only) / n
    l1, u1 = wilson_interval(both + a_only, n, z=z)
    l2, u2 = wilson_interval(both + b_only, n, z=z)
    den = sqrt((both + a_only) * (b_only + neither) * (both + b_only) * (a_only + neither))
    phi = 0.0 if den == 0 else (both * neither - a_only * b_only) / den
    diff = p1 - p2
    return (diff - sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0)),
            diff + sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0)))


def minimum_detectable_net(discordant: int, alpha: float = 0.05) -> int | None:
    """Smallest `|repair - harm|` that would reach `alpha` at this discordant count.

    Returns None when no split of `discordant` pairs can reach significance, which is
    the honest answer for a contrast with very few discordant rows: such a design
    cannot produce a significant result in either direction and its null finding says
    nothing about the effect.
    """
    for net in range(discordant % 2, discordant + 1, 2):
        if net == 0:
            continue
        repair = (discordant + net) // 2
        if mcnemar_exact_p(repair, discordant - repair) < alpha:
            return net
    return None


def two_sample_repair_share_z(
    repair_a: int, harm_a: int, repair_b: int, harm_b: int
) -> tuple[float, float]:
    """Pooled two-proportion z for two conditions' repair shares, with its p-value.

    This compares the *direction* of intervention across conditions while their
    discordance *rates* may be equal. It is an unpaired comparison of two archives and
    inherits every confound between them, so it describes, it does not attribute.
    """
    n_a, n_b = repair_a + harm_a, repair_b + harm_b
    if n_a == 0 or n_b == 0:
        raise ValueError("both conditions need discordant rows")
    pooled = (repair_a + repair_b) / (n_a + n_b)
    if pooled in (0.0, 1.0):
        return (0.0, 1.0)
    z = (repair_a / n_a - repair_b / n_b) / sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    return (z, _normal_two_sided_p(z))


def _normal_two_sided_p(z: float) -> float:
    from math import erfc

    return erfc(abs(z) / sqrt(2))


def contrast_report(
    outcomes: Mapping[str, tuple[int, int, int]], lower: str, upper: str
) -> dict[str, Any]:
    """Everything the repository reports about one cap contrast, as plain arithmetic."""
    repair, harm = margin(outcomes, lower, upper)
    rows = len(outcomes)
    discordant = repair + harm
    low, high = accuracy_difference_interval(repair, harm, rows)
    return {
        "contrast": f"{upper}-{lower}",
        "rows": rows,
        "repair": repair,
        "harm": harm,
        "net": repair - harm,
        "discordant": discordant,
        "discordant_share": discordant / rows,
        "accuracy_difference": (repair - harm) / rows,
        "accuracy_difference_ci95": [low, high],
        "mcnemar_exact_p": mcnemar_exact_p(repair, harm),
        "minimum_detectable_net": minimum_detectable_net(discordant),
    }
