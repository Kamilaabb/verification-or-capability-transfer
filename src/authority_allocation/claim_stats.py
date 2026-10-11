"""Standard tests for claims stated over many model pairs and several domains.

A claim such as "the stronger model generating and the weaker reviewing beats the
reverse" is a contrast between two systems answered on the same questions. Counting
how many pairs go one way is the sign test, and it is a valid test, but it alone says
nothing about how large the difference is, whether a single pair's difference is
noise, or how much the count leans on one model that sits in many pairs. This module
reports those four things side by side so that no claim rests on a bare count.

Everything uses a fixed seed, so it gives the same numbers on every run. Nothing here
reads a manifest or a grade file; callers pass 0/1 vectors that are already aligned by
question.

Unit
    One paired contrast: `a` and `b` are the 0/1 correctness of the two systems on the
    same questions in the same order. `models` names every model in either system and
    is used by the leave-one-model-out check and to group the two directions of a pair.
    `domain` groups units that share a question set, so the bootstrap resamples those
    questions jointly.

Inference (statistics audit, docs/analysis/stats_audit_20261005, approved 2026-10-05).
Models, pairs and decoding are fixed and greedy, so the intervals quantify the uncertainty
from which questions were sampled, under that fixed design:

    one unit       paired question bootstrap (`paired_unit`)
    mean of units  units fixed, one shared question resample per domain (`resample="fixed"`)
    relations      model pairs resampled with both directions together, plus shared questions
                   (`resample="pairs"`), for statistics that compare pairs, such as a slope

Questions are resampled within fixed strata: the dev/test split everywhere, and the
constituent dataset in nl_reasoning (ReClor, FOLIO, GPQA), so a benchmark's composition is
the same in every replicate. `resample="units"` reproduces the pre-audit bootstrap.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from math import comb
from typing import Iterable, Mapping, Sequence

import numpy as np

from authority_allocation.decision_density import (
    accuracy_difference_interval,
    mcnemar_exact_p,
    newcombe_paired_interval,
)

# Domains whose question set joins several datasets in fixed numbers; their problem ids
# are "<dataset>/<id>" (scripts/build_nlreason_manifests.py).
SOURCE_STRATA_DOMAINS = frozenset({"nl_reasoning"})
RESAMPLE_MODES = ("fixed", "pairs", "units")


def question_stratum(domain: str, key) -> tuple:
    """The fixed stratum a question is resampled within: its split, and its dataset in
    domains listed in SOURCE_STRATA_DOMAINS. Keys are (split, problem_id) or a bare id."""
    split, pid = key if isinstance(key, tuple) and len(key) == 2 else (None, key)
    source = ""
    if domain in SOURCE_STRATA_DOMAINS and "/" in str(pid):
        source = str(pid).split("/", 1)[0]
    return (split, source)


def _strata_of(domain: str, keys, strata, n: int) -> tuple:
    if strata is not None:
        if len(strata) != n:
            raise ValueError("strata must match the vectors")
        return tuple(strata)
    if keys is None:
        return (("", ""),) * n
    return tuple(question_stratum(domain, k) for k in keys)


@dataclass(frozen=True)
class Unit:
    id: str
    domain: str
    a: Sequence[int]
    b: Sequence[int]
    models: tuple[str, ...] = field(default_factory=tuple)
    keys: tuple | None = None  # question identities; units of a domain without keys must share one question list
    strata: tuple | None = None  # per-question stratum; derived from keys with question_stratum when None

    def __post_init__(self) -> None:
        if len(self.a) != len(self.b) or not self.a:
            raise ValueError(f"{self.id}: empty or unequal vectors")
        if self.keys is not None and (len(self.keys) != len(self.a) or len(set(self.keys)) != len(self.keys)):
            raise ValueError(f"{self.id}: keys must be unique and match the vectors")
        if any(x not in (0, 1) for x in (*self.a, *self.b)):
            raise ValueError(f"{self.id}: non-binary outcome")
        if self.strata is not None and len(self.strata) != len(self.a):
            raise ValueError(f"{self.id}: strata must match the vectors")

    def stratum_labels(self) -> tuple:
        return _strata_of(self.domain, self.keys, self.strata, self.n)

    @property
    def n(self) -> int:
        return len(self.a)

    @property
    def diff_pp(self) -> float:
        return 100 * (sum(self.a) - sum(self.b)) / self.n


def binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided p for k successes in n fair coin flips (doubling the smaller tail)."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def sign_test(diffs: Iterable[float]) -> dict:
    """Exact sign test over unit-level differences. Exact zeros are ties and are dropped."""
    values = list(diffs)
    wins = sum(d > 0 for d in values)
    losses = sum(d < 0 for d in values)
    return {"units": len(values), "wins": wins, "losses": losses,
            "ties": len(values) - wins - losses,
            "p_two_sided": binomial_two_sided_p(wins, wins + losses)}


def _unit_seed(seed: int, unit_id: str) -> int:
    """A per-unit seed that does not depend on the order units are evaluated in."""
    return (seed * 1_000_003 + zlib.crc32(unit_id.encode("utf-8"))) % 2**32


def paired_bootstrap_interval(a: Sequence[int], b: Sequence[int], strata: Sequence, reps: int = 2000,
                              seed: int = 20261003) -> tuple[float, float]:
    """Percentile 95% interval (pp) for mean(a) - mean(b), resampling questions within strata.

    Only the per-stratum counts of a-only, b-only and tied questions matter, so each stratum
    is one multinomial draw per replicate, which is exactly resampling its questions.
    """
    cells: dict = {}
    for x, y, s in zip(a, b, strata):
        c = cells.setdefault(s, [0, 0, 0])
        c[0 if x > y else 1 if x < y else 2] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    net = np.zeros(reps)
    for c in cells.values():
        m = sum(c)
        draws = rng.multinomial(m, [c[0] / m, c[1] / m, c[2] / m], size=reps)
        net += draws[:, 0] - draws[:, 1]
    diffs = np.sort(100 * net / len(a))
    return (float(_quantile(diffs, 0.025)), float(_quantile(diffs, 0.975)))


def paired_unit(unit: Unit, reps: int = 2000, seed: int = 20261003) -> dict:
    """Within-unit paired comparison: exact McNemar and the paired accuracy-difference CI.

    `ci95_pp` is the paired question bootstrap, stratified as `unit.stratum_labels()`.
    Newcombe's analytic interval is reported beside it as a check, and the pre-audit
    interval (Wilson on the discordant share, discordant count held fixed) for comparison.
    """
    a_only = sum(x > y for x, y in zip(unit.a, unit.b))
    b_only = sum(x < y for x, y in zip(unit.a, unit.b))
    both = sum(x == y == 1 for x, y in zip(unit.a, unit.b))
    low, high = paired_bootstrap_interval(unit.a, unit.b, unit.stratum_labels(), reps,
                                          _unit_seed(seed, unit.id))
    n_low, n_high = newcombe_paired_interval(both, a_only, b_only, unit.n - both - a_only - b_only)
    c_low, c_high = accuracy_difference_interval(a_only, b_only, unit.n)
    return {"id": unit.id, "domain": unit.domain, "n": unit.n,
            "a_correct": sum(unit.a), "b_correct": sum(unit.b),
            "a_only": a_only, "b_only": b_only, "diff_pp": unit.diff_pp,
            "ci95_pp": [low, high],
            "ci95_pp_newcombe": [100 * n_low, 100 * n_high],
            "ci95_pp_pre_audit": [100 * c_low, 100 * c_high],
            "mcnemar_p": mcnemar_exact_p(a_only, b_only)}


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def _domain_groups(units: Sequence[Unit]) -> dict[str, list[Unit]]:
    groups: dict[str, list[Unit]] = {}
    for unit in units:
        groups.setdefault(unit.domain, []).append(unit)
    for domain, members in groups.items():
        if any(u.keys is None for u in members) and len({u.n for u in members}) != 1:
            raise ValueError(f"domain {domain}: units without keys must share one aligned question set")
    return groups


def _unit_keys(unit: Unit) -> tuple:
    return unit.keys if unit.keys is not None else tuple(range(unit.n))


def _strata_index(domain: str, members, keys_of, labels_of) -> tuple[list, dict, list[list[int]]]:
    """The domain's question universe, its index, and the universe split into fixed strata.

    A question must sit in the same stratum for every unit that answers it."""
    stratum_of: dict = {}
    for u in members:
        for k, s in zip(keys_of(u), labels_of(u)):
            if stratum_of.setdefault(k, s) != s:
                raise ValueError(f"domain {domain}: question {k!r} is in two strata")
    universe = sorted(stratum_of, key=repr)
    index = {k: i for i, k in enumerate(universe)}
    by_stratum: dict = {}
    for k in universe:
        by_stratum.setdefault(stratum_of[k], []).append(index[k])
    return universe, index, [by_stratum[s] for s in sorted(by_stratum, key=repr)]


def _question_weights(rng: random.Random, n: int, strata: list[list[int]] | None) -> list[int]:
    """One bootstrap draw of question multiplicities; within each stratum when strata are given."""
    weights = [0] * n
    for members in strata if strata is not None else [list(range(n))]:
        for q in rng.choices(members, k=len(members)):
            weights[q] += 1
    return weights


def _pair_clusters(members) -> list[list[int]]:
    """Indices of members grouped by model pair, so a pair's two directions travel together."""
    clusters: dict = {}
    for i, u in enumerate(members):
        clusters.setdefault(frozenset(u.models) if u.models else u.id, []).append(i)
    return list(clusters.values())


def _pick(rng: random.Random, mode: str, count: int, clusters: list[list[int]]) -> list[int]:
    if mode == "fixed":
        return list(range(count))
    if mode == "units":
        return rng.choices(range(count), k=count)
    return [i for c in rng.choices(clusters, k=len(clusters)) for i in c]


def two_level_bootstrap(units: Sequence[Unit], reps: int = 2000, seed: int = 20261003,
                        resample: str = "fixed") -> dict:
    """Percentile CI for the mean unit difference (pp) under the chosen resampling.

    Each replicate, within every domain, draws one shared set of question weights over every
    question any unit of the domain answered, within that domain's fixed strata, because units
    answering the same questions are correlated through them. A unit's resampled difference
    uses the weights of its own questions only, so a pair run on one split and a pair run on
    both sit in the same domain without discarding questions. Domains are weighted equally in
    the overall mean, so a domain with more pairs does not dominate it.

    resample="fixed" keeps the evaluated units (the primary analysis); "pairs" also draws model
    pairs with replacement, both directions together (sensitivity); "units" draws units
    independently and ignores strata, which is the pre-audit bootstrap, kept to reproduce it.
    """
    if resample not in RESAMPLE_MODES:
        raise ValueError(f"resample must be one of {RESAMPLE_MODES}")
    groups = _domain_groups(units)
    if not groups:
        raise ValueError("no units to test")
    prepared = {}
    for domain, members in groups.items():
        universe, index, strata = _strata_index(domain, members, _unit_keys, Unit.stratum_labels)
        key_sets = sorted({_unit_keys(u) for u in members}, key=len)
        set_of = [key_sets.index(_unit_keys(u)) for u in members]
        sparse = [[(index[k], x - y) for k, x, y in zip(_unit_keys(u), u.a, u.b) if x != y] for u in members]
        prepared[domain] = (len(universe), [[index[k] for k in ks] for ks in key_sets], set_of, sparse,
                            None if resample == "units" else strata, _pair_clusters(members))
    rng = random.Random(seed)
    overall, per_domain = [], {d: [] for d in groups}
    for _ in range(reps):
        domain_means = []
        for domain, (n, key_sets, set_of, sparse, strata, clusters) in prepared.items():
            weights = _question_weights(rng, n, strata)
            totals = [sum(weights[i] for i in ks) for ks in key_sets]
            picks = _pick(rng, resample, len(sparse), clusters)
            value = sum(100 * sum(weights[q] * d for q, d in sparse[i]) / max(totals[set_of[i]], 1)
                        for i in picks) / len(picks)
            per_domain[domain].append(value)
            domain_means.append(value)
        overall.append(sum(domain_means) / len(domain_means))

    def summary(values: list[float], point: float) -> dict:
        ordered = sorted(values)
        return {"mean_diff_pp": point,
                "ci95_pp": [_quantile(ordered, 0.025), _quantile(ordered, 0.975)],
                "p_two_sided": min(1.0, 2 * min(sum(v <= 0 for v in values),
                                                sum(v >= 0 for v in values)) / len(values))}

    points = {d: sum(u.diff_pp for u in us) / len(us) for d, us in groups.items()}
    return {"reps": reps, "seed": seed, "resample": resample,
            "overall": summary(overall, sum(points.values()) / len(points)),
            "per_domain": {d: summary(per_domain[d], points[d]) for d in groups}}


def leave_one_model_out(items: Iterable[tuple[str, Sequence[str], float]]) -> list[dict]:
    """Drop every unit containing one model and recompute the sign test and mean difference.

    Units share models, so they are not independent votes. If removing one model flips
    the conclusion, the claim is about that model, not about the pairs in general.
    `items` are `(unit_id, models, diff_pp)`.
    """
    rows = list(items)
    out = []
    for model in sorted({m for _, models, _ in rows for m in models}):
        kept = [d for _, models, d in rows if model not in models]
        out.append({"dropped_model": model, "units_left": len(kept),
                    "mean_diff_pp": sum(kept) / len(kept) if kept else None,
                    **{k: v for k, v in sign_test(kept).items() if k != "units"}})
    return out


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjustment for a family of claims tested together."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[key] = running
    return adjusted


def domain_replication(per_domain: Mapping[str, dict]) -> dict:
    """How many domains agree in sign, and in how many the CI excludes zero."""
    signs = {d: (r["mean_diff_pp"] > 0) - (r["mean_diff_pp"] < 0) for d, r in per_domain.items()}
    return {"domains": len(per_domain),
            "positive": sum(s > 0 for s in signs.values()),
            "negative": sum(s < 0 for s in signs.values()),
            "ci_excludes_zero_positive": sum(r["ci95_pp"][0] > 0 for r in per_domain.values()),
            "ci_excludes_zero_negative": sum(r["ci95_pp"][1] < 0 for r in per_domain.values())}


def claim_report(units: Sequence[Unit], reps: int = 2000, seed: int = 20261003) -> dict:
    """Everything a claim card needs: per-unit tests, sign test, bootstrap, LOMO, replication.

    `bootstrap` keeps the evaluated units fixed (primary); `bootstrap_pairs` also resamples
    model pairs (appendix sensitivity)."""
    boot = two_level_bootstrap(units, reps=reps, seed=seed, resample="fixed")
    return {"units": [paired_unit(u, reps=reps, seed=seed) for u in units],
            "sign_test": sign_test(u.diff_pp for u in units),
            "sign_test_per_domain": {d: sign_test(u.diff_pp for u in us)
                                     for d, us in _domain_groups(units).items()},
            "bootstrap": boot,
            "bootstrap_pairs": two_level_bootstrap(units, reps=reps, seed=seed, resample="pairs"),
            "domain_replication": domain_replication(boot["per_domain"]),
            "leave_one_model_out": leave_one_model_out((u.id, u.models, u.diff_pp) for u in units)}


# ---------------------------------------------------------------------------
# When does a claim hold? Relating unit-level differences to properties of the unit.
#
# A claim such as "review raises the generator" can hold for some pairs and not others.
# The question then is which property of a pair predicts it: how far apart the two models
# are, how much room their independent answers leave, who reviews whom. These helpers fit
# such relations across units and give them the same two-level bootstrap as the claims.


@dataclass(frozen=True)
class Measured:
    """One unit with several per-question columns; each column becomes a percentage.

    `columns[name][i]` is the contribution of question `keys[i]` (in {-1, 0, 1}), so the
    unit's value of `name` is 100 * mean(columns[name]) over its own questions.
    """
    id: str
    domain: str
    keys: tuple
    columns: Mapping[str, Sequence[int]]
    models: tuple[str, ...] = field(default_factory=tuple)
    strata: tuple | None = None  # per-key stratum; derived from keys with question_stratum when None

    def __post_init__(self) -> None:
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError(f"{self.id}: keys must be unique and non-empty")
        for name, col in self.columns.items():
            if len(col) != len(self.keys) or any(x not in (-1, 0, 1) for x in col):
                raise ValueError(f"{self.id}: column {name} must hold -1/0/1 per key")
        if self.strata is not None and len(self.strata) != len(self.keys):
            raise ValueError(f"{self.id}: strata must match the keys")

    def stratum_labels(self) -> tuple:
        return _strata_of(self.domain, self.keys, self.strata, len(self.keys))

    def value(self, name: str) -> float:
        return 100 * sum(self.columns[name]) / len(self.keys)

    def record(self) -> dict:
        return {"id": self.id, "domain": self.domain, "models": self.models,
                **{name: self.value(name) for name in self.columns}}


def _demeaned(records: Sequence[Mapping], names: Sequence[str]) -> list[tuple[float, list[float]]]:
    """Subtract each domain's mean from every named field; weight each domain equally.

    Returns (weight, [values in `names` order]) per record. A domain with one record
    contributes nothing, which is what a per-domain intercept implies.
    """
    groups: dict[str, list[Mapping]] = {}
    for r in records:
        groups.setdefault(r["domain"], []).append(r)
    out = []
    for members in groups.values():
        means = [sum(r[n] for r in members) / len(members) for n in names]
        out += [(1 / len(members), [r[n] - m for n, m in zip(names, means)]) for r in members]
    return out


def fe_slopes(records: Sequence[Mapping], y: str, xs: Sequence[str]) -> dict[str, float] | None:
    """Least-squares slopes of `y` on `xs` with one intercept per domain, domains weighted equally.

    The per-domain intercept means only differences within a domain inform the slope, so a
    domain that is easier overall does not masquerade as an effect of `xs`. Returns None when
    the slopes are not identified (fewer usable records than slopes, or collinear `xs`).
    """
    rows = _demeaned(records, [y, *xs])
    k = len(xs)
    xtx = [[sum(w * v[1 + i] * v[1 + j] for w, v in rows) for j in range(k)] for i in range(k)]
    xty = [sum(w * v[1 + i] * v[0] for w, v in rows) for i in range(k)]
    scale = max((xtx[i][i] for i in range(k)), default=0.0)
    if scale <= 0:
        return None
    # Gaussian elimination with partial pivoting on the k x k normal equations.
    m = [row[:] + [b] for row, b in zip(xtx, xty)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) <= 1e-9 * scale:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return {x: m[i][k] / m[i][i] for i, x in enumerate(xs)}


def fe_correlation(records: Sequence[Mapping], x: str, y: str) -> float | None:
    """Pearson correlation of `x` and `y` within domains (domain means removed, domains equal)."""
    rows = _demeaned(records, [x, y])
    sxx = sum(w * v[0] ** 2 for w, v in rows)
    syy = sum(w * v[1] ** 2 for w, v in rows)
    if sxx <= 0 or syy <= 0:
        return None
    return sum(w * v[0] * v[1] for w, v in rows) / (sxx * syy) ** 0.5


def bootstrap_statistic(items: Sequence[Measured], statistic, reps: int = 2000,
                        seed: int = 20261003, resample: str = "pairs") -> dict:
    """Two-level bootstrap (as in two_level_bootstrap) of any statistic of unit-level values.

    `statistic(records)` takes `Measured.record()` dicts and returns {name: float or None}.
    Each replicate draws, per domain, one shared set of question weights within the fixed
    strata and the units under `resample`, recomputes every unit's columns under those
    weights, and re-evaluates the statistic. The default "pairs" suits statistics that compare
    model pairs, such as a slope across pairs; "units" is the pre-audit draw. A None (for
    example slopes that are not identified in that replicate) is left out of that name's
    interval and counted in `undefined_reps`.
    """
    if resample not in RESAMPLE_MODES:
        raise ValueError(f"resample must be one of {RESAMPLE_MODES}")
    if not items:
        raise ValueError("no units to fit")
    point = statistic([u.record() for u in items])
    groups: dict[str, list[Measured]] = {}
    for u in items:
        groups.setdefault(u.domain, []).append(u)
    prepared = []
    for domain, members in groups.items():
        universe, index, strata = _strata_index(domain, members, lambda u: u.keys, Measured.stratum_labels)
        unit_index = [[index[k] for k in u.keys] for u in members]
        sparse = [{n: [(index[k], x) for k, x in zip(u.keys, col) if x] for n, col in u.columns.items()}
                  for u in members]
        prepared.append((domain, len(universe), members, unit_index, sparse,
                         None if resample == "units" else strata, _pair_clusters(members)))
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    undefined = {name: 0 for name in point}
    for _ in range(reps):
        records = []
        for domain, n, members, unit_index, sparse, strata, clusters in prepared:
            weights = _question_weights(rng, n, strata)
            for i in _pick(rng, resample, len(members), clusters):
                total = max(sum(weights[q] for q in unit_index[i]), 1)
                records.append({"id": members[i].id, "domain": domain, "models": members[i].models,
                                **{name: 100 * sum(weights[q] * x for q, x in pairs) / total
                                   for name, pairs in sparse[i].items()}})
        values = statistic(records)
        for name in point:
            if values.get(name) is None:
                undefined[name] += 1
            else:
                draws[name].append(values[name])
    summary = {}
    for name, p in point.items():
        ordered = sorted(draws[name])
        if p is None or not ordered:
            summary[name] = {"point": p, "ci95": None, "p_two_sided": None, "undefined_reps": undefined[name]}
            continue
        summary[name] = {"point": p, "ci95": [_quantile(ordered, 0.025), _quantile(ordered, 0.975)],
                         "p_two_sided": min(1.0, 2 * min(sum(v <= 0 for v in ordered),
                                                         sum(v >= 0 for v in ordered)) / len(ordered)),
                         "undefined_reps": undefined[name]}
    return {"reps": reps, "seed": seed, "resample": resample, "statistics": summary}
