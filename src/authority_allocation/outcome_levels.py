"""Five-level outcome of one review direction (H09), and a domain-stratified rank test.

A direction is placed by the 95% intervals H08 already reports for Δ_G = acc(GV) − acc(G) and
Δ_C = acc(GV) − acc(stronger alone):

    1 worse than G        Δ_G interval entirely below 0
    2 about G             Δ_G interval contains 0
    3 between             Δ_G interval above 0, Δ_C interval entirely below 0
    4 about the stronger  Δ_G interval above 0, Δ_C interval contains 0
    5 beyond the stronger Δ_C interval entirely above 0

When the weaker model reviews, G is the stronger model, Δ_G = Δ_C, and only levels 1, 2 and 5 occur.
"""
import random

LEVELS = (1, 2, 3, 4, 5)
LEVEL_NAMES = {1: 'worse than G', 2: 'about G', 3: 'above G, below the stronger',
               4: 'about the stronger', 5: 'beyond the stronger'}


def outcome_level(dG_ci, dC_ci):
    if dC_ci[0] > 0:
        return 5
    if dG_ci[1] < 0:
        return 1
    if dG_ci[0] <= 0:
        return 2
    if dC_ci[1] < 0:
        return 3
    return 4


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def stratified_spearman(groups, reps, seed):
    """Mean within-group Spearman of (x, y) and a two-sided permutation p-value.

    `groups` maps a stratum (domain) to a list of (x, y). x is permuted within each stratum only,
    so domains with different gap ranges or interval widths are never compared with each other.
    Strata whose rank correlation is undefined (constant x or y) are left out of the mean.
    """
    def stat(gs):
        vals = [spearman([p[0] for p in g], [p[1] for p in g]) for g in gs.values()]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    observed = stat(groups)
    if observed is None:
        return {'point': None, 'p_two_sided': None, 'reps': reps}
    rnd = random.Random(seed)
    hits = 0
    for _ in range(reps):
        shuffled = {}
        for k, g in groups.items():
            xs = [p[0] for p in g]
            rnd.shuffle(xs)
            shuffled[k] = [(x, p[1]) for x, p in zip(xs, g)]
        s = stat(shuffled)
        if s is not None and abs(s) >= abs(observed) - 1e-12:
            hits += 1
    return {'point': observed, 'p_two_sided': (hits + 1) / (reps + 1), 'reps': reps}
