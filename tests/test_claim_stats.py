"""Claim statistics: exact values for the closed-form tests, invariants for the bootstrap."""

from __future__ import annotations

import unittest

import random

from authority_allocation.claim_stats import (
    Measured,
    Unit,
    _quantile,
    _question_weights,
    _strata_index,
    _unit_keys,
    bootstrap_statistic,
    paired_bootstrap_interval,
    question_stratum,
    binomial_two_sided_p,
    claim_report,
    holm,
    leave_one_model_out,
    paired_unit,
    sign_test,
    two_level_bootstrap,
)


class ClosedForm(unittest.TestCase):
    def test_sign_test_13_of_14(self):
        # 2 * (C(14,0) + C(14,1)) / 2**14 = 30 / 16384.
        self.assertAlmostEqual(binomial_two_sided_p(13, 14), 30 / 16384)
        result = sign_test([1.0] * 13 + [-0.5])
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (13, 1, 0))
        self.assertAlmostEqual(result["p_two_sided"], 30 / 16384)

    def test_ties_are_dropped(self):
        result = sign_test([0.0, 0.0, 1.0])
        self.assertEqual(result["ties"], 2)
        self.assertEqual(result["p_two_sided"], 1.0)

    def test_paired_unit_counts_discordant_rows(self):
        unit = Unit("u", "d", a=[1, 1, 0, 0, 1], b=[1, 0, 0, 1, 0])
        r = paired_unit(unit)
        self.assertEqual((r["a_only"], r["b_only"]), (2, 1))
        self.assertAlmostEqual(r["diff_pp"], 20.0)
        self.assertEqual(r["mcnemar_p"], 1.0)

    def test_holm(self):
        adjusted = holm({"x": 0.01, "y": 0.04, "z": 0.03})
        self.assertAlmostEqual(adjusted["x"], 0.03)
        self.assertAlmostEqual(adjusted["z"], 0.06)
        self.assertAlmostEqual(adjusted["y"], 0.06)

    def test_leave_one_model_out(self):
        rows = leave_one_model_out([("p1", ("A", "B"), 2.0), ("p2", ("A", "C"), 1.0),
                                    ("p3", ("B", "C"), -1.0)])
        by_model = {r["dropped_model"]: r for r in rows}
        self.assertEqual(by_model["A"]["units_left"], 1)
        self.assertAlmostEqual(by_model["A"]["mean_diff_pp"], -1.0)
        self.assertAlmostEqual(by_model["C"]["mean_diff_pp"], 2.0)

    def test_rejects_bad_units(self):
        with self.assertRaises(ValueError):
            Unit("u", "d", a=[1, 0], b=[1])
        with self.assertRaises(ValueError):
            Unit("u", "d", a=[2], b=[1])


class Bootstrap(unittest.TestCase):
    def test_identical_systems_give_zero(self):
        units = [Unit(f"p{i}", "d", a=[1, 0, 1, 0], b=[1, 0, 1, 0]) for i in range(3)]
        r = two_level_bootstrap(units, reps=200)
        self.assertEqual(r["overall"]["mean_diff_pp"], 0.0)
        self.assertEqual(r["overall"]["ci95_pp"], [0.0, 0.0])

    def test_clear_effect_excludes_zero_and_is_reproducible(self):
        a = [1] * 60 + [0] * 40
        b = [1] * 40 + [0] * 60
        units = [Unit(f"p{i}", dom, a=a, b=b, models=(f"m{i}", "shared"))
                 for i, dom in enumerate(["math", "math", "knowledge", "knowledge"])]
        first = claim_report(units, reps=300, seed=7)
        second = claim_report(units, reps=300, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["bootstrap"]["overall"]["mean_diff_pp"], 20.0)
        self.assertGreater(first["bootstrap"]["overall"]["ci95_pp"][0], 0)
        self.assertEqual(first["domain_replication"]["ci_excludes_zero_positive"], 2)

    def test_domains_weighted_equally(self):
        up = Unit("u", "one", a=[1, 1], b=[0, 0])
        downs = [Unit(f"d{i}", "two", a=[0, 0], b=[0, 0]) for i in range(5)]
        r = two_level_bootstrap([up, *downs], reps=50)
        self.assertAlmostEqual(r["overall"]["mean_diff_pp"], 50.0)

    def test_units_on_different_splits_keep_their_questions(self):
        dev = Unit("dev_only", "d", a=[1, 1, 0, 1], b=[0, 1, 0, 0], keys=("d1", "d2", "d3", "d4"))
        both = Unit("both", "d", a=[1, 0, 1, 1, 0, 1], b=[1, 0, 0, 1, 0, 0],
                    keys=("d1", "d2", "t1", "t2", "t3", "t4"))
        r = two_level_bootstrap([dev, both], reps=300, seed=3)
        self.assertAlmostEqual(r["overall"]["mean_diff_pp"], (50.0 + 100 * 2 / 6) / 2)
        lo, hi = r["overall"]["ci95_pp"]
        self.assertLessEqual(lo, hi)
        with self.assertRaises(ValueError):
            Unit("bad", "d", a=[1, 0], b=[0, 0], keys=("x", "x"))

    def test_domain_units_must_share_questions(self):
        with self.assertRaises(ValueError):
            two_level_bootstrap([Unit("a", "d", [1], [0]), Unit("b", "d", [1, 0], [0, 0])])


def _legacy_two_level_bootstrap(units, reps, seed):
    """The pre-audit bootstrap, copied verbatim in behaviour: units and questions drawn freely."""
    groups = {}
    for u in units:
        groups.setdefault(u.domain, []).append(u)
    prepared = {}
    for domain, members in groups.items():
        universe = sorted({k for u in members for k in _unit_keys(u)}, key=repr)
        index = {k: i for i, k in enumerate(universe)}
        key_sets = sorted({_unit_keys(u) for u in members}, key=len)
        set_of = [key_sets.index(_unit_keys(u)) for u in members]
        sparse = [[(index[k], x - y) for k, x, y in zip(_unit_keys(u), u.a, u.b) if x != y] for u in members]
        prepared[domain] = (len(universe), [[index[k] for k in ks] for ks in key_sets], set_of, sparse)
    rng = random.Random(seed)
    overall = []
    for _ in range(reps):
        means = []
        for n, key_sets, set_of, sparse in prepared.values():
            weights = [0] * n
            for q in rng.choices(range(n), k=n):
                weights[q] += 1
            totals = [sum(weights[i] for i in ks) for ks in key_sets]
            picks = rng.choices(range(len(sparse)), k=len(sparse))
            means.append(sum(100 * sum(weights[q] * d for q, d in sparse[i]) / max(totals[set_of[i]], 1)
                             for i in picks) / len(picks))
        overall.append(sum(means) / len(means))
    return sorted(overall)


def _random_unit(rng, uid, domain, n, p_a, p_b, models=(), keys=None):
    a = [int(rng.random() < p_a) for _ in range(n)]
    b = [int(rng.random() < p_b) for _ in range(n)]
    return Unit(uid, domain, a, b, models=models, keys=keys)


class AuditedInference(unittest.TestCase):
    """The resampling design approved on 2026-10-05 (docs/analysis/stats_audit_20261005)."""

    def test_strata_follow_split_and_reasoning_dataset(self):
        self.assertEqual(question_stratum("nl_reasoning", ("dev", "reclor/val_12")), ("dev", "reclor"))
        self.assertEqual(question_stratum("nl_reasoning", "gpqa/rec9"), (None, "gpqa"))
        self.assertEqual(question_stratum("math", ("test", "algebra/12.json")), ("test", ""))
        self.assertEqual(question_stratum("code", 7), (None, ""))

    def test_question_weights_keep_every_stratum_size(self):
        strata = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]
        rng = random.Random(1)
        for _ in range(50):
            w = _question_weights(rng, 10, strata)
            self.assertEqual([sum(w[i] for i in s) for s in strata], [3, 2, 5])

    def test_a_question_in_two_strata_is_rejected(self):
        u1 = Unit("x", "d", [1, 0], [0, 0], keys=("q1", "q2"), strata=("s", "s"))
        u2 = Unit("y", "d", [1, 0], [0, 0], keys=("q1", "q2"), strata=("s", "t"))
        with self.assertRaises(ValueError):
            _strata_index("d", [u1, u2], _unit_keys, Unit.stratum_labels)

    def test_paired_bootstrap_agrees_with_newcombe_and_keeps_old_interval(self):
        rng = random.Random(3)
        unit = _random_unit(rng, "big", "math", 2500, 0.75, 0.50)  # a large difference
        r = paired_unit(unit)
        for got, ref in zip(r["ci95_pp"], r["ci95_pp_newcombe"]):
            self.assertAlmostEqual(got, ref, delta=0.4)
        self.assertLess(r["ci95_pp"][0], r["diff_pp"])
        self.assertGreater(r["ci95_pp"][1], r["diff_pp"])
        # The pre-audit interval held the discordant count fixed and is narrower here.
        old = r["ci95_pp_pre_audit"]
        self.assertLess(old[1] - old[0], r["ci95_pp"][1] - r["ci95_pp"][0])

    def test_paired_bootstrap_resamples_within_strata(self):
        # Stratum "t" is all ties, so only stratum "s" can move the difference.
        a = [1, 0, 1, 0] + [1] * 6
        b = [0, 1, 1, 0] + [1] * 6
        lo, hi = paired_bootstrap_interval(a, b, ["s"] * 4 + ["t"] * 6, reps=500, seed=5)
        self.assertGreaterEqual(lo, -40.0)
        self.assertLessEqual(hi, 40.0)
        self.assertEqual(paired_bootstrap_interval(a, b, ["s"] * 10, reps=500, seed=5),
                         paired_bootstrap_interval(a, b, ["s"] * 10, reps=500, seed=5))

    def test_units_mode_reproduces_the_pre_audit_bootstrap(self):
        rng = random.Random(11)
        keys = tuple(range(300))
        units = [_random_unit(rng, f"u{i}", d, 300, 0.6, 0.5 + 0.02 * i, keys=keys)
                 for d in ("x", "y") for i in range(4)]
        legacy = _legacy_two_level_bootstrap(units, reps=200, seed=9)
        got = two_level_bootstrap(units, reps=200, seed=9, resample="units")["overall"]["ci95_pp"]
        self.assertAlmostEqual(got[0], _quantile(legacy, 0.025), places=9)
        self.assertAlmostEqual(got[1], _quantile(legacy, 0.975), places=9)

    def test_fixed_mode_with_one_unit_matches_the_paired_bootstrap(self):
        rng = random.Random(4)
        unit = _random_unit(rng, "solo", "math", 1500, 0.70, 0.62)
        fixed = two_level_bootstrap([unit], reps=2000, seed=2, resample="fixed")["overall"]["ci95_pp"]
        single = paired_unit(unit)["ci95_pp"]
        for x, y in zip(fixed, single):
            self.assertAlmostEqual(x, y, delta=0.35)

    def test_fixed_is_narrower_than_resampling_heterogeneous_units(self):
        rng = random.Random(8)
        keys = tuple(range(800))
        units = [_random_unit(rng, f"u{i}", "d", 800, 0.5 + 0.06 * i, 0.5, keys=keys, models=(f"m{i}", f"n{i}"))
                 for i in range(6)]
        width = lambda m: (lambda c: c[1] - c[0])(two_level_bootstrap(units, reps=400, seed=1, resample=m)["overall"]["ci95_pp"])
        self.assertLess(width("fixed"), width("units"))
        self.assertLess(width("fixed"), width("pairs"))

    def test_pairs_mode_keeps_both_directions_together(self):
        # Each pair's two directions cancel exactly, so any draw of whole pairs gives 0 on fixed questions.
        keys = tuple(range(40))
        a = [1] * 20 + [0] * 20
        b = [0] * 20 + [1] * 20
        units = []
        for i in range(5):
            units.append(Unit(f"p{i}:fwd", "d", a, b, models=(f"m{i}", f"n{i}"), keys=keys))
            units.append(Unit(f"p{i}:rev", "d", b, a, models=(f"n{i}", f"m{i}"), keys=keys))
        ci = two_level_bootstrap(units, reps=200, seed=3, resample="pairs")["overall"]["ci95_pp"]
        self.assertAlmostEqual(ci[0], 0.0)
        self.assertAlmostEqual(ci[1], 0.0)
        loose = two_level_bootstrap(units, reps=200, seed=3, resample="units")["overall"]["ci95_pp"]
        self.assertGreater(loose[1] - loose[0], 1.0)

    def test_bootstrap_statistic_reports_its_resampling(self):
        rng = random.Random(6)
        keys = tuple(("dev", f"reclor/{i}") for i in range(30)) + tuple(("dev", f"gpqa/{i}") for i in range(20))
        items = [Measured(f"u{i}", "nl_reasoning", keys,
                          {"x": [rng.choice((-1, 0, 1)) for _ in keys]}, models=(f"m{i}", "z")) for i in range(4)]
        stat = lambda rs: {"mean": sum(r["x"] for r in rs) / len(rs)}
        out = bootstrap_statistic(items, stat, reps=100, seed=1)
        self.assertEqual(out["resample"], "pairs")
        self.assertIsNotNone(out["statistics"]["mean"]["ci95"])
        with self.assertRaises(ValueError):
            bootstrap_statistic(items, stat, reps=10, resample="bogus")


if __name__ == "__main__":
    unittest.main()
