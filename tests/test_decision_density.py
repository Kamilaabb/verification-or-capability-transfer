"""Exact arithmetic over archived cap outcomes, checked against published results.

The two published pattern tables are the fixtures. If this module ever disagrees with
`docs/analysis/gate1_c007.md` or `docs/analysis/gate1_c013.md`, one of them is wrong,
and these tests say which.
"""

from __future__ import annotations

import unittest

from authority_allocation.decision_density import (
    accuracy_difference_interval,
    cap_correct,
    contrast_report,
    decision_relevant_rows,
    load_cap_outcomes,
    margin,
    mcnemar_exact_p,
    minimum_detectable_net,
    pattern_counts,
    restrict,
    two_sample_repair_share_z,
    wilson_interval,
)

# docs/analysis/gate1_c013.md, "结果": the eight (z0,z1,z2) counts over 300 rows.
C013_PATTERNS = {
    "000": 137, "001": 0, "010": 1, "011": 10,
    "100": 13, "101": 1, "110": 0, "111": 138,
}

# docs/analysis/gate1_c007.md, section 3: the same table over 832 development rows.
C007_PATTERNS = {
    "000": 472, "001": 5, "010": 1, "011": 48,
    "100": 24, "101": 1, "110": 1, "111": 280,
}


def archive_from_patterns(patterns: dict[str, int], prefix: str) -> list[dict]:
    """Expand a pattern table into counterfactual-shaped rows with distinct ids."""
    rows = []
    for pattern, count in patterns.items():
        for index in range(count):
            rows.append({
                "problem_id": f"{prefix}/{pattern}/{index}",
                "z0": int(pattern[0]),
                "z1": int(pattern[1]),
                "z2": int(pattern[2]),
            })
    return rows


class LoadingTests(unittest.TestCase):
    def test_rejects_duplicate_problem_ids(self):
        rows = [{"problem_id": "p", "z0": 1, "z1": 1, "z2": 1}] * 2
        with self.assertRaises(ValueError):
            load_cap_outcomes(rows)

    def test_rejects_unresolved_cap_outcome(self):
        rows = [{"problem_id": "p", "z0": 1, "z1": None, "z2": 0}]
        with self.assertRaises(ValueError):
            load_cap_outcomes(rows)

    def test_restrict_requires_every_requested_row(self):
        outcomes = load_cap_outcomes(archive_from_patterns({"111": 3}, "a"))
        with self.assertRaises(KeyError):
            restrict(outcomes, list(outcomes) + ["absent"])

    def test_restrict_keeps_only_the_requested_rows(self):
        outcomes = load_cap_outcomes(archive_from_patterns({"111": 2, "000": 2}, "a"))
        wanted = [pid for pid in outcomes if "/111/" in pid]
        self.assertEqual(len(restrict(outcomes, wanted)), 2)


class PublishedC013Tests(unittest.TestCase):
    def setUp(self):
        self.outcomes = load_cap_outcomes(archive_from_patterns(C013_PATTERNS, "c013"))

    def test_reproduces_published_cap_totals(self):
        self.assertEqual(len(self.outcomes), 300)
        self.assertEqual(cap_correct(self.outcomes, "z0"), 152)
        self.assertEqual(cap_correct(self.outcomes, "z1"), 149)
        self.assertEqual(cap_correct(self.outcomes, "z2"), 149)

    def test_reproduces_published_repair_and_harm(self):
        self.assertEqual(margin(self.outcomes, "z0", "z1"), (11, 14))
        self.assertEqual(margin(self.outcomes, "z1", "z2"), (1, 1))
        self.assertEqual(margin(self.outcomes, "z0", "z2"), (10, 13))

    def test_oracle_and_decision_relevant_rows(self):
        self.assertEqual(300 - pattern_counts(self.outcomes)["000"], 163)
        # 300 - 137 all-wrong - 138 all-right. Every one of them is an M0/M1
        # disagreement, because 001 and 110 are both empty.
        self.assertEqual(decision_relevant_rows(self.outcomes), 25)

    def test_m1_result_is_not_significant_and_could_not_have_been(self):
        report = contrast_report(self.outcomes, "z0", "z1")
        self.assertEqual(report["net"], -3)
        self.assertEqual(report["discordant"], 25)
        self.assertAlmostEqual(report["mcnemar_exact_p"], 0.6900, places=4)
        low, high = report["accuracy_difference_ci95"]
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)
        # 25 discordant pairs cannot resolve anything smaller than an 18:7 split.
        self.assertEqual(report["minimum_detectable_net"], 11)

    def test_c007_sized_effect_would_have_been_below_the_detection_floor(self):
        expected_net_here = 24 / 832 * 300
        self.assertLess(expected_net_here, 11)


class PublishedC007Tests(unittest.TestCase):
    def setUp(self):
        self.outcomes = load_cap_outcomes(archive_from_patterns(C007_PATTERNS, "c007"))

    def test_reproduces_published_accuracies(self):
        self.assertEqual(len(self.outcomes), 832)
        self.assertAlmostEqual(cap_correct(self.outcomes, "z0") / 832, 0.3678, places=4)
        self.assertAlmostEqual(cap_correct(self.outcomes, "z1") / 832, 0.3966, places=4)
        self.assertAlmostEqual(cap_correct(self.outcomes, "z2") / 832, 0.4014, places=4)

    def test_reproduces_published_margins(self):
        self.assertEqual(margin(self.outcomes, "z0", "z1"), (49, 25))
        self.assertEqual(margin(self.outcomes, "z1", "z2"), (6, 2))
        self.assertEqual(margin(self.outcomes, "z0", "z2"), (53, 25))
        self.assertEqual(decision_relevant_rows(self.outcomes), 80)

    def test_m1_result_is_significant(self):
        report = contrast_report(self.outcomes, "z0", "z1")
        self.assertAlmostEqual(report["mcnemar_exact_p"], 0.0071, places=4)
        low, high = report["accuracy_difference_ci95"]
        self.assertGreater(low, 0.0)

    def test_m2_over_m1_was_never_resolvable_either(self):
        report = contrast_report(self.outcomes, "z1", "z2")
        self.assertEqual(report["discordant"], 8)
        self.assertGreater(report["mcnemar_exact_p"], 0.05)
        # 8 pairs are resolvable only by a clean 8:0 sweep (p=0.0078). The observed
        # 6:2 is the second-best split available and already misses, so M2 over M1
        # was underpowered in c007 too, at four times c013's row count.
        self.assertEqual(minimum_detectable_net(8), 8)
        self.assertAlmostEqual(mcnemar_exact_p(8, 0), 2 / 256, places=10)


class CrossConditionTableTests(unittest.TestCase):
    """The cross-condition table in gate1_c013.md, recomputed from its sources.

    c003's repair/harm come from PROJECT_STATE.md rather than a pattern table, because
    c003_decision_structure.md publishes request shares and the OVERRIDE subset, not the
    eight (z0,z1,z2) counts. Its 74.3% is the KEEP rate, which bounds decision-relevant
    rows from above at 77 but does not determine them, so no density is claimed for it.
    """

    CONTRASTS = {
        ("c003", "M1-M0"): (38, 2),
        ("c003", "M2-M1"): (10, 9),
        ("c007", "M1-M0"): (49, 25),
        ("c007", "M2-M1"): (6, 2),
        ("c013", "M1-M0"): (11, 14),
        ("c013", "M2-M1"): (1, 1),
    }

    def test_published_detection_floors(self):
        expected = {
            ("c003", "M1-M0"): 14, ("c003", "M2-M1"): 11,
            ("c007", "M1-M0"): 18, ("c007", "M2-M1"): 8,
            ("c013", "M1-M0"): 11, ("c013", "M2-M1"): None,
        }
        for key, (repair, harm) in self.CONTRASTS.items():
            with self.subTest(contrast=key):
                self.assertEqual(minimum_detectable_net(repair + harm), expected[key])

    def test_m2_over_m1_was_never_powered_in_any_condition(self):
        for condition in ("c003", "c007", "c013"):
            repair, harm = self.CONTRASTS[(condition, "M2-M1")]
            with self.subTest(condition=condition):
                self.assertGreater(mcnemar_exact_p(repair, harm), 0.05)
                floor = minimum_detectable_net(repair + harm)
                self.assertTrue(floor is None or abs(repair - harm) < floor)

    def test_m1_over_m0_reached_significance_everywhere_except_c013(self):
        self.assertLess(mcnemar_exact_p(38, 2), 0.05)
        self.assertLess(mcnemar_exact_p(49, 25), 0.05)
        self.assertGreater(mcnemar_exact_p(11, 14), 0.05)


class StatisticsTests(unittest.TestCase):
    def test_mcnemar_matches_known_binomial_values(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertEqual(mcnemar_exact_p(5, 5), 1.0)
        # b=6,c=0 is a one-sided tail of 2^-6, doubled.
        self.assertAlmostEqual(mcnemar_exact_p(6, 0), 2 / 64, places=10)
        self.assertAlmostEqual(mcnemar_exact_p(0, 6), 2 / 64, places=10)

    def test_mcnemar_is_symmetric_in_direction(self):
        self.assertEqual(mcnemar_exact_p(11, 14), mcnemar_exact_p(14, 11))

    def test_minimum_detectable_net_matches_parity_and_is_achievable(self):
        for discordant in range(1, 60):
            net = minimum_detectable_net(discordant)
            if net is None:
                continue
            self.assertEqual(net % 2, discordant % 2)
            repair = (discordant + net) // 2
            self.assertLess(mcnemar_exact_p(repair, discordant - repair), 0.05)
            if net > discordant % 2 + 1:
                smaller = repair - 1
                self.assertGreaterEqual(
                    mcnemar_exact_p(smaller, discordant - smaller), 0.05
                )

    def test_no_split_of_five_pairs_can_reach_significance(self):
        self.assertIsNone(minimum_detectable_net(5))

    def test_wilson_interval_stays_in_unit_range_at_extremes(self):
        low, high = wilson_interval(0, 3)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_accuracy_interval_is_zero_width_without_discordant_rows(self):
        self.assertEqual(accuracy_difference_interval(0, 0, 300), (0.0, 0.0))

    def test_repair_share_comparison_reproduces_the_cross_condition_z(self):
        z, p = two_sample_repair_share_z(49, 25, 11, 14)
        self.assertAlmostEqual(z, 1.97, places=2)
        self.assertAlmostEqual(p, 0.049, places=3)


if __name__ == "__main__":
    unittest.main()
