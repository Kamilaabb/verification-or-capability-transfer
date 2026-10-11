"""router_v1 semantics: the two-arm fallback, the pools, and cost-aware policy value."""

from __future__ import annotations

import unittest

from authority_allocation.router_semantics import (
    best_fixed_arm,
    bootstrap_policy_ci,
    deployed_outcomes,
    fixed_arm_values,
    logit_identity_error,
    oracle_headroom,
    policy_value,
    pool_counts,
)


def row(pid, u0, u_self, status="ok", t1=100, t2=50):
    outcomes = deployed_outcomes(u0=u0, u_self=u_self, self_check_status=status)
    return {"problem_id": pid, "u0": u0, "u_self": u_self,
            "self_check_status": status, "tokens_pass1": t1, "tokens_pass2": t2,
            **outcomes}


class FallbackTests(unittest.TestCase):
    def test_successful_call_deploys_the_self_check_answer(self):
        self.assertEqual(deployed_outcomes(u0=1, u_self=0, self_check_status="ok"),
                         {"a0": 1, "a1": 0})

    def test_failed_call_falls_back_to_y0_rather_than_scoring_zero(self):
        for status in ("model_invalid", "call_error", "fallback"):
            with self.subTest(status=status):
                self.assertEqual(
                    deployed_outcomes(u0=1, u_self=None, self_check_status=status),
                    {"a0": 1, "a1": 1},
                )

    def test_retry_ok_is_a_success_not_a_fallback(self):
        self.assertEqual(deployed_outcomes(u0=0, u_self=1, self_check_status="retry_ok"),
                         {"a0": 0, "a1": 1})

    def test_null_u_self_on_a_successful_call_is_rejected(self):
        with self.assertRaises(ValueError):
            deployed_outcomes(u0=1, u_self=None, self_check_status="ok")

    def test_non_binary_outcomes_are_rejected(self):
        with self.assertRaises(ValueError):
            deployed_outcomes(u0=2, u_self=1, self_check_status="ok")


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row("repair1", 0, 1), row("repair2", 0, 1),
            row("harm1", 1, 0),
            row("bothright", 1, 1), row("bothwrong", 0, 0),
            row("failed", 1, None, status="call_error"),
        ]

    def test_counts_each_class(self):
        counts = pool_counts(self.rows)
        self.assertEqual(counts["repair"], 2)
        self.assertEqual(counts["harm"], 1)
        # The failed row falls back to y0, so both arms agree and it is not decisive.
        self.assertEqual(counts["both_right"], 2)
        self.assertEqual(counts["both_wrong"], 1)
        self.assertEqual(counts["decision_relevant"], 3)

    def test_harm_pool_bounds_the_gain_over_a1_specifically(self):
        counts = pool_counts(self.rows)
        always = policy_value(self.rows, [r["problem_id"] for r in self.rows])
        perfect = policy_value(
            self.rows, [r["problem_id"] for r in self.rows if r["problem_id"] != "harm1"]
        )
        self.assertEqual(perfect["correct"] - always["correct"], counts["harm"])


class HeadroomTests(unittest.TestCase):
    """min(R, H), not H, is what a router can add over the best fixed arm."""

    def build(self, repair, harm, both_right=5, both_wrong=5):
        rows = []
        for i in range(repair):     rows.append(row(f"r{i}", 0, 1))
        for i in range(harm):       rows.append(row(f"h{i}", 1, 0))
        for i in range(both_right): rows.append(row(f"y{i}", 1, 1))
        for i in range(both_wrong): rows.append(row(f"n{i}", 0, 0))
        return rows

    def perfect_router(self, rows):
        return policy_value(rows, [r["problem_id"] for r in rows if not (r["a0"] == 1 and r["a1"] == 0)])

    def test_headroom_equals_min_r_h_in_every_regime(self):
        for repair, harm in [(100, 3), (20, 20), (5, 30), (53, 25), (0, 7), (7, 0)]:
            with self.subTest(R=repair, H=harm):
                rows = self.build(repair, harm)
                head = oracle_headroom(rows)
                self.assertEqual(head["headroom_over_best_fixed"], min(repair, harm))
                # and it is exactly what a perfect router adds over the best fixed arm
                gain = self.perfect_router(rows)["correct"] - best_fixed_arm(rows)["correct"]
                self.assertEqual(gain, min(repair, harm))

    def test_a_zero_mean_effect_is_the_largest_headroom_not_the_smallest(self):
        # The case the first draft of Gate 0 would have rejected: A0 and A1 tie, so the
        # average effect of self-checking is zero, yet a perfect router gains 20 rows.
        balanced = self.build(20, 20)
        arms = fixed_arm_values(balanced)
        self.assertEqual(arms["A0_never_self_check"]["correct"],
                         arms["A1_always_self_check"]["correct"])
        self.assertEqual(best_fixed_arm(balanced)["name"], "tie")
        self.assertEqual(oracle_headroom(balanced)["headroom_over_best_fixed"], 20)

        # A large positive mean effect can carry far less headroom than a zero one.
        lopsided = self.build(100, 3)
        self.assertGreater(
            arms["A1_always_self_check"]["correct"] - arms["A0_never_self_check"]["correct"], -1)
        self.assertLess(oracle_headroom(lopsided)["headroom_over_best_fixed"],
                        oracle_headroom(balanced)["headroom_over_best_fixed"])

    def test_best_fixed_arm_picks_the_stronger_baseline(self):
        self.assertEqual(best_fixed_arm(self.build(30, 5))["name"], "A1_always_self_check")
        self.assertEqual(best_fixed_arm(self.build(5, 30))["name"], "A0_never_self_check")


class BootstrapTests(unittest.TestCase):
    def test_reports_intervals_for_both_accuracy_and_tokens(self):
        rows = [row(f"r{i}", 0, 1) for i in range(20)] + [row(f"h{i}", 1, 0) for i in range(20)]
        called = [r["problem_id"] for r in rows if r["problem_id"].startswith("r")]
        out = bootstrap_policy_ci(rows, called, baseline=[], seed=7, repetitions=400)
        self.assertGreater(out["accuracy_difference"], 0)
        self.assertLessEqual(out["accuracy_ci"][0], out["accuracy_difference"])
        self.assertGreaterEqual(out["accuracy_ci"][1], out["accuracy_difference"])
        # Calling pass 2 on 20 rows costs tokens relative to never calling it.
        self.assertGreater(out["token_difference"], 0)
        self.assertLessEqual(out["token_ci"][0], out["token_ci"][1])

    def test_cost_is_not_treated_as_error_free_across_problem_samples(self):
        rows = [row(f"r{i}", 0, 1, t2=10 * (i + 1)) for i in range(30)]
        called = [r["problem_id"] for r in rows]
        out = bootstrap_policy_ci(rows, called, baseline=[], seed=3, repetitions=400)
        self.assertLess(out["token_ci"][0], out["token_ci"][1])


class PolicyValueTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("a", 0, 1), row("b", 1, 0), row("c", 1, 1), row("d", 0, 0)]

    def test_never_calling_pays_pass1_only(self):
        value = policy_value(self.rows, [])
        self.assertEqual(value["output_tokens"], 4 * 100)
        self.assertEqual(value["correct"], 2)          # u0 of c and b
        self.assertEqual(value["call_fraction"], 0.0)

    def test_always_calling_pays_both_passes(self):
        value = policy_value(self.rows, [r["problem_id"] for r in self.rows])
        self.assertEqual(value["output_tokens"], 4 * 150)
        self.assertEqual(value["correct"], 2)          # a repaired, b harmed
        self.assertEqual(value["call_fraction"], 1.0)

    def test_a_selective_router_can_match_accuracy_at_lower_cost(self):
        # Call only the row self-check repairs; skip the one it breaks.
        value = policy_value(self.rows, ["a"])
        always = policy_value(self.rows, [r["problem_id"] for r in self.rows])
        self.assertGreater(value["correct"], always["correct"])
        self.assertLess(value["output_tokens"], always["output_tokens"])

    def test_fixed_arms_match_the_manual_extremes(self):
        arms = fixed_arm_values(self.rows)
        self.assertEqual(arms["A0_never_self_check"], policy_value(self.rows, []))
        self.assertEqual(arms["A1_always_self_check"]["call_fraction"], 1.0)

    def test_routing_an_unknown_row_is_rejected(self):
        with self.assertRaises(KeyError):
            policy_value(self.rows, ["a", "not_in_archive"])

    def test_ungraded_rows_are_counted_not_silently_scored(self):
        rows = self.rows + [row("x", None, None, status="call_error")]
        value = policy_value(rows, [])
        self.assertEqual(value["ungraded"], 1)
        self.assertEqual(value["correct"], 2)


class LogitIdentityTests(unittest.TestCase):
    def test_exact_greedy_trace_has_near_zero_error(self):
        positions = [{"max_logit": 12.0, "logsumexp": 12.5, "sampled_logprob": -0.5},
                     {"max_logit": 8.25, "logsumexp": 9.0, "sampled_logprob": -0.75}]
        self.assertLess(logit_identity_error(positions), 1e-9)

    def test_a_misaligned_capture_is_detected(self):
        positions = [{"max_logit": 12.0, "logsumexp": 12.5, "sampled_logprob": -3.0}]
        self.assertGreater(logit_identity_error(positions), 1e-4)

    def test_logprob_is_scale_free_but_max_logit_is_not(self):
        # Shifting every logit by a constant leaves the log-probability untouched.
        shift = 7.0
        base = {"max_logit": 12.0, "logsumexp": 12.5, "sampled_logprob": -0.5}
        shifted = {"max_logit": base["max_logit"] + shift,
                   "logsumexp": base["logsumexp"] + shift,
                   "sampled_logprob": base["sampled_logprob"]}
        self.assertLess(logit_identity_error([shifted]), 1e-9)
        self.assertNotEqual(base["max_logit"], shifted["max_logit"])


if __name__ == "__main__":
    unittest.main()
