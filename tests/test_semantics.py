from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.semantics import (  # noqa: E402
    Action,
    AuthorityCap,
    counterfactual_outcomes,
    derive_call_seed,
    effective_action,
    paired_policy_metrics,
    select_y0,
)


class AuthoritySemanticsTests(unittest.TestCase):
    def test_all_nine_cap_request_mappings(self) -> None:
        expected = {
            ("M0", "KEEP"): "KEEP",
            ("M0", "CRITIQUE_REVISE"): "KEEP",
            ("M0", "OVERRIDE"): "KEEP",
            ("M1", "KEEP"): "KEEP",
            ("M1", "CRITIQUE_REVISE"): "CRITIQUE_REVISE",
            ("M1", "OVERRIDE"): "CRITIQUE_REVISE",
            ("M2", "KEEP"): "KEEP",
            ("M2", "CRITIQUE_REVISE"): "CRITIQUE_REVISE",
            ("M2", "OVERRIDE"): "OVERRIDE",
        }
        for (cap, request), action in expected.items():
            with self.subTest(cap=cap, request=request):
                self.assertEqual(effective_action(cap, request).name, action)

    def test_y0_is_always_sample_zero(self) -> None:
        rows = [
            {"sample_index": 0, "parse_status": "model_invalid", "raw_output": ""},
            {"sample_index": 1, "parse_status": "ok", "raw_output": "42"},
        ]
        self.assertIs(select_y0(rows), rows[0])

    def test_y0_missing_or_duplicate_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            select_y0([{"sample_index": 1}])
        with self.assertRaises(ValueError):
            select_y0([{"sample_index": 0}, {"sample_index": 0}])

    def test_counterfactual_override_request(self) -> None:
        self.assertEqual(
            counterfactual_outcomes("OVERRIDE", u0=0, u1=1, u2=0, rewrite_status="ok"),
            {"z0": 0, "z1": 1, "z2": 0},
        )

    def test_failed_rewrite_falls_back_to_keep(self) -> None:
        self.assertEqual(
            counterfactual_outcomes("OVERRIDE", u0=1, u1=None, u2=0, rewrite_status="model_invalid"),
            {"z0": 1, "z1": 1, "z2": 0},
        )

    def test_keep_does_not_require_unobserved_branches(self) -> None:
        self.assertEqual(
            counterfactual_outcomes("KEEP", u0=1, u1=None, u2=None, rewrite_status="not_requested"),
            {"z0": 1, "z1": 1, "z2": 1},
        )

    def test_seed_is_stable_and_sensitive_to_registered_fields(self) -> None:
        args = ("r001", "controller", "c01", 2026091101, "V-A", "problem-1", 0)
        seed = derive_call_seed(*args)
        self.assertEqual(seed, derive_call_seed(*args))
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**31)
        self.assertNotEqual(
            seed,
            derive_call_seed("r001", "controller", "c02", 2026091102, "V-A", "problem-1", 0),
        )

    def test_seed_changes_when_only_base_seed_changes(self) -> None:
        seed_a = derive_call_seed("r001", "controller", "c01", 1, "V-A", "problem-1", 0)
        seed_b = derive_call_seed("r001", "controller", "c01", 2, "V-A", "problem-1", 0)
        self.assertNotEqual(seed_a, seed_b)

    def test_paired_repair_harm_identity(self) -> None:
        metrics = paired_policy_metrics([0, 0, 1, 1], [1, 0, 0, 1])
        self.assertEqual(metrics["accuracy_m0"], 0.5)
        self.assertEqual(metrics["accuracy_policy"], 0.5)
        self.assertEqual(metrics["repair"], 0.25)
        self.assertEqual(metrics["harm"], 0.25)
        self.assertEqual(metrics["delta_accuracy"], metrics["repair"] - metrics["harm"])

    def test_main_metrics_reject_missing_grades(self) -> None:
        with self.assertRaises(ValueError):
            paired_policy_metrics([1, None], [1, 0])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
