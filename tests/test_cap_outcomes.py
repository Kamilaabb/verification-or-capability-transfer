from __future__ import annotations

import itertools
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.cap_outcomes import build_cap_outcome_target  # noqa: E402


class CapOutcomeTargetTests(unittest.TestCase):
    def test_all_eight_patterns_are_lossless(self) -> None:
        caps = ("M0", "M1", "M2")
        for pattern in itertools.product((0, 1), repeat=3):
            with self.subTest(pattern=pattern):
                target = build_cap_outcome_target(*pattern)
                self.assertEqual(
                    target["cap_outcomes"], dict(zip(caps, pattern, strict=True))
                )
                expected_actions = [
                    cap for cap, outcome in zip(caps, pattern, strict=True) if outcome
                ]
                self.assertEqual(target["oracle_actions"], expected_actions)
                self.assertEqual(
                    target["minimal_oracle_action"],
                    expected_actions[0] if expected_actions else None,
                )

    def test_all_correct_remains_multilabel(self) -> None:
        target = build_cap_outcome_target(1, 1, 1)
        self.assertEqual(target["oracle_actions"], ["M0", "M1", "M2"])
        self.assertEqual(target["minimal_oracle_action"], "M0")

    def test_unfixable_is_all_zero(self) -> None:
        target = build_cap_outcome_target(0, 0, 0)
        self.assertEqual(target["oracle_actions"], [])
        self.assertEqual(target["oracle_class"], "unfixable")

    def test_missing_or_nonbinary_outcomes_fail_closed(self) -> None:
        for values in (
            (1, 0, None),
            (1, -1, 0),
            (1, 2, 0),
            (1, True, 0),
            (1, 1.0, 0),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_cap_outcome_target(*values)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
