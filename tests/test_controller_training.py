from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.controller_training import choose_action, explicit_missing_matrix


class ControllerTrainingTests(unittest.TestCase):
    def test_explicit_missing_matrix_has_value_and_flag_for_each_feature(self) -> None:
        matrix = explicit_missing_matrix(
            [{"a": 1.0, "b": None}, {"a": None, "b": 2.0}], ("a", "b")
        )
        self.assertEqual(matrix.shape, (2, 4))
        np.testing.assert_array_equal(matrix[:, 2:], [[0.0, 1.0], [1.0, 0.0]])
        self.assertTrue(np.isnan(matrix[0, 1]))

    def test_exact_ties_and_tolerance_choose_lower_authority(self) -> None:
        probabilities = {"M0": 0.60, "M1": 0.61, "M2": 0.61}
        self.assertEqual(choose_action(probabilities, ("M0", "M1", "M2")), "M1")
        self.assertEqual(
            choose_action(
                probabilities,
                ("M0", "M1", "M2"),
                lower_authority_tolerance=0.02,
            ),
            "M0",
        )


if __name__ == "__main__":
    unittest.main()
