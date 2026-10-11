"""Five-level outcome placement and the domain-stratified rank test used by H09."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from authority_allocation.outcome_levels import outcome_level, spearman, stratified_spearman  # noqa: E402


class OutcomeLevels(unittest.TestCase):
    def test_each_level(self):
        self.assertEqual(outcome_level((-3, -1), (-9, -6)), 1)
        self.assertEqual(outcome_level((-1, 1), (-5, -2)), 2)
        self.assertEqual(outcome_level((2, 6), (-5, -1)), 3)
        self.assertEqual(outcome_level((2, 6), (-1, 1)), 4)
        self.assertEqual(outcome_level((5, 9), (0.5, 2)), 5)

    def test_weaker_reviews_has_three_levels(self):
        # G is the stronger model, so the two intervals are the same.
        for ci, level in [((-3, -1), 1), ((-1, 1), 2), ((1, 3), 5)]:
            self.assertEqual(outcome_level(ci, ci), level)

    def test_spearman_with_ties(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [0, 0, 1, 1]), 0.8944271909999159)
        self.assertIsNone(spearman([1, 2, 3], [1, 1, 1]))

    def test_stratified_permutation(self):
        groups = {'a': [(i, int(i > 5)) for i in range(10)], 'b': [(i, int(i > 3)) for i in range(8)]}
        res = stratified_spearman(groups, reps=999, seed=1)
        self.assertGreater(res['point'], 0.8)
        self.assertLess(res['p_two_sided'], 0.01)
        flat = {'a': [(i, 1) for i in range(5)]}
        self.assertIsNone(stratified_spearman(flat, reps=10, seed=1)['point'])


if __name__ == '__main__':
    unittest.main()
