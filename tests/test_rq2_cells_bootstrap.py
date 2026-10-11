"""RQ2 quantities recomputed from per-question data must equal rq2_cells.py's count derivation."""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
spec = importlib.util.spec_from_file_location("rq2_cells_bootstrap", ROOT / "scripts/rq2_cells_bootstrap.py")
rq2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq2)

from authority_allocation.claim_stats import Measured  # noqa: E402


def by_counts(d, stronger_is_g):
    """The formulas of scripts/rq2_cells.py, applied to counts taken straight from (g, v, gv)."""
    rows = list(d.values())
    n = len(rows)
    g = sum(r[0] for r in rows)
    v = sum(r[1] for r in rows)
    gv = sum(r[2] for r in rows)
    dis = sum(r[0] != r[1] for r in rows)
    n_a = sum(r[0] == 1 and r[1] == 0 for r in rows)
    n_b = sum(r[0] == 0 and r[1] == 1 for r in rows)
    r_a = sum(r == (1, 0, 1) for r in rows)
    r_b = sum(r == (0, 1, 1) for r in rows)
    fixed = sum(r == (0, 0, 1) for r in rows)
    broken = sum(r == (1, 1, 0) for r in rows)
    both_right = g - n_a
    follow_g = sum(r[0] != r[1] and r[2] == r[0] for r in rows)
    s = g if stronger_is_g else v
    return dict(follow_g=100 * follow_g / dis, keep_rate=100 * r_a / n_a, adopt_rate=100 * r_b / n_b,
                gain_from_v=100 * r_b / n, loss_from_g=100 * (n_a - r_a) / n, net_both=100 * (fixed - broken) / n,
                oracle_before=100 * (n - (n - both_right - dis) - max(g, v)) / n,
                oracle_after=100 * (n - (n - both_right - dis - fixed) - (n_b - r_b) - max(g, v)) / n,
                dC=100 * (gv - s) / n, dis_before=100 * dis / n,
                dis_after=100 * sum(r[0] != r[2] for r in rows) / n)


class Rq2Quantities(unittest.TestCase):
    def test_columns_reproduce_the_count_formulas(self):
        rng = random.Random(2)
        for trial in range(30):
            d = {("dev", str(i)): (int(rng.random() < 0.6), int(rng.random() < 0.5), int(rng.random() < 0.6))
                 for i in range(300)}
            for stronger_is_g in (True, False):
                keys, cols = rq2.columns(d, stronger_is_g)
                got = rq2.quantities(Measured(f"t{trial}", "math", keys, cols).record())
                want = by_counts(d, stronger_is_g)
                for key in rq2.KEYS:
                    self.assertAlmostEqual(got[key], want[key], places=9, msg=key)

    def test_rates_without_a_denominator_are_undefined(self):
        d = {("dev", str(i)): (1, 1, 1) for i in range(10)}
        keys, cols = rq2.columns(d, True)
        got = rq2.quantities(Measured("same", "math", keys, cols).record())
        self.assertIsNone(got["keep_rate"])
        self.assertIsNone(got["adopt_rate"])
        self.assertIsNone(got["follow_g"])
        self.assertEqual(rq2.equal_domain_mean([dict(domain="m", x=None), dict(domain="m", x=2.0)], "x"), 2.0)


if __name__ == "__main__":
    unittest.main()
