"""Backfill must preserve frozen ensemble behavior and reject corrupt records."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import code_selector_backfill as backfill


class CodeBackfillTests(unittest.TestCase):
    def test_retaining_predictions_preserves_original_ensemble_and_v_ties(self):
        cps = backfill.cps
        ids, yg, yv = ["a", "b"], np.asarray([1, 0]), np.asarray([0, 1])
        fsets = {"F2": {"g": ["g"], "v": ["v"]}, "F3": {"g": ["g"], "v": ["v"]}}
        def crossfit(xg, xv, yg, yv, family, form, seed):
            return np.asarray([.5, .7]), np.asarray([.5, .2])
        with patch.object(cps.sel, "matrix", return_value=np.zeros((2, 1))), patch.object(cps.sel, "crossfit", side_effect=crossfit) as fit:
            old = cps.ensemble({}, ids, yg, yv, fsets, cps.MEMBERS["primary"])
            new = cps.ensemble({}, ids, yg, yv, fsets, cps.MEMBERS["primary"], retain_predictions=True)
        self.assertEqual(fit.call_count, 20)
        self.assertEqual(old["per_seed"], new["per_seed"])
        self.assertEqual(old["mean"], new["mean"])
        np.testing.assert_array_equal(old["rows"], new["rows"])
        self.assertEqual(old["rows"][0], yv[0])
        self.assertNotIn("p_g", old)
        self.assertEqual(new["seed_p_g"].shape, (5, 2))

    def validate(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            return backfill.validate_decisions(path, ["a"], np.asarray([1]), np.asarray([0]), np.zeros((5, 1), dtype=int))

    def records(self):
        return [dict(problem_id="a", p_g=.5, p_v=.5, pick_v=True,
            per_seed=[dict(seed=s, fold=0, p_g=.5, p_v=.5) for s in backfill.cps.sel.SEEDS])]

    def test_seed_counts_reconstructed_with_same_tie_rule(self):
        yy, seeds = self.validate(self.records())
        self.assertEqual(yy.tolist(), [0])
        self.assertEqual(seeds, [0] * 5)

    def test_corruption_is_rejected(self):
        for kind in ("label", "tie", "seed", "fold", "nan", "mean", "missing", "duplicate"):
            with self.subTest(kind=kind):
                rows = copy.deepcopy(self.records())
                r = rows[0]
                if kind == "label": r["u"] = 1
                elif kind == "tie": r["pick_v"] = False
                elif kind == "seed": r["per_seed"][0]["seed"] += 1
                elif kind == "fold": r["per_seed"][0]["fold"] = 1
                elif kind == "nan": r["p_g"] = float("nan")
                elif kind == "mean": r["p_g"] = .4
                elif kind == "missing": rows = []
                elif kind == "duplicate": rows.append(copy.deepcopy(r))
                with self.assertRaises(ValueError): self.validate(rows)

    def test_original_stratified_fold_assignments_are_retained(self):
        yg = np.tile([0, 0, 1, 1], 10)
        yv = np.tile([0, 1, 0, 1], 10)
        folds = backfill.assignments(yg, yv)
        self.assertEqual(folds.shape, (5, 40))
        for row in folds:
            self.assertEqual(set(row), set(range(5)))
            self.assertEqual(np.bincount(row).tolist(), [8] * 5)


if __name__ == "__main__":
    unittest.main()
