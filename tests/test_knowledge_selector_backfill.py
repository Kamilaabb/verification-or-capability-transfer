import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_knowledge_selector_backfill import compare_report, validate_decisions


class BackfillChecks(unittest.TestCase):
    def test_array_covers_exactly_the_old_thirteen_pairs(self):
        from scripts import gv_table_selector_fit as fit
        indices = [45, 48, 53, 55, 57, 60, 63, 66, 67, 69, 72, 78, 79]
        tasks = fit.tasks(False)
        actual = {frozenset((tasks[i][1], tasks[i][2])) for i in indices}
        expected = {frozenset((p[1][1], p[2][1])) for p in fit.existing_mc.PAIRS}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 13)
        self.assertTrue(all(tasks[i][0] == "knowledge" for i in indices))

    def test_equal_counts_do_not_hide_changed_source_role(self):
        old = {"pair": "p", "g": ["G", "c075", "c121", 600], "v": ["V", "c071", "c082", 900],
               "oracle": 1000, "selectors": {"primary": {"correct": 920}, "secondary": {"correct": 910}}}
        new = {"domain": "knowledge", "n": 1500, "G": 600, "V": 900, "oracle": 1000,
               "runs": ["r__c075__dev", "r__c071__dev"], "selectors": old["selectors"]}
        compare_report(old, new)
        new["runs"].reverse()
        with self.assertRaisesRegex(ValueError, "roles changed"):
            compare_report(old, new)

    def test_one_row_mismatch_stops(self):
        old = {"pair": "p", "g": ["G", "c075", "c121", 600], "v": ["V", "c071", "c082", 900],
               "oracle": 1000, "selectors": {"primary": {"correct": 920}, "secondary": {"correct": 910}}}
        new = {"domain": "knowledge", "n": 1500, "G": 600, "V": 900, "oracle": 1000,
               "runs": ["r__c075__dev", "r__c071__dev"],
               "selectors": {"primary": {"correct": 919}, "secondary": {"correct": 910}}}
        with self.assertRaisesRegex(ValueError, "919 != archived 920"):
            compare_report(old, new)

    def test_delivery_and_leakage_contract(self):
        yg = {str(i): int(i % 2 == 0) for i in range(1500)}
        yv = {str(i): int(i % 3 == 0) for i in range(1500)}
        rows = [{"problem_id": str(i), "p_g": 0.5, "p_v": 0.5, "pick_v": True} for i in range(1500)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            def write():
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            write()
            self.assertEqual(validate_decisions(path, yg, yv), sum(yv.values()))
            rows[0]["pick_v"] = False
            write()
            with self.assertRaisesRegex(ValueError, "delivery rule"):
                validate_decisions(path, yg, yv)
            rows[0]["pick_v"] = True
            rows[0]["correct"] = 1
            write()
            with self.assertRaisesRegex(ValueError, "contains labels"):
                validate_decisions(path, yg, yv)


if __name__ == "__main__":
    unittest.main()
