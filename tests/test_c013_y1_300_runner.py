from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import scripts.run_c013_y1_300 as c013


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ConfigurationTests(unittest.TestCase):
    def test_configures_c013_as_complete_prompt_300_row_diagnostic(self):
        self.assertEqual(c013.CONDITION_ID, "c013")
        self.assertEqual(c013.DATASET_ID, "bigcodebench-b74c0d0-complete-s20260917")
        self.assertEqual(c013.PROBLEM_FIELD, "complete_prompt")
        self.assertEqual(c013.REPLICATE_PLAN_ID, "r005")
        self.assertEqual(c013.EXPECTED_ROWS, 300)


class Y1OnlyTests(unittest.TestCase):
    def test_requests_only_existing_rewrite_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "w.jsonl", [
                {"problem_id": "keep", "raw_output": None},
                {"problem_id": "revise", "raw_output": "def f(): return 1"},
            ])
            c013.stage_grade_requests_y1(argparse.Namespace(
                rewrite_records=str(root / "w.jsonl"), run_id="run",
                output=str(root / "requests.jsonl"),
            ))
            rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([(row["task_id"], row["outcome"]) for row in rows], [("revise", "u1")])

    def test_merge_rejects_duplicate_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"problem_id": "p", "outcome": "u1"}
            write_jsonl(root / "a.jsonl", [row])
            write_jsonl(root / "b.jsonl", [row])
            with self.assertRaises(ValueError):
                c013.stage_merge_grades(argparse.Namespace(
                    input=[str(root / "a.jsonl"), str(root / "b.jsonl")],
                    output=str(root / "merged.jsonl"),
                ))


if __name__ == "__main__":
    unittest.main()
