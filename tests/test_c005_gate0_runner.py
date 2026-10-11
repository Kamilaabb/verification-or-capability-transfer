from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scripts.run_c005_gate0 as gate0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def g_row(problem_id: str, raw_output: str = "```python\ndef f(): pass\n```") -> dict:
    return {
        "run_id": "run",
        "problem_id": problem_id,
        "problem_text": "task",
        "outputs": [{
            "sample_index": 0,
            "raw_output": raw_output,
            "usage": {"finish_reason": "stop", "input_tokens": 1, "output_tokens": 1},
        }],
    }


def v_row(problem_id: str, report: dict | None, parse_status: str = "ok") -> dict:
    return {
        "problem_id": problem_id,
        "cell_id": "V-A",
        "v_report": report,
        "report_parse_status": parse_status,
        "registered_request": "KEEP" if report is None else report["requested_action"],
        "usage": {"finish_reason": "stop", "input_tokens": 1, "output_tokens": 1},
    }


def report(action: str, candidate: str = "def f(): pass") -> dict:
    return {
        "verdict": "incorrect",
        "critique": "c",
        "candidate_answer": candidate,
        "requested_action": action,
    }


def grade_row(problem_id: str, outcome: str, u, status: str, detail: str) -> dict:
    return {
        "problem_id": problem_id,
        "outcome": outcome,
        "u": u,
        "status": status,
        "detail": detail,
        "seconds": 0.1,
    }


class GradeRequestTests(unittest.TestCase):
    def test_unparsed_report_gets_no_candidate_grading_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "g.jsonl", [g_row("BigCodeBench/1"), g_row("BigCodeBench/2")])
            write_jsonl(root / "v.jsonl", [
                v_row("BigCodeBench/1", report("OVERRIDE")),
                v_row("BigCodeBench/2", None, parse_status="fallback"),
            ])
            gate0.stage_grade_requests(argparse.Namespace(
                g_records=str(root / "g.jsonl"),
                v_records=str(root / "v.jsonl"),
                run_id="run",
                output=str(root / "requests.jsonl"),
            ))
            rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]

        outcomes = {(row["task_id"], row["outcome"]) for row in rows}
        self.assertIn(("BigCodeBench/1", "u0"), outcomes)
        self.assertIn(("BigCodeBench/1", "u2"), outcomes)
        self.assertIn(("BigCodeBench/2", "u0"), outcomes)
        # A fallback report has no candidate program; M2 would deploy y0, so scoring an
        # absent program as wrong would depress mean(u2) with a parsing failure.
        self.assertNotIn(("BigCodeBench/2", "u2"), outcomes)

    def test_candidate_is_requested_even_when_the_verifier_asked_to_keep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "g.jsonl", [g_row("BigCodeBench/1")])
            write_jsonl(root / "v.jsonl", [v_row("BigCodeBench/1", report("KEEP"))])
            gate0.stage_grade_requests(argparse.Namespace(
                g_records=str(root / "g.jsonl"),
                v_records=str(root / "v.jsonl"),
                run_id="run",
                output=str(root / "requests.jsonl"),
            ))
            rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]

        # Grading the candidate unconditionally is what makes P(u2=1 | request=KEEP)
        # computable — whether the verifier sat on an answer that was in fact correct.
        self.assertIn(("BigCodeBench/1", "u2"), {(r["task_id"], r["outcome"]) for r in rows})


class GradeIngestTests(unittest.TestCase):
    def test_missing_result_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "requests.jsonl", [
                {"record_id": "a", "task_id": "BigCodeBench/1", "outcome": "u0", "text": "x"},
                {"record_id": "b", "task_id": "BigCodeBench/1", "outcome": "u2", "text": "y"},
            ])
            write_jsonl(root / "results.jsonl", [
                {"record_id": "a", "task_id": "BigCodeBench/1", "u": 1,
                 "status": "ok", "detail": "pass", "seconds": 0.1},
            ])
            with patch.object(gate0, "condition_hash", return_value="hash"):
                with self.assertRaises(RuntimeError) as caught:
                    gate0.stage_grade(argparse.Namespace(
                        requests=str(root / "requests.jsonl"),
                        results=str(root / "results.jsonl"),
                        run_id="run",
                        output=str(root / "grades.jsonl"),
                    ))
        self.assertIn("no result", str(caught.exception))


class AnalyzeTests(unittest.TestCase):
    def run_analyze(self, root: Path, g_rows, v_rows, grade_rows):
        write_jsonl(root / "g.jsonl", g_rows)
        write_jsonl(root / "v.jsonl", v_rows)
        write_jsonl(root / "grades.jsonl", grade_rows)
        gate0.stage_analyze(argparse.Namespace(
            g_records=str(root / "g.jsonl"),
            v_records=str(root / "v.jsonl"),
            grade_records=str(root / "grades.jsonl"),
            run_id="run",
            output_dir=str(root / "reports"),
        ))
        return json.loads((root / "reports" / "gate0_summary.json").read_text(encoding="utf-8"))

    def test_not_run_candidate_is_excluded_from_mean_rather_than_scored_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.run_analyze(
                root,
                [g_row("BigCodeBench/1"), g_row("BigCodeBench/2")],
                [
                    v_row("BigCodeBench/1", report("OVERRIDE")),
                    v_row("BigCodeBench/2", None, parse_status="fallback"),
                ],
                [
                    grade_row("BigCodeBench/1", "u0", 0, "ok", "fail"),
                    grade_row("BigCodeBench/1", "u2", 1, "ok", "pass"),
                    grade_row("BigCodeBench/2", "u0", 1, "ok", "pass"),
                ],
            )
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["u2_not_run"], 1)
        self.assertEqual(summary["mean_u2_denominator"], 1)
        # Scored 1 of 1, not 1 of 2.
        self.assertEqual(summary["mean_u2"], 1.0)
        self.assertEqual(summary["acc_G"], 0.5)

    def test_grader_error_blocks_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError) as caught:
                self.run_analyze(
                    root,
                    [g_row("BigCodeBench/1")],
                    [v_row("BigCodeBench/1", report("OVERRIDE"))],
                    [
                        grade_row("BigCodeBench/1", "u0", None, "grader_error", "driver_error:X"),
                        grade_row("BigCodeBench/1", "u2", 1, "ok", "pass"),
                    ],
                )
        self.assertIn("grader error", str(caught.exception))

    def test_timeout_counts_as_a_graded_wrong_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.run_analyze(
                root,
                [g_row("BigCodeBench/1")],
                [v_row("BigCodeBench/1", report("OVERRIDE"))],
                [
                    grade_row("BigCodeBench/1", "u0", 0, "ok", "timeout"),
                    grade_row("BigCodeBench/1", "u2", 1, "ok", "pass"),
                ],
            )
        # A hang must not null the outcome, or one wrong answer would void the run.
        self.assertEqual(summary["acc_G"], 0.0)
        self.assertEqual(summary["grader_timeouts"], 1)

    def test_over_correction_rate_uses_the_correct_answer_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.run_analyze(
                root,
                [g_row(f"BigCodeBench/{index}") for index in range(1, 5)],
                [
                    v_row("BigCodeBench/1", report("OVERRIDE")),
                    v_row("BigCodeBench/2", report("KEEP")),
                    v_row("BigCodeBench/3", report("KEEP")),
                    v_row("BigCodeBench/4", report("OVERRIDE")),
                ],
                [
                    grade_row("BigCodeBench/1", "u0", 1, "ok", "pass"),
                    grade_row("BigCodeBench/1", "u2", 0, "ok", "fail"),
                    grade_row("BigCodeBench/2", "u0", 1, "ok", "pass"),
                    grade_row("BigCodeBench/2", "u2", 1, "ok", "pass"),
                    grade_row("BigCodeBench/3", "u0", 0, "ok", "fail"),
                    grade_row("BigCodeBench/3", "u2", 0, "ok", "fail"),
                    grade_row("BigCodeBench/4", "u0", 0, "ok", "fail"),
                    grade_row("BigCodeBench/4", "u2", 1, "ok", "pass"),
                ],
            )
        # Two correct answers, one of them acted on; two wrong answers, one acted on.
        self.assertEqual(summary["over_correction_denominator"], 2)
        self.assertEqual(summary["over_correction_rate"], 0.5)
        self.assertEqual(summary["error_coverage_denominator"], 2)
        self.assertEqual(summary["error_coverage_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
