import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.run_c002_dev30 import (
    PROMPTS,
    audit_gold_answers,
    format_repair_call,
    math_parse_gold,
    manifest_fields,
    parse_json_object,
    render,
    split_prompt,
    stage_analyze,
    validate_v_report,
)


class C002Dev30AnalysisTests(unittest.TestCase):
    def test_bare_math500_latex_gold_is_parsed_as_math(self):
        with patch("scripts.run_c002_dev30.math_parse", return_value=(["parsed"], "parsed")) as parse:
            parsed, _ = math_parse_gold(r"\frac{\sqrt{a}}{\sqrt{b}}")
        self.assertTrue(parsed)
        parse.assert_called_once_with(r"$\frac{\sqrt{a}}{\sqrt{b}}$")

    def test_gold_audit_accepts_bare_latex_answers(self):
        with patch("scripts.run_c002_dev30.math_parse", return_value=(["parsed"], "parsed")):
            audit_gold_answers({"test/algebra/example.json": {"gold": r"\frac{1}{2}"}})

    def test_http_rejection_of_format_repair_returns_fallback_signal(self):
        error = HTTPError("http://localhost", 400, "Bad Request", {}, None)
        with patch("authority_allocation.api.api_call", side_effect=error):
            self.assertIsNone(format_repair_call())

    def test_render_accepts_nested_latex_braces_from_actual_math_problem(self):
        problem = r"\frac{\sqrt{a b + a}}{\sqrt{a b - b}}"
        rendered = render("Problem:\n{{problem}}", problem=problem)
        self.assertEqual(rendered, f"Problem:\n{problem}")

    def test_render_ignores_non_placeholder_braces(self):
        latex = r"\frac{\sqrt{a}}{\sqrt{b}}"
        literal = 'JSON: {"value":"{{not-a-variable}}"}'
        self.assertEqual(render(latex), latex)
        self.assertEqual(render(literal), literal)

    def test_render_reports_missing_original_template_variables(self):
        with self.assertRaisesRegex(ValueError, "missing prompt template variables: critique, y0"):
            render("{{problem}} {{y0}} {{critique}}", problem="test")

    def test_all_c002_prompt_templates_accept_braces_in_inserted_values(self):
        values = {
            "problem": r"\frac{\sqrt{a b + a}}{\sqrt{a b - b}}",
            "y0": r"Reasoning with {{literal text}}. \boxed{1}",
            "critique": r"Compare \frac{\sqrt{a}}{\sqrt{b}}.",
            "controller_input_json": '{"features":{"note":"{{literal}}"}}',
        }
        for name, path in PROMPTS.items():
            with self.subTest(prompt=name):
                _, template = split_prompt(path)
                rendered = render(template, **values)
                self.assertIsInstance(rendered, str)

    def test_frozen_id_only_manifest_is_accepted(self):
        problem_id, metadata = manifest_fields({
            "problem_id": "test/algebra/1.json",
            "subject": "Algebra",
            "level": 3,
        })
        self.assertEqual(problem_id, "test/algebra/1.json")
        self.assertEqual(metadata, {"subject": "Algebra", "level": 3})

    def test_analysis_writes_passed_human_report_for_complete_dev30(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counterfactuals = root / "counterfactuals.jsonl"
            decisions = root / "controller_decisions.jsonl"
            counterfactual_rows = []
            decision_rows = []
            for index in range(30):
                problem_id = f"p{index:02d}"
                u0 = index % 2
                counterfactual_rows.append({
                    "problem_id": problem_id,
                    "registered_request": "CRITIQUE_REVISE",
                    "u0": u0,
                    "z0": u0,
                    "z1": 1,
                    "z2": 1,
                    "grader_status": {"y0": "ok", "y1": "ok", "y2": "ok"},
                })
                for menu in ("binary", "ternary"):
                    decision_rows.append({
                        "problem_id": problem_id,
                        "menu": menu,
                        "replicate_id": "c01",
                        "authority_cap": "M0",
                        "parse_status": "ok",
                    })
            counterfactuals.write_text(
                "".join(json.dumps(row) + "\n" for row in counterfactual_rows),
                encoding="utf-8",
            )
            decisions.write_text(
                "".join(json.dumps(row) + "\n" for row in decision_rows),
                encoding="utf-8",
            )

            reports = root / "reports"
            stage_analyze(argparse.Namespace(
                counterfactuals=str(counterfactuals),
                controller_records=str(decisions),
                output_dir=str(reports),
            ))

            qa = json.loads((reports / "qa_report.json").read_text(encoding="utf-8"))
            metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(qa["status"], "passed")
            self.assertEqual(qa["counterfactual_rows"], 30)
            self.assertEqual(qa["controller_rows"], 60)
            self.assertIn("|M0|", (reports / "summary.md").read_text(encoding="utf-8"))
            self.assertEqual(metrics["diagnostics"]["sample_oracle_correct"], 30)
            self.assertEqual(metrics["diagnostics"]["oracle_gap"]["M1"], 0)
            self.assertEqual(metrics["diagnostics"]["repair_overlap"]["M1_and_M2"], 15)
            self.assertEqual(metrics["diagnostics"]["oracle_class_counts"]["M0"], 0)
            self.assertEqual(
                metrics["diagnostics"]["oracle_class_counts"]["M1_and_M2"], 15
            )
            self.assertEqual(
                metrics["diagnostics"]["oracle_class_counts"]["M0_and_M1_and_M2"],
                15,
            )
            oracle_rows = [
                json.loads(line)
                for line in (reports / "oracle_assignments.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(oracle_rows), 30)
            self.assertEqual(oracle_rows[0]["oracle_actions"], ["M1", "M2"])
            self.assertEqual(oracle_rows[0]["minimal_oracle_action"], "M1")
            self.assertEqual(oracle_rows[1]["oracle_actions"], ["M0", "M1", "M2"])
            self.assertEqual(oracle_rows[1]["minimal_oracle_action"], "M0")
            self.assertIn(
                "Sample oracle",
                (reports / "summary.md").read_text(encoding="utf-8"),
            )


class ReportParsingTests(unittest.TestCase):
    """The grammar backend emits raw tabs inside string values; strict JSON rejects them."""

    def _report(self, candidate: str) -> str:
        document = json.dumps(
            {
                "verdict": "incorrect",
                "critique": "The angle identity is misapplied.",
                "candidate_answer": candidate,
                "requested_action": "OVERRIDE",
            }
        )
        return document.replace("\\t", "\t")

    def test_raw_tab_inside_string_value_is_parsed(self):
        report = parse_json_object(self._report("Use $ \text{cos} $ here."))
        self.assertIsNotNone(report)
        self.assertTrue(validate_v_report(report))
        self.assertIn("\text{cos}", report["candidate_answer"])

    def test_raw_tab_survives_the_embedded_object_scan(self):
        raw = "Here is the report: " + self._report("A $ \text{sin} $ step.")
        report = parse_json_object(raw)
        self.assertIsNotNone(report)
        self.assertTrue(validate_v_report(report))

    def test_malformed_object_still_returns_none(self):
        self.assertIsNone(parse_json_object('{"verdict": "correct", '))


if __name__ == "__main__":
    unittest.main()
