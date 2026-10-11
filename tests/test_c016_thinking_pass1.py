from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
import run_c016_thinking_pass1 as c016  # noqa: E402


class C016ConfigurationTests(unittest.TestCase):
    def test_frozen_thinking_on_contract(self):
        cfg = resolve_condition(ROOT / "conditions/c016/condition.yaml", repository_root=ROOT)
        self.assertTrue(cfg["model"]["thinking"])
        self.assertEqual(cfg["model"]["reasoning_parser"], "qwen3")
        self.assertEqual(cfg["model"]["pass1_decoding"]["max_new_tokens"], 4096)
        self.assertEqual(cfg["model"]["pass1_decoding"]["temperature"], 0.0)
        self.assertEqual(cfg["replicates"]["call_seed"], 42)
        self.assertFalse(cfg["model"]["finalizer"]["enabled"])
        self.assertEqual(cfg["grading"]["length_finish_policy"],
                         "grade_if_answer_extractable_otherwise_wrong")

    def test_extractable_length_finish_is_graded_normally(self):
        record = {"output": {"extracted_answer": "7", "raw_output": "So \\boxed{7}.",
                             "usage": {"finish_reason": "length"}}}
        with patch.object(c016.base, "grade_prediction", return_value=(1, "ok")) as grade:
            self.assertEqual(c016.grade_one("7", record), (1, "ok"))
        # The raw text is graded, not the extracted sympy string.
        grade.assert_called_once_with("7", "So \\boxed{7}.")

    def test_unextractable_length_finish_is_wrong(self):
        record = {"output": {"extracted_answer": None,
                             "usage": {"finish_reason": "length"}}}
        with patch.object(c016.base, "grade_prediction") as grade:
            self.assertEqual(c016.grade_one("7", record), (0, "model_invalid"))
        grade.assert_not_called()


if __name__ == "__main__":
    unittest.main()
