"""Math answers are graded and compared from the parsed text, never from extracted_answer.

extracted_answer is the sympy string of an already-parsed answer. Re-parsing it marked 45
correct c017 answers wrong and left 20 matching c018 G/V answers without an equivalence;
see docs/analysis/c018_report_format.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c002_dev30 as base  # noqa: E402
import run_c014_router as c014  # noqa: E402

SQRT_GOLD = "\\frac{\\sqrt{3}}{3}"
SQRT_RAW = "Rationalizing gives the result.\n\nFinal answer: $\\boxed{\\frac{\\sqrt{3}}{3}}$"
PI_RAW = "The circumference over the diameter is $\\boxed{\\pi}$."


class ExtractedAnswerDefectTests(unittest.TestCase):
    def test_extracted_answer_is_a_sympy_string_that_does_not_regrade(self) -> None:
        _, extracted = base.math_parse(SQRT_RAW)
        self.assertEqual(extracted, "sqrt(3)/3")
        self.assertNotEqual(base.grade_prediction(SQRT_GOLD, extracted)[0], 1)
        self.assertIsNone(base.math_equivalent(base.math_parse(PI_RAW)[1], "$\\pi$")[0])


class AnswerTextTests(unittest.TestCase):
    def test_answer_text_returns_the_parsed_text(self) -> None:
        output = {"raw_output": SQRT_RAW, "extracted_answer": "sqrt(3)/3"}
        self.assertEqual(base.answer_text(output), SQRT_RAW)
        self.assertEqual(base.grade_prediction(SQRT_GOLD, base.answer_text(output))[0], 1)

    def test_no_extracted_answer_means_no_text(self) -> None:
        self.assertIsNone(base.answer_text({"raw_output": "no answer", "extracted_answer": None}))

    def test_equivalence_uses_the_parsed_text(self) -> None:
        output = {"raw_output": PI_RAW, "extracted_answer": "pi"}
        self.assertTrue(base.math_equivalent(base.answer_text(output), "$$\\boxed{\\pi}$$")[0])

    def test_c014_self_check_grades_its_final_answer_text(self) -> None:
        check = {"extracted_answer": "pi",
                 "raw_output": json.dumps({"verdict": "correct", "final_answer": "\\boxed{\\pi}"})}
        text = c014.pass2_answer_text(check)
        self.assertEqual(text, "\\boxed{\\pi}")
        self.assertEqual(base.grade_prediction("\\pi", text)[0], 1)
        self.assertIsNone(c014.pass2_answer_text({"extracted_answer": None, "raw_output": "{}"}))
        self.assertIsNone(c014.pass2_answer_text({"extracted_answer": "pi", "raw_output": "not json"}))


if __name__ == "__main__":
    unittest.main()
