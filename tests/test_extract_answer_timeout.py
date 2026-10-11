"""A degenerate output whose answer math_verify cannot parse in time is recorded, not fatal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c014_router as shared  # noqa: E402

HAS_MATH_VERIFY = importlib.util.find_spec("math_verify") is not None


@unittest.skipUnless(HAS_MATH_VERIFY, "math_verify is installed on the cluster, not required locally")
class ExtractAnswerTests(unittest.TestCase):
    def test_parser_timeout_records_no_answer_and_model_invalid(self) -> None:
        from math_verify.errors import TimeoutException

        with patch.object(shared.base, "math_parse", side_effect=TimeoutException("slow")):
            self.assertEqual(shared.extract_answer("x = \\boxed{1} = \\boxed{1}"), (None, "model_invalid"))

    def test_other_parser_errors_still_stop_the_run(self) -> None:
        with patch.object(shared.base, "math_parse", side_effect=ValueError("unexpected")):
            with self.assertRaises(ValueError):
                shared.extract_answer("\\boxed{1}")

    def test_integer_over_the_digit_limit_records_model_invalid(self) -> None:
        from sympy.core.sympify import SympifyError

        huge = "5" + "0" * 11858
        self.assertEqual(shared.extract_answer(f"The answer is {huge}"), (None, "model_invalid"))
        with patch.object(shared.base, "math_parse", side_effect=SympifyError("x", ValueError("other"))):
            with self.assertRaises(SympifyError):
                shared.extract_answer("\\boxed{1}")

    def test_grading_timeout_on_the_prediction_is_wrong_not_fatal(self) -> None:
        from math_verify.errors import TimeoutException

        with patch("math_verify.verify", side_effect=TimeoutException("slow")):
            self.assertEqual(shared.base.grade_prediction("1", "\\boxed{1}"), (0, "grader_timeout"))

    def test_grading_timeout_on_the_gold_stays_a_grader_error(self) -> None:
        from math_verify.errors import TimeoutException

        with patch.object(shared.base, "math_parse_gold", side_effect=TimeoutException("slow")):
            self.assertEqual(shared.base.grade_prediction("1", "\\boxed{1}"), (None, "grader_error"))

    def test_normal_and_empty_outputs_are_unchanged(self) -> None:
        with patch.object(shared.base, "math_parse", return_value=([1], "1")):
            self.assertEqual(shared.extract_answer("\\boxed{1}"), ("1", "ok"))
        self.assertEqual(shared.extract_answer(""), (None, "model_invalid"))
        self.assertEqual(shared.extract_answer(None), (None, "model_invalid"))


if __name__ == "__main__":
    unittest.main()
