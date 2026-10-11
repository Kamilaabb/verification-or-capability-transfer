from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.code_scalar_features import (
    FEATURE_NAMES,
    build_code_scalar_payload,
)


class CodeScalarFeatureTests(unittest.TestCase):
    def _build(self, report=None, status="ok"):
        if report is None and status == "ok":
            report = {
                "verdict": "incorrect",
                "critique": "The function signature misses an empty edge case.",
                "candidate_answer": "def solve(x):\n    if not x:\n        return 0\n    return max(x)\n",
            }
        with patch(
            "authority_allocation.code_scalar_features.derive_scalar_logprob_features_v3",
            return_value={
                "g_output_mean_logprob": -0.2,
                "v_verdict_mean_logprob": -0.1,
                "v_request_mean_logprob": -0.3,
                "v_candidate_answer_mean_logprob": -0.4,
            },
        ):
            return build_code_scalar_payload(
                problem_text="Return the maximum integer.",
                g_text="```python\ndef solve(x):\n    return x[0]\n```",
                g_trace={},
                v_text="{}",
                v_trace={},
                v_report=report,
                v_report_parse_status=status,
                registered_request="CRITIQUE_REVISE",
            )

    def test_exact_registered_membership_and_information(self) -> None:
        payload = self._build()
        self.assertEqual(tuple(payload["features"]), FEATURE_NAMES)
        self.assertEqual(len(FEATURE_NAMES), 21)
        self.assertEqual(payload["features"]["v_verdict_incorrect"], 1)
        self.assertEqual(payload["features"]["critique_category_interface"], 1)
        self.assertEqual(payload["features"]["critique_category_edge_case"], 1)
        self.assertGreater(payload["features"]["gv_control_flow_log_abs_delta"], 0)
        self.assertFalse(any(payload["missingness"].values()))

    def test_fallback_marks_report_derived_features_missing(self) -> None:
        payload = self._build(report=None, status="fallback")
        for name in (
            "v_verdict_correct",
            "critique_words_log",
            "gv_unique_call_names_log_abs_delta",
            "critique_category_logic_algorithm",
        ):
            self.assertIsNone(payload["features"][name])
            self.assertTrue(payload["missingness"][name])
        self.assertEqual(payload["features"]["v_request_keep"], 0)


if __name__ == "__main__":
    unittest.main()
