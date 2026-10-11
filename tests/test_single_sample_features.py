from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.single_sample_features import (  # noqa: E402
    FEATURE_NAMES,
    build_single_sample_payload,
)


class SingleSampleFeatureTests(unittest.TestCase):
    def test_valid_disagreement_payload_has_exact_nine_features(self) -> None:
        payload = build_single_sample_payload(
            y0_extractable=True,
            v_report={"verdict": "incorrect", "requested_action": "OVERRIDE"},
            candidate_extractable=True,
            answers_equivalent=False,
        )
        self.assertEqual(set(payload["features"]), set(FEATURE_NAMES))
        self.assertEqual(len(payload["features"]), 9)
        self.assertEqual(payload["features"]["g_answer_extractable"], 1)
        self.assertEqual(payload["features"]["v_verdict_incorrect"], 1)
        self.assertEqual(payload["features"]["v_request_override"], 1)
        self.assertFalse(payload["features"]["gv_final_answer_equivalent"])
        self.assertFalse(any(payload["missingness"].values()))

    def test_v_fallback_registers_keep_and_marks_v_semantics_missing(self) -> None:
        payload = build_single_sample_payload(
            y0_extractable=True,
            v_report=None,
            candidate_extractable=None,
            answers_equivalent=None,
        )
        features = payload["features"]
        self.assertEqual(features["v_request_keep"], 1)
        self.assertEqual(features["v_request_revise"], 0)
        self.assertEqual(features["v_request_override"], 0)
        self.assertIsNone(features["v_verdict_correct"])
        self.assertIsNone(features["v_candidate_extractable"])
        self.assertIsNone(features["gv_final_answer_equivalent"])

    def test_unextractable_answer_makes_equivalence_missing(self) -> None:
        payload = build_single_sample_payload(
            y0_extractable=False,
            v_report={"verdict": "incorrect", "requested_action": "OVERRIDE"},
            candidate_extractable=True,
            answers_equivalent=True,
        )
        self.assertIsNone(payload["features"]["gv_final_answer_equivalent"])
        self.assertTrue(payload["missingness"]["gv_final_answer_equivalent"])

    def test_invalid_v_enum_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_single_sample_payload(
                y0_extractable=True,
                v_report={"verdict": "maybe", "requested_action": "KEEP"},
                candidate_extractable=True,
                answers_equivalent=True,
            )


if __name__ == "__main__":
    unittest.main()
