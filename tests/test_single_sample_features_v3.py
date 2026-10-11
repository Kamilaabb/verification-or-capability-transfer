from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.generated_token_logprobs import (  # noqa: E402
    build_generated_token_trace,
    derive_scalar_logprob_features_v3,
)
from authority_allocation.single_sample_features_v3 import (  # noqa: E402
    FEATURE_NAMES,
    build_single_sample_v3_payload,
)


def response_for(text: str, token_chunks: list[str]) -> dict:
    return {
        "choices": [{
            "message": {"content": text},
            "logprobs": {"content": [
                {
                    "token": token,
                    "bytes": list(token.encode("utf-8")),
                    "logprob": -0.1 * (index + 1),
                }
                for index, token in enumerate(token_chunks)
            ]},
            "token_ids": list(range(len(token_chunks))),
        }]
    }


class ScalarSingleV3Tests(unittest.TestCase):
    def test_candidate_field_logprob_does_not_require_boxed_span(self) -> None:
        report = {
            "verdict": "incorrect",
            "critique": "wrong step",
            "candidate_answer": "Complete reasoning; final answer is 3.",
            "requested_action": "OVERRIDE",
        }
        text = json.dumps(report, separators=(",", ":"))
        trace = build_generated_token_trace(
            response_for(text, [text]),
            expected_text=text,
            logprobs_mode="raw_logprobs",
        )
        derived = derive_scalar_logprob_features_v3(
            g_text="\\boxed{2}",
            g_trace={"status": "missing"},
            v_text=text,
            v_trace=trace,
            v_report_parse_status="ok",
        )
        self.assertEqual(derived["v_candidate_answer_mean_logprob"], -0.1)
        self.assertNotIn("v_candidate_final_answer_mean_logprob", derived)

    def test_candidate_field_likelihood_survives_math_parse_failure(self) -> None:
        payload = build_single_sample_v3_payload(
            y0_extractable=True,
            v_report={
                "verdict": "uncertain",
                "critique": "cannot parse",
                "candidate_answer": "narrative only",
                "requested_action": "KEEP",
            },
            candidate_extractable=False,
            answers_equivalent=None,
            g_output_mean_logprob=-0.2,
            g_final_answer_mean_logprob=-0.3,
            v_verdict_mean_logprob=-0.4,
            v_request_mean_logprob=-0.5,
            v_candidate_answer_mean_logprob=-0.6,
            v_critique_token_length_log=1.0,
            gv_answer_type_match=None,
        )
        self.assertEqual(set(payload["features"]), set(FEATURE_NAMES))
        self.assertEqual(payload["features"]["v_candidate_answer_mean_logprob"], -0.6)
        self.assertFalse(payload["missingness"]["v_candidate_answer_mean_logprob"])

    def test_retry_keeps_v3_v_likelihoods_missing(self) -> None:
        text = '{"candidate_answer":"3"}'
        trace = build_generated_token_trace(
            response_for(text, [text]),
            expected_text=text,
            logprobs_mode="raw_logprobs",
        )
        derived = derive_scalar_logprob_features_v3(
            g_text="\\boxed{2}",
            g_trace={"status": "missing"},
            v_text=text,
            v_trace=trace,
            v_report_parse_status="retry_ok",
        )
        self.assertIsNone(derived["v_candidate_answer_mean_logprob"])


if __name__ == "__main__":
    unittest.main()
