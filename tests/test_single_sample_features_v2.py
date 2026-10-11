from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.generated_token_logprobs import (  # noqa: E402
    boxed_payload_span,
    build_generated_token_trace,
    derive_scalar_logprob_features,
    json_string_value_span,
)
from authority_allocation.single_sample_features_v2 import (  # noqa: E402
    FEATURE_NAMES,
    build_single_sample_v2_payload,
)


def response_for(text: str, token_chunks: list[str]) -> dict:
    return {
        "choices": [{
            "message": {"content": text},
            "logprobs": {"content": [
                {"token": token, "bytes": list(token.encode("utf-8")), "logprob": -0.1 * (index + 1)}
                for index, token in enumerate(token_chunks)
            ]},
            "token_ids": list(range(len(token_chunks))),
        }]
    }


class GeneratedLogprobTests(unittest.TestCase):
    def test_trace_reconstructs_utf8_and_means_are_span_based(self) -> None:
        text = "α=\\boxed{2}"
        chunks = ["α", "=", "\\boxed{", "2", "}"]
        trace = build_generated_token_trace(
            response_for(text, chunks), expected_text=text, logprobs_mode="raw_logprobs"
        )
        self.assertEqual(trace["status"], "ok")
        span = boxed_payload_span(text)
        self.assertEqual(span, (9, 10))

    def test_reasoning_prefix_is_excluded_from_content_features(self) -> None:
        content = '{"verdict":"correct"}'
        full = "private reasoning\n" + content
        trace = build_generated_token_trace(
            response_for(full, ["private reasoning\n", content]),
            expected_text=content,
            logprobs_mode="raw_logprobs",
        )
        self.assertEqual(trace["status"], "ok")
        self.assertEqual(trace["content_start_byte"], len("private reasoning\n"))
        derived = derive_scalar_logprob_features(
            g_text="\\boxed{1}",
            g_trace={"status": "missing"},
            v_text=content,
            v_trace=trace,
            v_report_parse_status="ok",
        )
        self.assertEqual(derived["v_verdict_mean_logprob"], -0.2)

    def test_json_string_spans_handle_escaped_latex(self) -> None:
        text = json.dumps(
            {
                "verdict": "incorrect",
                "critique": "检查 \\sqrt{2}",
                "candidate_answer": "work \\boxed{3}",
                "requested_action": "OVERRIDE",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for key in ("verdict", "critique", "candidate_answer", "requested_action"):
            self.assertIsNotNone(json_string_value_span(text, key))
        candidate_span = json_string_value_span(text, "candidate_answer")
        self.assertIsNotNone(boxed_payload_span(text, within=candidate_span))  # type: ignore[arg-type]

    def test_v2_derivation_and_payload_have_exact_sixteen_features(self) -> None:
        g_text = "Reasoning. \\boxed{2}"
        g_trace = build_generated_token_trace(
            response_for(g_text, ["Reasoning. ", "\\boxed{", "2", "}"]),
            expected_text=g_text,
            logprobs_mode="raw_logprobs",
        )
        report = {
            "verdict": "incorrect",
            "critique": "wrong step",
            "candidate_answer": "New work. \\boxed{3}",
            "requested_action": "OVERRIDE",
        }
        v_text = json.dumps(report, separators=(",", ":"))
        v_trace = build_generated_token_trace(
            response_for(v_text, [v_text]),
            expected_text=v_text,
            logprobs_mode="raw_logprobs",
        )
        derived = derive_scalar_logprob_features(
            g_text=g_text,
            g_trace=g_trace,
            v_text=v_text,
            v_trace=v_trace,
            v_report_parse_status="ok",
        )
        payload = build_single_sample_v2_payload(
            y0_extractable=True,
            v_report=report,
            candidate_extractable=True,
            answers_equivalent=False,
            gv_answer_type_match=True,
            **derived,
        )
        self.assertEqual(set(payload["features"]), set(FEATURE_NAMES))
        self.assertEqual(len(payload["features"]), 16)
        self.assertTrue(math.isfinite(payload["features"]["g_output_mean_logprob"]))
        self.assertFalse(any(payload["missingness"].values()))

    def test_repaired_v_logprobs_are_missing(self) -> None:
        text = '{"verdict":"correct"}'
        trace = build_generated_token_trace(
            response_for(text, [text]), expected_text=text, logprobs_mode="raw_logprobs"
        )
        derived = derive_scalar_logprob_features(
            g_text="\\boxed{1}",
            g_trace=trace,
            v_text=text,
            v_trace=trace,
            v_report_parse_status="retry_ok",
        )
        self.assertIsNone(derived["v_verdict_mean_logprob"])
        self.assertIsNone(derived["v_critique_token_length_log"])


if __name__ == "__main__":
    unittest.main()
