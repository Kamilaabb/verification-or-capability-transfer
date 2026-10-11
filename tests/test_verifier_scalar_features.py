from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.verifier_scalar_features import (  # noqa: E402
    FEATURE_NAMES,
    LOCATION_FEATURES,
    derive_verifier_scalar_evidence,
    parse_verifier_report,
)


def trace_for(text: str) -> dict:
    tokens = []
    byte_at = 0
    for index, character in enumerate(text):
        encoded = character.encode("utf-8")
        tokens.append({
            "index": index,
            "token_id": index,
            "token": character,
            "bytes": list(encoded),
            "start_byte": byte_at,
            "end_byte": byte_at + len(encoded),
            "logprob": -0.1,
        })
        byte_at += len(encoded)
    return {
        "status": "ok",
        "tokens": tokens,
        "content_start_byte": 0,
        "content_end_byte": byte_at,
    }


def native_for(text: str, value: float) -> dict:
    return {
        "status": "ok",
        "positions": [
            {"index": index, "max_raw_logit": value}
            for index, _ in enumerate(text)
        ],
    }


def report_text(
    verdict: str = "INCORRECT",
    location: str = "$x+1=3$",
    action: str = "CRITIQUE_REFINE",
) -> str:
    return (
        f"VERDICT: {verdict}\n"
        f"ACTION: {action}\n"
        "ERROR LOCATION:\n"
        f"{location}\n"
        "END ERROR LOCATION\n"
        "CRITIQUE:\n"
        "The displayed equation is false because $x=1$ gives $2$, not $3$.\n"
        "END CRITIQUE\n"
        "CORRECTED CONTENT:\n"
        "Replace it with $x+1=2$.\n"
        "END CORRECTED CONTENT\n"
        "FINAL ANSWER:\n"
        "\\boxed{1}\n"
        "END FINAL ANSWER"
    )


REPORT = report_text()
G_TEXT = "A first line. The claim $x+1=3$ is used. Final: \\boxed{2}."


def evidence(g_text: str, v_text: str) -> dict:
    parsed = parse_verifier_report(v_text)
    report, spans = (None, {}) if parsed is None else parsed
    return derive_verifier_scalar_evidence(
        g_text=g_text,
        g_trace=trace_for(g_text),
        g_native=native_for(g_text, 7.0),
        g_answer="2",
        v_text=v_text,
        v_trace=trace_for(v_text),
        v_native=native_for(v_text, 11.0),
        report=report,
        field_spans=spans,
        v_answer=None if report is None else "1",
        answers_equivalent=None if report is None else False,
    )


def location_block(result: dict) -> dict:
    return {name: result["features"][name] for name in LOCATION_FEATURES}


def block(given: int, outcome: str | None = None) -> dict:
    values = {name: 0 for name in LOCATION_FEATURES}
    values["v_error_location_given"] = given
    if outcome is not None:
        values[f"v_error_location_{outcome}"] = 1
    return values


class VerifierReportParserTests(unittest.TestCase):
    def assert_spans_index_raw_text(self, text: str) -> dict:
        parsed = parse_verifier_report(text)
        self.assertIsNotNone(parsed)
        report, spans = parsed  # type: ignore[misc]
        for name, span in spans.items():
            raw = text[span[0]:span[1]]
            expected = report[name]
            self.assertEqual(raw, "NONE" if expected is None else expected, name)
        return report

    def test_exact_tagged_report_and_spans(self) -> None:
        report = self.assert_spans_index_raw_text(REPORT)
        self.assertEqual(report["verdict"], "INCORRECT")
        self.assertEqual(report["action"], "CRITIQUE_REFINE")
        self.assertEqual(report["error_location"], "$x+1=3$")

    def test_none_location_is_not_forced(self) -> None:
        text = report_text("CORRECT", "NONE", "KEEP")
        report, _ = parse_verifier_report(text)  # type: ignore[misc]
        self.assertIsNone(report["error_location"])

    def test_whitespace_only_deviations_parse_and_spans_stay_raw(self) -> None:
        variants = {
            "leading_blank_lines": "\n\n" + REPORT,
            "trailing_spaces": REPORT.replace("\n", "  \n"),
            "blank_lines_between_sections": REPORT.replace("\nACTION", "\n\nACTION")
            .replace("END CRITIQUE\n", "END CRITIQUE\n\n\n"),
            "value_on_header_line": REPORT.replace(
                "ERROR LOCATION:\n$x+1=3$", "ERROR LOCATION: $x+1=3$"
            ),
            "no_space_after_colon": REPORT.replace("VERDICT: ", "VERDICT:"),
        }
        for name, text in variants.items():
            with self.subTest(name):
                report = self.assert_spans_index_raw_text(text)
                self.assertEqual(report["verdict"], "INCORRECT")
                self.assertEqual(report["error_location"], "$x+1=3$")

    def test_none_on_header_line_is_none(self) -> None:
        text = report_text("CORRECT", "NONE", "KEEP").replace(
            "ERROR LOCATION:\nNONE", "ERROR LOCATION: NONE"
        ).replace("CORRECTED CONTENT:\nReplace it with $x+1=2$.",
                  "CORRECTED CONTENT: NONE")
        report = self.assert_spans_index_raw_text(text)
        self.assertIsNone(report["error_location"])
        self.assertIsNone(report["corrected_content"])

    def test_omitted_final_end_marker_is_tolerated(self) -> None:
        # Verbatim content of the c018 smoke that stopped job 65117940: the model
        # emitted EOS right after the final answer.
        smoke = (
            "VERDICT: CORRECT\nACTION: KEEP\nERROR LOCATION:\nNONE\n"
            "END ERROR LOCATION\nCRITIQUE:\nThe reasoning and final answer are correct. "
            "The computation $2+2=4$ is accurate, and the final answer is properly boxed."
            "\nEND CRITIQUE\nCORRECTED CONTENT:\nNONE\nEND CORRECTED CONTENT\n"
            "FINAL ANSWER:\n\\boxed{4}"
        )
        for name, text in {
            "c018_smoke": smoke,
            "trailing_newline": smoke + "\n",
            "synthetic": REPORT.removesuffix("\nEND FINAL ANSWER"),
        }.items():
            with self.subTest(name):
                report = self.assert_spans_index_raw_text(text)
                self.assertTrue(report["final_answer"].startswith("\\boxed{"))
                self.assertNotIn("END", report["final_answer"])

    def test_end_marker_may_be_omitted_before_the_next_header(self) -> None:
        # On c018, 110 of 300 reports omitted END CRITIQUE, most of them INCORRECT;
        # the next header already delimits the field, so these must parse unchanged.
        expected, _ = parse_verifier_report(REPORT)  # type: ignore[misc]
        markers = ["END ERROR LOCATION", "END CRITIQUE", "END CORRECTED CONTENT",
                   "END FINAL ANSWER"]
        variants = {
            f"without {marker}": REPORT.replace(f"\n{marker}", "", 1)
            for marker in markers
        }
        variants["without END CRITIQUE, blank line before next header"] = REPORT.replace(
            "\nEND CRITIQUE\n", "\n\n"
        )
        variants["without every END marker"] = REPORT
        for marker in markers:
            variants["without every END marker"] = variants[
                "without every END marker"].replace(f"\n{marker}", "", 1)
        for name, text in variants.items():
            with self.subTest(name):
                report = self.assert_spans_index_raw_text(text)
                self.assertEqual(report, expected)

    def test_token_deviations_fall_back_without_repair(self) -> None:
        variants = {
            "leading_prose": "Here is my report:\n" + REPORT,
            "trailing_prose": REPORT + "\nHope this helps.",
            "prose_after_final_marker_same_line": REPORT + " Hope this helps.",
            "missing_section_header": REPORT.replace("CORRECTED CONTENT:\n", ""),
            "truncated_before_final_answer": REPORT[:REPORT.index("FINAL ANSWER:")],
            "truncated_inside_critique": REPORT[:REPORT.index("gives $2$")],
            "markdown_bold": REPORT.replace("VERDICT:", "**VERDICT:**"),
            "code_fence": "```\n" + REPORT + "\n```",
            "echoed_choice_bars": REPORT.replace(
                "VERDICT: INCORRECT", "VERDICT: CORRECT | INCORRECT | UNCERTAIN"
            ),
            "lowercase_marker": REPORT.replace("VERDICT:", "Verdict:"),
            "end_marker_not_on_own_line": REPORT.replace(
                "$x+1=3$\nEND ERROR LOCATION", "$x+1=3$ END ERROR LOCATION"
            ),
        }
        for name, text in variants.items():
            with self.subTest(name):
                self.assertIsNone(parse_verifier_report(text))


class VerifierScalarTests(unittest.TestCase):
    def assert_location_contract(self, result: dict) -> None:
        values = location_block(result)
        if result["location_match_status"] == "report_fallback":
            self.assertTrue(all(value is None for value in values.values()))
            return
        outcomes = [values[name] for name in LOCATION_FEATURES
                    if name != "v_error_location_given"]
        self.assertEqual(sum(outcomes), values["v_error_location_given"])

    def assert_no_location_scalars(self, result: dict) -> None:
        for name in ("v_error_location_mean_logprob", "v_error_location_mean_max_logit",
                     "g_error_location_mean_logprob", "g_error_location_mean_max_logit"):
            self.assertIsNone(result["features"][name], name)
        self.assertIsNone(result["g_error_location_char_span"])
        self.assertEqual(result["g_error_location_token_indices"], [])

    def test_all_requested_feature_groups_and_exact_location(self) -> None:
        result = evidence(G_TEXT, REPORT)
        features = result["features"]
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertEqual(features["v_verdict_incorrect"], 1)
        self.assertEqual(features["v_verdict_correct"], 0)
        self.assertEqual(features["v_action_critique_refine"], 1)
        self.assertEqual(features["v_candidate_answer_parseable"], 1)
        self.assertFalse(features["gv_final_answer_equivalent"])
        self.assertAlmostEqual(features["v_verdict_incorrect_mean_logprob"], -0.1)
        self.assertIsNone(features["v_verdict_correct_mean_logprob"])
        self.assertEqual(features["v_verdict_incorrect_mean_max_logit"], 11.0)
        self.assertEqual(features["v_math_expression_mean_max_logit"], 11.0)
        self.assertEqual(result["location_match_status"], "exact")
        self.assertEqual(location_block(result), block(1, "exact"))
        self.assertEqual(features["v_error_location_mean_max_logit"], 11.0)
        self.assertEqual(features["g_error_location_mean_max_logit"], 7.0)
        start = G_TEXT.index("$x+1=3$")
        self.assertEqual(result["g_error_location_char_span"], [start, start + 7])
        self.assertTrue(result["g_error_location_token_indices"])
        self.assert_location_contract(result)

    def test_quoted_location_is_classified_under_every_verdict(self) -> None:
        for verdict in ("CORRECT", "INCORRECT", "UNCERTAIN"):
            with self.subTest(verdict):
                result = evidence(G_TEXT, report_text(verdict, action="KEEP"))
                self.assertEqual(result["location_match_status"], "exact")
                self.assertEqual(location_block(result), block(1, "exact"))
                self.assertEqual(result["features"]["g_error_location_mean_max_logit"], 7.0)

    def test_no_location_is_zero_under_every_verdict(self) -> None:
        for verdict in ("CORRECT", "INCORRECT", "UNCERTAIN"):
            with self.subTest(verdict):
                result = evidence(G_TEXT, report_text(verdict, "NONE", "KEEP"))
                self.assertEqual(result["location_match_status"], "not_given")
                self.assertEqual(location_block(result), block(0))
                self.assert_no_location_scalars(result)

    def test_whitespace_variant_is_counted_but_yields_no_span(self) -> None:
        result = evidence(G_TEXT, report_text(location="$x + 1 = 3$"))
        self.assertEqual(result["location_match_status"], "whitespace_match")
        self.assertEqual(location_block(result), block(1, "whitespace_match"))
        self.assert_no_location_scalars(result)

    def test_absent_quote_is_not_found(self) -> None:
        result = evidence(G_TEXT, report_text(location="$y=5$"))
        self.assertEqual(result["location_match_status"], "not_found")
        self.assertEqual(location_block(result), block(1, "not_found"))
        self.assert_no_location_scalars(result)

    def test_repeated_exact_quote_is_ambiguous(self) -> None:
        result = evidence("$x+1=3$ and again $x+1=3$", REPORT)
        self.assertEqual(result["location_match_status"], "ambiguous")
        self.assertEqual(location_block(result), block(1, "ambiguous"))
        self.assert_no_location_scalars(result)

    def test_repeated_after_whitespace_removal_is_ambiguous(self) -> None:
        result = evidence("$x+1=3$ and $x + 1 = 3$", report_text(location="$x  +1=3$"))
        self.assertEqual(result["location_match_status"], "ambiguous")
        self.assert_no_location_scalars(result)

    def test_overlapping_occurrences_are_ambiguous(self) -> None:
        result = evidence("aaa", report_text(location="aa"))
        self.assertEqual(result["location_match_status"], "ambiguous")

    def test_fallback_leaves_location_block_null(self) -> None:
        result = evidence(G_TEXT, "Not a tagged report.")
        self.assertEqual(result["location_match_status"], "report_fallback")
        self.assertTrue(all(value is None for value in location_block(result).values()))
        self.assert_location_contract(result)
        self.assert_no_location_scalars(result)

    def test_prompt_is_neutral_and_forbids_forced_location(self) -> None:
        prompt = (ROOT / "conditions/c018/prompts/v_report_v1.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("Do not assume that the Generator answer is wrong", prompt)
        self.assertIn("Never force or invent an error location", prompt)
        self.assertIn("ACTION: KEEP | CRITIQUE_REFINE", prompt)


if __name__ == "__main__":
    unittest.main()
