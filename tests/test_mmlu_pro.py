"""MMLU-Pro helpers and the study's frozen prompts (docs/plan/mmlu_pro_study.md section 4).
Any edit to a frozen prompt file fails here, which stops every sbatch before a job starts."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import mmlu_pro  # noqa: E402
from authority_allocation.prompts import PROMPT_VARIABLE, split_prompt  # noqa: E402

LAST_LINE = ('The last line of your response must be of the form "Answer: X", where X is the '
             'letter of exactly one option.')


class FrozenPromptTests(unittest.TestCase):
    def test_frozen_files_keep_their_registered_hash(self) -> None:
        for relative in mmlu_pro.FROZEN_PROMPTS:
            mmlu_pro.require_frozen_prompt(ROOT, relative)

    def test_solve_prompt_renders_to_the_registered_messages(self) -> None:
        system, user = split_prompt(ROOT / mmlu_pro.MC_SOLVE_PROMPT)
        self.assertEqual(system, "Answer the multiple-choice question provided by the user. "
                                 "Think step by step before answering.\n" + LAST_LINE)
        self.assertEqual(set(PROMPT_VARIABLE.findall(user)), {"question", "options"})
        self.assertEqual(mmlu_pro.user_message(user, question="Q {{x}}?", options=["a", "b"]),
                         "Question:\nQ {{x}}?\n\nOptions:\nA. a\nB. b")

    def test_review_prompt_renders_to_the_registered_messages(self) -> None:
        system, user = split_prompt(ROOT / mmlu_pro.MC_REVIEW_PROMPT)
        self.assertEqual(system, (
            "Review the proposed solution to the multiple-choice question for correctness.\n"
            "The proposed solution may be correct or incorrect. Carefully check its facts, its "
            "reasoning, and its final choice.\n"
            "If the proposed solution is correct, retain its answer. If it contains any error, "
            "correct the error and produce a correct solution.\n"
            "Return a complete final solution to the original question. " + LAST_LINE))
        self.assertEqual(set(PROMPT_VARIABLE.findall(user)), {"question", "options", "y0"})
        self.assertEqual(mmlu_pro.user_message(user, question="Q", options=["a", "b"], y0="S\nAnswer: B"),
                         "Question:\nQ\n\nOptions:\nA. a\nB. b\n\nProposed solution:\nS\nAnswer: B")

    def test_frozen_prompts_carry_no_experiment_framing(self) -> None:
        for relative in mmlu_pro.FROZEN_PROMPTS:
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
            for word in ("generator", "verifier", "experiment", "authority", "another model", "image"):
                self.assertNotIn(word, text, f"{relative} mentions {word!r}")

    def test_template_variables_must_match_the_call(self) -> None:
        _, user = split_prompt(ROOT / mmlu_pro.MC_SOLVE_PROMPT)
        with self.assertRaises(ValueError):
            mmlu_pro.user_message(user, question="Q", options=["a"], y0="extra")


class OfficialParserTests(unittest.TestCase):
    def test_extraction_chain(self) -> None:
        self.assertEqual(mmlu_pro.official_parse_without_random("so the answer is (C).", 10), "C")
        # The official ".*[aA]nswer:" pattern has no DOTALL, so across lines it stops at the
        # first match; the port keeps that, which is why it is only a comparison reading.
        self.assertEqual(mmlu_pro.official_parse_without_random("Answer: A\nAnswer: D", 10), "A")
        self.assertEqual(mmlu_pro.official_parse_without_random("x Answer: A y Answer: D", 10), "D")
        self.assertEqual(mmlu_pro.official_parse_without_random("I think B is right", 10), "B")

    def test_no_random_guess_and_option_range(self) -> None:
        self.assertIsNone(mmlu_pro.official_parse_without_random("no idea", 10))
        self.assertIsNone(mmlu_pro.official_parse_without_random(None, 10))
        self.assertIsNone(mmlu_pro.official_parse_without_random("the answer is (H)", 4))


class ManifestRowTests(unittest.TestCase):
    def test_origin_and_row(self) -> None:
        self.assertEqual(mmlu_pro.source_origin("ori_mmlu-high_school_physics"), "ori_mmlu")
        self.assertEqual(mmlu_pro.source_origin("theoremQA-Math"), "theoremQA")
        row = {"question_id": 70, "category": "law", "src": "stemez-Law", "options": ["a", "b", "c"]}
        self.assertEqual(mmlu_pro.manifest_row(row),
                         {"problem_id": "70", "category": "law", "origin": "stemez", "n_options": 3})

    def test_question_key_ignores_surrounding_space(self) -> None:
        self.assertEqual(mmlu_pro.question_key(" Q ", ["a ", "b"]), mmlu_pro.question_key("Q", ["a", "b"]))


if __name__ == "__main__":
    unittest.main()
