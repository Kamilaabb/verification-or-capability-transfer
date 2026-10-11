"""The MMMU-Pro study's prompts are frozen for dev and test alike
(docs/plan/vision_mmmu_pro_study.md section 5). Any edit to a frozen file fails here, which
stops every sbatch before a job starts."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import mmmu_pro  # noqa: E402
from authority_allocation.prompts import PROMPT_VARIABLE, split_prompt  # noqa: E402

LAST_LINE = ('The last line of your response must be of the form "Answer: X", where X is the '
             'letter of exactly one option.')


class FrozenVisionPromptTests(unittest.TestCase):
    def test_frozen_files_keep_their_registered_hash(self) -> None:
        for relative in mmmu_pro.FROZEN_PROMPTS:
            mmmu_pro.require_frozen_prompt(ROOT, relative)

    def test_solve_prompt_renders_to_the_registered_messages(self) -> None:
        system, user = split_prompt(ROOT / mmmu_pro.VL_SOLVE_PROMPT)
        self.assertEqual(system, (
            "Answer the multiple-choice question provided by the user, using the image(s) and "
            "text it contains. Think step by step before answering.\n" + LAST_LINE))
        self.assertEqual(set(PROMPT_VARIABLE.findall(user)), {"question", "options"})
        parts = mmmu_pro.user_content_parts(user, question="What does <image 1> show?",
                                            options=["a {{x}}", "b"])
        self.assertEqual(parts, [
            {"type": "text", "text": "Question:\nWhat does <image 1>"},
            {"type": "image", "number": 1},
            {"type": "text", "text": " show?\n\nOptions:\nA. a {{x}}\nB. b"},
        ])

    def test_review_prompt_renders_to_the_registered_messages(self) -> None:
        system, user = split_prompt(ROOT / mmmu_pro.VL_REVIEW_PROMPT)
        self.assertEqual(system, (
            "Review the proposed solution to the multiple-choice question for correctness.\n"
            "The proposed solution may be correct or incorrect. Carefully check its reading of "
            "the image(s), its reasoning, and its final choice.\n"
            "If the proposed solution is correct, retain its answer. If it contains any error, "
            "correct the error and produce a correct solution.\n"
            "Return a complete final solution to the original question. " + LAST_LINE))
        self.assertEqual(set(PROMPT_VARIABLE.findall(user)), {"question", "options", "y0"})
        parts = mmmu_pro.user_content_parts(user, question="Q <image 1>", options=["a", "b"],
                                            y0="G-SOLUTION\nAnswer: B")
        self.assertEqual(parts[-1], {"type": "text", "text": (
            "\n\nOptions:\nA. a\nB. b\n\nProposed solution:\nG-SOLUTION\nAnswer: B")})

    def test_frozen_prompts_carry_no_experiment_framing(self) -> None:
        for relative in mmmu_pro.FROZEN_PROMPTS:
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
            for word in ("generator", "verifier", "experiment", "authority", "another model"):
                self.assertNotIn(word, text, f"{relative} mentions {word!r}")


if __name__ == "__main__":
    unittest.main()
