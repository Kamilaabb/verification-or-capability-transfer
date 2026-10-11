"""The mathfull study's prompts are frozen for dev and test alike (docs/plan/gv_vs_max_g_v_study.md
section 4.7). Any edit to a frozen file fails here, which stops every sbatch before a job starts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.prompts import PROMPT_VARIABLE, render, split_prompt  # noqa: E402


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


solo = load("run_mathfull_solo")
review = load("run_mathfull_review")


def model(condition_id: str) -> dict:
    return resolve_condition(ROOT / f"conditions/{condition_id}/condition.yaml", repository_root=ROOT)["model"]


class FrozenPromptTests(unittest.TestCase):
    def test_frozen_files_keep_their_registered_hash(self) -> None:
        for relative in solo.FROZEN_PROMPTS:
            solo.require_frozen_prompt(relative)

    def test_conditions_point_at_the_frozen_files(self) -> None:
        self.assertEqual(model("c025")["prompt_pass1"], solo.SOLVE_PROMPT_V2)
        self.assertEqual(model("c026")["prompt_pass1"], solo.SOLVE_PROMPT_V2)
        self.assertEqual(model("c029")["prompt_pass1"], solo.SOLVE_PROMPT_V2)
        self.assertEqual(model("c030")["prompt_pass1"], solo.SOLVE_PROMPT_V2)
        self.assertEqual(model("c031")["prompt_pass1"], solo.SOLVE_PROMPT_V2)
        self.assertEqual(model("c032")["prompt_pass1"], solo.SOLVE_PROMPT_V2)
        self.assertEqual(model("c027")["prompt_pass1"], solo.REVIEW_PROMPT_V2)
        self.assertEqual(model("c028")["prompt_pass1"], solo.REVIEW_PROMPT_V2)

    def test_solve_prompt_renders_to_the_registered_messages(self) -> None:
        system, user = split_prompt(ROOT / solo.SOLVE_PROMPT_V2)
        self.assertEqual(system, (
            "Solve the mathematical problem provided by the user. Use mathematically sound "
            "reasoning and calculations to produce a complete solution.\n"
            "End with exactly one final answer in LaTeX boxed form: \\boxed{...}."))
        self.assertEqual(set(PROMPT_VARIABLE.findall(user)), {"problem"})
        self.assertEqual(render(user, problem="$x^{2}$ and {{a}}"), "Problem:\n$x^{2}$ and {{a}}")

    def test_review_prompt_renders_to_the_registered_messages(self) -> None:
        review.configure("c027")
        try:
            msgs = review.messages(model("c027"), "PROBLEM", "G-SOLUTION \\boxed{4}")
        finally:
            review.configure("c024")
        self.assertEqual(msgs[0]["content"], (
            "Review the proposed mathematical solution for correctness.\n"
            "The proposed solution may be correct or incorrect. Carefully check the reasoning, "
            "calculations, assumptions, and final answer.\n"
            "If the proposed solution is correct, retain it. If it contains any mathematical "
            "error, correct the error and produce a correct solution.\n"
            "Return a complete final solution to the original problem. End with exactly one "
            "final answer in LaTeX boxed form: \\boxed{...}."))
        self.assertEqual(msgs[1]["content"],
                         "Problem:\nPROBLEM\n\nProposed solution:\nG-SOLUTION \\boxed{4}")

    def test_frozen_prompts_carry_no_experiment_framing(self) -> None:
        for relative in solo.FROZEN_PROMPTS:
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
            for word in ("generator", "verifier", "experiment", "authority", "another model"):
                self.assertNotIn(word, text, f"{relative} mentions {word!r}")


if __name__ == "__main__":
    unittest.main()
