"""BigCodeBench study helpers and its frozen solve prompt (docs/plan/code_bigcodebench_study.md).
Any edit to the frozen prompt file fails here, which stops every sbatch before a job starts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import bigcodebench_study  # noqa: E402
from authority_allocation.code_solve_features import FEATURES, derive_features  # noqa: E402
from authority_allocation.prompts import PROMPT_VARIABLE, split_prompt  # noqa: E402


class FrozenPromptTests(unittest.TestCase):
    def test_frozen_file_keeps_its_registered_hash(self) -> None:
        for relative in bigcodebench_study.FROZEN_PROMPTS:
            bigcodebench_study.require_frozen_prompt(ROOT, relative)

    def test_solve_prompt_renders_to_the_registered_messages(self) -> None:
        system, user = split_prompt(ROOT / bigcodebench_study.CODE_SOLVE_PROMPT)
        self.assertEqual(system, (
            "Complete the Python programming task provided by the user. Use sound reasoning to "
            "produce a correct and complete solution.\n"
            "End with exactly one complete, self-contained Python code block (```python ... ```) "
            "that includes all required imports and the requested function definition."))
        self.assertEqual(set(PROMPT_VARIABLE.findall(user)), {"problem"})
        task = "Write task_func.\n```\ndef task_func(d={{}}):\n```"
        self.assertEqual(bigcodebench_study.user_message(user, problem=task), "Task:\n" + task)

    def test_frozen_prompt_carries_no_experiment_framing(self) -> None:
        text = (ROOT / bigcodebench_study.CODE_SOLVE_PROMPT).read_text(encoding="utf-8").lower()
        for word in ("generator", "verifier", "experiment", "authority", "another model", "review"):
            self.assertNotIn(word, text)

    def test_template_variables_must_match_the_call(self) -> None:
        _, user = split_prompt(ROOT / bigcodebench_study.CODE_SOLVE_PROMPT)
        with self.assertRaises(ValueError):
            bigcodebench_study.user_message(user, problem="T", y0="extra")


class PythonBlockTests(unittest.TestCase):
    def test_counts_only_complete_python_blocks(self) -> None:
        count = bigcodebench_study.python_block_count
        self.assertEqual(count("```python\nx = 1\n```"), 1)
        self.assertEqual(count("```Python3\nx = 1\n```\ntext\n```py\ny = 2\n```"), 2)
        self.assertEqual(count("```\nx = 1\n```"), 0)
        self.assertEqual(count("```python\nx = 1\n"), 0)
        self.assertEqual(count(None), 0)


class FeatureTests(unittest.TestCase):
    def test_features_follow_the_contract_and_null_when_unpaired(self) -> None:
        trace = {"status": "ok", "content_start_byte": 0, "content_end_byte": 6,
                 "tokens": [{"index": 0, "start_byte": 0, "end_byte": 3},
                            {"index": 1, "start_byte": 3, "end_byte": 6}]}
        output = {"raw_output": "```python\nx\n```", "token_logprobs": trace,
                  "usage": {"input_tokens": 50, "output_tokens": 2, "finish_reason": "stop"}}
        position = {"sampled_logprob": -0.1, "entropy": 0.2, "top1_top2_margin": 1.0, "max_raw_logit": 9.0}
        native = {"positions": [{"index": 0, **position}, {"index": 1, **position}]}
        features, audit = derive_features(output=output, native=native, problem_text="abcd")
        self.assertEqual(tuple(features), FEATURES)
        self.assertEqual((features["problem_char_count"], features["python_block_count"],
                          features["truncated"]), (4, 1, 0))
        self.assertAlmostEqual(features["out_lp_mean"], -0.1)
        self.assertEqual(audit, {"content_token_count": 2, "paired_token_count": 2})
        features, _ = derive_features(output=output, native={"positions": native["positions"][:1]},
                                      problem_text="abcd")
        self.assertIsNone(features["out_lp_mean"])


if __name__ == "__main__":
    unittest.main()
