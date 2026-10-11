"""code_pair_scalar_v2 (authority_allocation.code_pair_features): spans, structure, task
agreement, hygiene and cross features, all static."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import code_pair_features as cpf  # noqa: E402
from authority_allocation.code_solve_features import FEATURES as EVIDENCE  # noqa: E402

TASK = ("Add two numbers.\nThe function should raise the exception for:\n    ValueError: if a is negative.\n"
        "    TypeError: if b is not a number.\nThe function should output with:\n    int: the sum.\n"
        "You should write self-contained code starting with:\n```\nimport math\ndef task_func(a, b):\n```")
PROGRAM = ("import math\nimport numpy as np\ndef task_func(a, b):\n    if a < 0:\n        raise ValueError('neg')\n"
           "    for i in range(2):\n        pass\n    x = np.random.rand()\n    return math.floor(a + b + y)\n")
TEXT = "Here it is.\n```python\n" + PROGRAM + "```\nDone."


def record(text: str, entry: str = "task_func", libs: str = "['math', 'numpy', 'pandas']",
           task: str = TASK) -> tuple[dict, dict, dict]:
    pieces = [text[i:i + 4] for i in range(0, len(text), 4)]
    tokens, byte = [], 0
    for k, piece in enumerate(pieces):
        n = len(piece.encode("utf-8"))
        tokens.append({"index": k, "start_byte": byte, "end_byte": byte + n})
        byte += n
    trace = {"status": "ok", "content_start_byte": 0, "content_end_byte": byte, "tokens": tokens}
    rec = {"problem_text": task, "outputs": [{"raw_output": text, "token_logprobs": trace}],
           "problem_metadata": {"entry_point": entry, "libs": libs}}
    native = {"positions": [{"index": k, "sampled_logprob": -0.2 if k % 2 else -1.0, "entropy": 0.3,
                             "top1_top2_margin": 2.0, "max_raw_logit": 9.0} for k in range(len(pieces))]}
    evidence = {"features": {name: 1.0 for name in EVIDENCE}}
    return rec, native, evidence


class ModelFeatureTests(unittest.TestCase):
    def test_contract_order_and_sizes(self) -> None:
        f, _ = cpf.model_features(*record(TEXT))
        self.assertEqual(tuple(f), cpf.MODEL_FEATURES)
        self.assertEqual(len(cpf.MODEL_FEATURES), 65)
        self.assertEqual(len(cpf.CROSS_FEATURES), 25)

    def test_structure_task_and_hygiene(self) -> None:
        f, extra = cpf.model_features(*record(TEXT))
        self.assertEqual((f["ast_parses"], f["defines_entry_point"], f["n_functions"], f["n_raise"]),
                         (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(f["required_lib_share"], 2 / 3)
        self.assertEqual(f["raises_when_asked"], 1.0)
        self.assertEqual(f["asked_exception_share"], 0.5)          # ValueError raised, TypeError not
        self.assertEqual(f["undefined_names"], 1.0)                # y
        self.assertEqual(f["unseeded_random"], 1.0)
        self.assertEqual((f["n_prints"], f["has_main_guard"], f["n_file_io"]), (0.0, 0.0, 0.0))
        self.assertEqual(f["n_library_calls"], 2.0)                # np.random.rand, math.floor
        self.assertEqual(extra["raise_types"], {"ValueError"})
        self.assertIsNotNone(f["code_lp_mean"])
        self.assertIsNotNone(f["prose_lp_mean"])

    def test_seeded_random_main_guard_prints_and_star_imports(self) -> None:
        program = ("import random\nfrom random import randint\ndef task_func(a):\n    random.seed(0)\n"
                   "    print(a)\n    return randint(0, a)\nif __name__ == '__main__':\n    task_func(3)\n")
        f, _ = cpf.model_features(*record("```python\n" + program + "```"))
        self.assertEqual((f["unseeded_random"], f["n_prints"], f["has_main_guard"], f["undefined_names"]),
                         (0.0, 1.0, 1.0, 0.0))
        unseeded = "from random import randint\ndef task_func(a):\n    return randint(0, a)\n"
        self.assertEqual(cpf.model_features(*record("```python\n" + unseeded + "```"))[0]["unseeded_random"], 1.0)
        star = "from os.path import *\ndef task_func(a):\n    return join(a, b)\n"
        self.assertEqual(cpf.model_features(*record("```python\n" + star + "```"))[0]["undefined_names"], 0.0)

    def test_task_without_a_raise_clause(self) -> None:
        f, _ = cpf.model_features(*record(TEXT, task="Add two numbers."))
        self.assertIsNone(f["raises_when_asked"])
        self.assertIsNone(f["asked_exception_share"])
        self.assertEqual(cpf.problem_features(record(TEXT, task="Add.")[0]),
                         {"n_required_libs": 3.0, "task_asks_raise": 0.0, "n_asked_exceptions": 0.0})
        self.assertEqual(cpf.problem_features(record(TEXT)[0]),
                         {"n_required_libs": 3.0, "task_asks_raise": 1.0, "n_asked_exceptions": 2.0})

    def test_code_and_prose_tokens_partition_the_output(self) -> None:
        rec, _, _ = record(TEXT)
        code, prose = cpf.token_split(TEXT, rec["outputs"][0]["token_logprobs"])
        self.assertEqual(sorted(code + prose), list(range(len(rec["outputs"][0]["token_logprobs"]["tokens"]))))
        self.assertTrue(code and prose)

    def test_no_program_and_unparseable_program(self) -> None:
        f, extra = cpf.model_features(*record("I cannot solve this."))
        self.assertEqual((f["code_lines"], f["ast_parses"], f["defines_entry_point"]), (0.0, 0.0, 0.0))
        self.assertIsNone(f["code_lp_mean"])
        self.assertIsNone(f["undefined_names"])
        self.assertIsNone(extra["program"])
        f, _ = cpf.model_features(*record("```python\ndef task_func(:\n```"))
        self.assertEqual((f["ast_parses"], f["defines_entry_point"]), (0.0, 0.0))
        self.assertIsNone(f["n_functions"])

    def test_wrong_entry_point_and_unreadable_libs(self) -> None:
        f, _ = cpf.model_features(*record(TEXT, entry="other", libs="not a list"))
        self.assertEqual(f["defines_entry_point"], 0.0)
        self.assertIsNone(f["required_lib_share"])


class CrossFeatureTests(unittest.TestCase):
    def test_identical_and_different_programs(self) -> None:
        fg, eg = cpf.model_features(*record(TEXT))
        fv, ev = cpf.model_features(*record(TEXT.replace("a + b", "a - b")))
        same = cpf.cross_features(fg, fg, eg, eg)
        self.assertEqual(tuple(same), cpf.CROSS_FEATURES)
        self.assertEqual((same["program_identical"], same["ast_identical"], same["program_similarity"],
                          same["call_jaccard"], same["raise_type_jaccard"]), (1.0, 1.0, 1.0, 1.0, 1.0))
        diff = cpf.cross_features(fg, fv, eg, ev)
        self.assertEqual((diff["program_identical"], diff["ast_identical"], diff["xor_raises_when_asked"]),
                         (0.0, 0.0, 0.0))
        self.assertGreater(diff["program_similarity"], 0.9)

    def test_missing_program_gives_none_not_a_guess(self) -> None:
        fg, eg = cpf.model_features(*record(TEXT))
        fv, ev = cpf.model_features(*record("No code."))
        c = cpf.cross_features(fg, fv, eg, ev)
        self.assertIsNone(c["program_identical"])
        self.assertIsNone(c["program_similarity"])
        self.assertIsNone(c["xor_raises_when_asked"])
        self.assertEqual((c["xor_ast_parses"], c["xor_defines_entry_point"]), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
