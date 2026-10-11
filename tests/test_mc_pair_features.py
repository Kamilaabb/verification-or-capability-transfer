"""mc_pair_scalar_v1 (authority_allocation.mc_pair_features): option distribution, trajectory,
computation, problem indicators and cross features."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import mc_pair_features as mpf  # noqa: E402
from authority_allocation.text_mc_features import FEATURES as EVIDENCE  # noqa: E402

# Byte-level BPE spellings: "A" and "ĠA" (leading space) both spell option A.
VOCAB = {10: "A", 11: "ĠA", 12: "ĠB", 13: "C", 14: "ĠC", 15: "the", 16: "ĠK", 17: "AB"}
LETTERS = mpf.letter_map(VOCAB)


def position(logits: dict[int, float]) -> dict:
    lse = math.log(sum(math.exp(v) for v in logits.values()) + math.exp(0.0) * 5)
    ids = list(logits)
    return {"index": 7, "logsumexp": lse, "top20_token_ids": ids, "top20_raw_logits": [logits[i] for i in ids]}


def record(text: str, letter: str | None, pos: dict | None, n_options: int = 4) -> tuple[dict, dict, dict]:
    rec = {"outputs": [{"raw_output": text, "extracted_answer": letter}],
           "problem_metadata": {"n_options": n_options, "category": "law", "origin": "stemez"}}
    native = {"positions": [pos] if pos else []}
    evidence = {"features": {n: 1.0 for n in EVIDENCE}, "audit": {"answer_token_index": 7 if pos else None}}
    return rec, native, evidence


class LetterTests(unittest.TestCase):
    def test_letter_map_accepts_spaced_spellings_only(self) -> None:
        self.assertEqual(LETTERS, {10: "A", 11: "A", 12: "B", 13: "C", 14: "C"})
        self.assertIsNone(mpf.letter_of_token("ĠE", 4))     # beyond this question's options


class ModelFeatureTests(unittest.TestCase):
    def test_option_distribution_sums_spellings(self) -> None:
        pos = position({14: 5.0, 13: 3.0, 12: 4.0, 15: 1.0})
        f, extra = mpf.model_features(*record("So the answer is (C).\nAnswer: C", "C", pos), LETTERS)
        self.assertEqual(tuple(f), mpf.MODEL_FEATURES)
        self.assertEqual(len(mpf.MODEL_FEATURES), 38)
        dist = extra["dist"]
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        pc = (math.exp(5) + math.exp(3)) / (math.exp(5) + math.exp(3) + math.exp(4))
        self.assertAlmostEqual(f["opt_p_chosen"], pc)
        self.assertEqual((f["opt_chosen_is_top"], f["opt_n_plausible"]), (1.0, 2.0))
        self.assertAlmostEqual(f["opt_logit_gap"], 1.0)             # best C (5) minus best other (B, 4)
        self.assertLess(f["opt_mass"], 1.0)

    def test_trajectory_and_computation(self) -> None:
        text = "Option (A) seems likely, but wait, 2 + 3 = 5 so (B) is incorrect.\nAnswer: A"
        f, _ = mpf.model_features(*record(text, "A", None), LETTERS)
        self.assertEqual((f["n_letters_mentioned"], f["n_answer_lines"], f["first_letter_is_final"]),
                         (2.0, 1.0, 1.0))
        self.assertEqual((f["n_hedges"], f["n_eliminations"], f["n_selfcorrect"]), (2.0, 1.0, 1.0))
        self.assertEqual((f["n_numbers"], f["n_equals"]), (3.0, 1.0))
        self.assertIsNone(f["opt_p_chosen"])                        # no registered answer token

    def test_unparseable_answer(self) -> None:
        f, extra = mpf.model_features(*record("Not sure.", None, position({10: 2.0})), LETTERS)
        self.assertIsNone(f["opt_p_chosen"])
        self.assertIsNone(f["first_letter_is_final"])
        self.assertIsNone(extra["letter"])

    def test_problem_indicators(self) -> None:
        p = mpf.problem_features(record("", "A", None)[0])
        self.assertEqual(tuple(p), mpf.PROBLEM_FEATURES)
        self.assertEqual((p["category_law"], p["origin_stemez"], sum(p.values())), (1.0, 1.0, 2.0))


class CrossFeatureTests(unittest.TestCase):
    def test_agreement_ranks_and_divergence(self) -> None:
        fg, eg = mpf.model_features(*record("Answer: C", "C", position({14: 5.0, 12: 4.0})), LETTERS)
        fv, ev = mpf.model_features(*record("Answer: B", "B", position({12: 6.0, 14: 2.0})), LETTERS)
        c = mpf.cross_features(fg, fv, eg, ev)
        self.assertEqual(tuple(c), mpf.CROSS_FEATURES)
        self.assertEqual((c["same_letter"], c["both_parseable"], c["g_letter_rank_in_v"], c["v_letter_rank_in_g"]),
                         (0.0, 1.0, 2.0, 2.0))
        self.assertGreater(c["option_js"], 0.0)
        self.assertLess(c["v_prob_of_g_letter"], 0.5)
        same = mpf.cross_features(fg, fg, eg, eg)
        self.assertEqual((same["same_letter"], same["option_js"]), (1.0, 0.0))

    def test_one_side_unparseable(self) -> None:
        fg, eg = mpf.model_features(*record("Answer: C", "C", position({14: 5.0})), LETTERS)
        fv, ev = mpf.model_features(*record("?", None, None), LETTERS)
        c = mpf.cross_features(fg, fv, eg, ev)
        self.assertIsNone(c["same_letter"])
        self.assertEqual((c["both_parseable"], c["xor_parseable"]), (0.0, 1.0))
        self.assertIsNone(c["option_js"])


if __name__ == "__main__":
    unittest.main()
