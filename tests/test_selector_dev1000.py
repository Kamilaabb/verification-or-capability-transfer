from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location("selector_dev1000", ROOT / "scripts/selector_dev1000.py")
selector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(selector)


class SelectorHelperTests(unittest.TestCase):
    def test_last_boxed_span_matches_nested_braces(self) -> None:
        text = r"first \boxed{1} then \boxed{\frac{1}{2}} end"
        start, end = selector.boxed_char_span(text)
        self.assertEqual(text[start:end], r"\boxed{\frac{1}{2}}")
        self.assertIsNone(selector.boxed_char_span("no answer"))
        self.assertIsNone(selector.boxed_char_span(r"\boxed{unclosed"))

    def test_answer_tokens_follow_byte_offsets(self) -> None:
        raw = "é=\\boxed{2}"
        pieces = ["é", "=", "\\", "boxed", "{", "2", "}"]
        tokens, cursor = [], 0
        for index, piece in enumerate(pieces):
            size = len(piece.encode("utf-8"))
            tokens.append({"index": index, "start_byte": cursor, "end_byte": cursor + size})
            cursor += size
        trace = {"status": "ok", "content_start_byte": 0, "tokens": tokens}
        self.assertEqual(selector.answer_tokens(raw, trace), [2, 3, 4, 5, 6])

    def test_answer_kind(self) -> None:
        self.assertEqual(selector.answer_kind("12"), "integer")
        self.assertEqual(selector.answer_kind("-3/4"), "fraction")
        self.assertEqual(selector.answer_kind(r"\frac{1}{2}"), "fraction")
        self.assertEqual(selector.answer_kind("x^2+1"), "other")
        self.assertIsNone(selector.answer_kind(None))

    def test_low_confidence_run(self) -> None:
        f = selector._stats("o", [-0.1, -1.0, -2.0, -0.05], [0.1, 0.5, 0.9, 0.2], [3, 1, 0.2, 4])
        self.assertEqual(f["o_lowconf_run"], 2.0)
        self.assertAlmostEqual(f["o_lowconf_frac"], 0.5)
        self.assertTrue(math.isclose(f["o_lp_min"], -2.0))
        empty = selector._stats("o", [], [], [])
        self.assertIsNone(empty["o_lp_mean"])

    def test_ties_deliver_v(self) -> None:
        import numpy as np
        r = selector.deliver(np.array([0.5, 0.9]), np.array([0.5, 0.1]),
                             np.array([1, 1]), np.array([0, 0]))
        self.assertEqual(r["chose_g"], 1)
        self.assertEqual(r["correct"], 1)


if __name__ == "__main__":
    unittest.main()
