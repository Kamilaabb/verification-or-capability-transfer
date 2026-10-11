import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.math_expression_spans import (  # noqa: E402
    content_token_indices,
    overlapping_token_indices,
    parse_math_spans,
)


class MathExpressionSpanTests(unittest.TestCase):
    def test_collects_all_explicit_math_islands(self):
        text = "Use $x+1=2$, then $$y=3$$ and \\(z=4\\)."
        spans = parse_math_spans(text)
        payloads = [text[span.start:span.end] for span in spans]
        self.assertEqual(payloads, ["x+1=2", "y=3", "z=4"])

    def test_excludes_markdown_code_and_adds_plain_equation(self):
        text = "`$not_math$` but x = 2 and ```\n$also_not$\n```"
        spans = parse_math_spans(text)
        self.assertEqual([text[s.start:s.end] for s in spans], ["x = 2"])

    def test_balanced_standalone_command_is_math(self):
        text = "Therefore \\boxed{\\frac{1}{2}} is final."
        spans = parse_math_spans(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(text[spans[0].start:spans[0].end], "\\boxed{\\frac{1}{2}}")

    def test_maps_unicode_character_spans_to_utf8_tokens(self):
        text = "前 $x=2$ 后"
        raw = text.encode("utf-8")
        chunks = ["前 ".encode(), "$x".encode(), "=2".encode(), "$ 后".encode()]
        tokens = []
        cursor = 0
        for index, chunk in enumerate(chunks):
            tokens.append({"index": index, "start_byte": cursor,
                           "end_byte": cursor + len(chunk), "logprob": -0.1})
            cursor += len(chunk)
        self.assertEqual(cursor, len(raw))
        trace = {"status": "ok", "content_start_byte": 0,
                 "content_end_byte": len(raw), "tokens": tokens}
        self.assertEqual(overlapping_token_indices(trace, text, parse_math_spans(text)), [1, 2])
        self.assertEqual(content_token_indices(trace), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

