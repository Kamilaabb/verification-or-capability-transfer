from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import mmmu_pro  # noqa: E402


class OptionsAndImagesTest(unittest.TestCase):
    def test_options_parse_from_the_source_literal(self) -> None:
        self.assertEqual(mmmu_pro.parse_options("['x', \"y's\"]"), ["x", "y's"])
        self.assertEqual(mmmu_pro.parse_options(["x", "y"]), ["x", "y"])
        with self.assertRaises(ValueError):
            mmmu_pro.parse_options("[1, 2]")
        with self.assertRaises((ValueError, SyntaxError)):
            mmmu_pro.parse_options("not a list")

    def test_options_render_one_letter_per_line(self) -> None:
        self.assertEqual(mmmu_pro.render_options(["p", "q", "r"]), "A. p\nB. q\nC. r")
        with self.assertRaises(ValueError):
            mmmu_pro.render_options(["o"] * 11)

    def test_images_in_order_of_first_reference(self) -> None:
        text = "See <image 2>, then <image 1>, and again <image 2>; also <image  3>."
        self.assertEqual(mmmu_pro.referenced_images(text), [2, 1, 3])
        self.assertEqual(mmmu_pro.referenced_images("no images"), [])

    def test_answer_must_be_an_existing_option(self) -> None:
        self.assertTrue(mmmu_pro.answer_is_option("J", 10))
        self.assertFalse(mmmu_pro.answer_is_option("E", 4))
        self.assertFalse(mmmu_pro.answer_is_option("AB", 10))


class VisualTokenTest(unittest.TestCase):
    def test_exact_multiple_is_unchanged(self) -> None:
        self.assertEqual(mmmu_pro.smart_resize(448, 640, min_pixels=65536, max_pixels=16777216),
                         (448, 640))
        self.assertEqual(mmmu_pro.visual_tokens(448, 640), 14 * 20)

    def test_small_image_is_raised_to_the_minimum(self) -> None:
        self.assertEqual(mmmu_pro.smart_resize(100, 100, min_pixels=65536, max_pixels=16777216),
                         (256, 256))
        self.assertEqual(mmmu_pro.visual_tokens(100, 100), 64)

    def test_large_image_is_capped(self) -> None:
        cap = 1280
        max_pixels = cap * mmmu_pro.QWEN3_VL_FACTOR ** 2
        self.assertEqual(mmmu_pro.smart_resize(4000, 3000, min_pixels=65536, max_pixels=max_pixels),
                         (1312, 960))
        self.assertEqual(mmmu_pro.visual_tokens(4000, 3000, max_pixels=max_pixels), 1230)
        self.assertLessEqual(mmmu_pro.visual_tokens(4000, 3000, max_pixels=max_pixels), cap)

    def test_extreme_aspect_ratio_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            mmmu_pro.visual_tokens(10, 2010)

    def test_agrees_with_transformers_when_available(self) -> None:
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        except Exception:
            self.skipTest("transformers is not installed locally")
        for height, width in ((100, 100), (448, 640), (4000, 3000), (37, 5000), (1023, 769)):
            for max_pixels in (786432, 1310720, 16777216):
                self.assertEqual(
                    tuple(mmmu_pro.smart_resize(height, width, min_pixels=65536, max_pixels=max_pixels)),
                    tuple(smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=max_pixels)))


class AllocationTest(unittest.TestCase):
    def test_largest_remainder_sums_exactly_and_is_proportional(self) -> None:
        sizes = {("Art", "test"): 7, ("Art", "validation"): 3, ("Math", "test"): 10}
        allocation = mmmu_pro.allocate_largest_remainder(sizes, 10)
        self.assertEqual(sum(allocation.values()), 10)
        self.assertEqual(allocation, {("Art", "test"): 4, ("Art", "validation"): 1, ("Math", "test"): 5})

    def test_ties_break_by_sorted_key(self) -> None:
        allocation = mmmu_pro.allocate_largest_remainder({"b": 1, "a": 1, "c": 1}, 1)
        self.assertEqual(allocation, {"a": 1, "b": 0, "c": 0})

    def test_total_outside_population_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            mmmu_pro.allocate_largest_remainder({"a": 2}, 3)


class AnswerParserTest(unittest.TestCase):
    def test_accepted_forms(self) -> None:
        for text, expected in (
            ("Reasoning.\nAnswer: B", "B"),
            ("**Answer:** (C)", "C"),
            ("**Answer: D**", "D"),
            ("**Answer**: D", "D"),
            ("Answer: $E$", "E"),
            ("Answer: [F]", "F"),
            ("Answer: $\\boxed{G}$", "G"),
            ("Answer: \\boxed{H}", "H"),
            ("### Final Answer: I", "I"),
            ("- Answer: J.", "J"),
            ("> Answer: A. The first option", "A"),
            ("  Answer:B", "B"),
        ):
            self.assertEqual(mmmu_pro.parse_answer_letter(text), expected, text)

    def test_only_the_last_answer_line_counts(self) -> None:
        self.assertEqual(mmmu_pro.parse_answer_letter("Answer: A\nWait.\nAnswer: H"), "H")
        # A last answer line with no letter does not fall back to an earlier one.
        self.assertIsNone(mmmu_pro.parse_answer_letter("Answer: A\nAnswer: see above"))

    def test_mid_sentence_answer_is_not_an_answer_line(self) -> None:
        text = "The proposed Answer: B is wrong because the curve decreases."
        self.assertIsNone(mmmu_pro.parse_answer_letter(text))
        self.assertEqual(mmmu_pro.parse_answer_letter(text + "\nAnswer: C"), "C")

    def test_unparseable_forms(self) -> None:
        for text in (None, "", "The answer is B", "answer: B", "Answer: As shown above",
                     "Answer: K", "Answer:\nB", "Answer: The best choice is B"):
            self.assertIsNone(mmmu_pro.parse_answer_letter(text), text)

    def test_letter_outside_this_question_is_unparseable(self) -> None:
        self.assertIsNone(mmmu_pro.parse_answer_letter("Answer: E", n_options=4))
        self.assertEqual(mmmu_pro.parse_answer_letter("Answer: D", n_options=4), "D")


class OfficialParserTest(unittest.TestCase):
    OPTIONS = ["red", "green", "blue", "yellow"]

    def test_answer_marker_with_one_letter(self) -> None:
        self.assertEqual(mmmu_pro.official_parse_without_random("so\nAnswer: C", self.OPTIONS), "C")

    def test_bracketed_and_spaced_letters(self) -> None:
        self.assertEqual(mmmu_pro.official_parse_without_random("I pick (B) here", self.OPTIONS), "B")
        self.assertEqual(mmmu_pro.official_parse_without_random("(A) or maybe (D)", self.OPTIONS), "D")

    def test_option_text_match(self) -> None:
        text = "the color of the sky in the image is blue today"
        self.assertEqual(mmmu_pro.official_parse_without_random(text, self.OPTIONS), "C")

    def test_where_official_code_guesses_this_returns_none(self) -> None:
        self.assertIsNone(mmmu_pro.official_parse_without_random("no idea", self.OPTIONS))
        self.assertIsNone(mmmu_pro.official_parse_without_random(None, self.OPTIONS))


class UserContentPartsTest(unittest.TestCase):
    SOLVE = "Question:\n{{question}}\n\nOptions:\n{{options}}"
    REVIEW = SOLVE + "\n\nProposed solution:\n{{y0}}"

    def test_images_follow_their_first_tag_and_repeats_stay_text(self) -> None:
        parts = mmmu_pro.user_content_parts(
            self.SOLVE, question="Compare <image 1> with <image 2>. Is <image 1> larger?",
            options=["yes", "<image 3>"])
        self.assertEqual(parts, [
            {"type": "text", "text": "Question:\nCompare <image 1>"},
            {"type": "image", "number": 1},
            {"type": "text", "text": " with <image 2>"},
            {"type": "image", "number": 2},
            {"type": "text", "text": ". Is <image 1> larger?\n\nOptions:\nA. yes\nB. <image 3>"},
            {"type": "image", "number": 3},
        ])

    def test_proposed_solution_never_carries_images(self) -> None:
        parts = mmmu_pro.user_content_parts(
            self.REVIEW, question="What is <image 1>?", options=["a", "b"],
            y0="In <image 1> and <image 2> I see a.\nAnswer: A")
        self.assertEqual([p["number"] for p in parts if p["type"] == "image"], [1])
        self.assertEqual(parts[-1], {"type": "text", "text": (
            "?\n\nOptions:\nA. a\nB. b\n\nProposed solution:\n"
            "In <image 1> and <image 2> I see a.\nAnswer: A")})

    def test_template_variables_must_match_the_call(self) -> None:
        with self.assertRaises(ValueError):
            mmmu_pro.user_content_parts(self.SOLVE, question="q <image 1>", options=["a"], y0="x")
        with self.assertRaises(ValueError):
            mmmu_pro.user_content_parts(self.REVIEW, question="q <image 1>", options=["a"])


if __name__ == "__main__":
    unittest.main()
