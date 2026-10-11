import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('examples', Path(__file__).resolve().parents[1] / 'scripts/select_example_candidates.py')
examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(examples)


class ExampleSelectionTests(unittest.TestCase):
    def test_shortest_eight_and_deterministic_tie(self):
        rows = [{'problem_id': str(i), 'total_chars': 10 + i // 2} for i in range(10)]
        result = examples.select_shortest(list(reversed(rows)))
        self.assertEqual([r['problem_id'] for r in result], [str(i) for i in range(8)])
        self.assertEqual(len(rows), 10)

    def test_insufficient_candidates_stop(self):
        with self.assertRaises(ValueError):
            examples.select_shortest([{'problem_id': 'synthetic', 'total_chars': 1}])

    def test_does_not_take_later_parseable_sample(self):
        row = {'y0_sample_index': 0, 'outputs': [{'sample_index': 1, 'raw_output': 'synthetic'}]}
        with self.assertRaises(ValueError):
            examples.output(row)

    def test_a_wrong_grade_without_an_answer_cannot_be_adoption(self):
        self.assertFalse(examples.has_answer({'extracted_answer': None, 'raw_output': 'unfinished'}))
        self.assertTrue(examples.has_answer({'extracted_answer': '8*pi', 'raw_output': r'\boxed{8\pi}'}))


if __name__ == '__main__':
    unittest.main()
