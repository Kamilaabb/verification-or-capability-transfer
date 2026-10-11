import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import nl_jev_version_probe as probe


class NLJEVVersionTests(unittest.TestCase):
    def test_version_mismatch_preserves_response_and_stops_after_one_call(self):
        calls = []
        response = {'model': 'jev-1.12.0', 'usage': {'input_tokens': 17}}
        def request(*args):
            calls.append(args)
            return response
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'responses.jsonl'
            with self.assertRaises(probe.VersionMismatch):
                probe.probe_call('dummy', {}, {'order': 'GV'}, path, request)
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(path.read_text())['response'], response)

    def test_absent_version_rejected(self):
        with self.assertRaises(probe.VersionMismatch):
            probe.check_version({})

    def test_accepted_version_keeps_usage(self):
        response = {'model': probe.EXPECTED_MODEL, 'usage': {'input_tokens': 17},
                    'answers': {'correct': {'probabilities': {'A': .25, 'B': .75}}}}
        with tempfile.TemporaryDirectory() as tmp:
            result = probe.probe_call('dummy', {}, {'order': 'GV'}, Path(tmp) / 'r.jsonl',
                                     lambda *args: response)
            self.assertEqual(result['response']['usage']['input_tokens'], 17)

    def test_source_projection_excludes_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'q.jsonl'
            path.write_text(json.dumps({'problem_id': 'synthetic', 'question': 'q', 'options': ['x', 'y'],
                                        'answer': 'A', 'label': 0}) + '\n')
            projected = probe.source_questions(path, ['synthetic'])['synthetic']
            self.assertEqual(set(projected), {'question', 'options'})
            self.assertEqual(probe.body(**projected, g='g', v='v')['questions']['correct']['instructions'],
                             probe.mc.INSTRUCTIONS)


if __name__ == '__main__':
    unittest.main()
