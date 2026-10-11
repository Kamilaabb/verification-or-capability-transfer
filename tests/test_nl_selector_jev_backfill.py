import io
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import nl_selector_crossfit as selector
import nl_jev_eval901 as jev
from nl_jev_version_probe import VersionMismatch


class NLBackfillTests(unittest.TestCase):
    def test_registration_matches_labels_and_frozen_knowledge_models(self):
        reg,ds=selector.registration()
        self.assertEqual(len(reg['pairs']),12)
        self.assertEqual({p['selector_condition'] for p in reg['pairs']},{f'c{i}' for i in range(374,386)})
        self.assertEqual(ds['manifests']['eval']['rows'],901)

    def test_mismatch_stops_new_requests_and_preserves_usage(self):
        calls=[]
        def request(*args):
            calls.append(args)
            return {'model':'unexpected','usage':{'input_tokens':7,'output_tokens':2}}
        todo=[({'problem_id':f'synthetic{i}','order':'GV'},{}) for i in range(3)]
        stream=io.StringIO();current={}
        with self.assertRaises(VersionMismatch):
            jev.call_requests(todo,'dummy',stream,current,workers=1,request_fn=request)
        self.assertEqual(len(calls),1)
        self.assertEqual(json.loads(stream.getvalue())['response']['usage']['input_tokens'],7)

    def test_missing_token_accounting_stops(self):
        rec={'order':'VG','response':{'model':'jev-1.13.0',
             'answers':{'correct':{'probabilities':{'A':.4,'B':.6}}},'usage':{'input_tokens':4}}}
        with self.assertRaises(ValueError):jev.validate_response(rec)

    def test_invalid_probabilities_stop(self):
        rec={'order':'GV','response':{'model':'jev-1.13.0',
             'answers':{'correct':{'probabilities':{'A':.4,'B':.9}}},
             'usage':{'input_tokens':4,'output_tokens':2}}}
        with self.assertRaises(ValueError):jev.validate_response(rec)


if __name__=='__main__':unittest.main()
