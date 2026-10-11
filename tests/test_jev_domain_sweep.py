import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('jev_domain_sweep',ROOT/'scripts/jev_domain_sweep.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class JudgeDecisionTests(unittest.TestCase):
    def test_orders_map_to_same_model_and_tie(self):
        rows=[dict(order='GV',response={'answers':{'correct':{'probabilities':{'A':.2,'B':.8}}}}),
              dict(order='VG',response={'answers':{'correct':{'probabilities':{'A':.2,'B':.8}}}})]
        self.assertEqual(mod.pick_v(rows),(True,.5))
        rows[1]['response']['answers']['correct']['probabilities']['A']=.1
        self.assertFalse(mod.pick_v(rows)[0])

    def test_one_failed_order_falls_back(self):
        self.assertEqual(mod.pick_v([dict(order='GV',error='network')]),(True,None))

    def test_both_unparseable_mc_answers_are_sent(self):
        self.assertFalse(mod.equivalent('knowledge',{'extracted_answer':None},{'extracted_answer':None}))

    def test_permanent_billing_error_stops_instead_of_becoming_judge_output(self):
        with patch.object(mod.urllib.request,'urlopen',side_effect=HTTPError('url',402,'payment required',{},None)):
            with self.assertRaises(mod.ServiceBlocked): mod.request('dummy-not-a-real-key','systemone',{})

    def test_request_contains_only_allowed_mc_content(self):
        st=mod.state('knowledge',{'question':'Choose','options':['one','two']},'answer A','answer B')
        self.assertIn('Response A:\nanswer A',st)
        self.assertIn('A. one',st)

    def test_resume_uses_only_test_requests_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            test=root/'math/pair/test/records/jev_responses.jsonl'
            dev=root/'math/pair/dev/records/jev_responses.jsonl'
            test.parent.mkdir(parents=True); dev.parent.mkdir(parents=True)
            raw=(json.dumps(dict(split='test',state_sha256='test-state',response={'mock':1}))+'\n').encode()
            test.write_bytes(raw+b'{"interrupted":')
            dev.write_text(json.dumps(dict(split='dev',state_sha256='dev-state',response={'mock':2}))+'\n')
            original=test.read_bytes()
            cache,hashes=mod.resume_test_cache(root,'math')
            self.assertEqual(set(cache),{'test-state'})
            self.assertEqual(set(hashes),{str(test)})
            self.assertEqual(test.read_bytes(),original)

    def test_resume_rejects_corruption_before_final_partial_write(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); path=root/'knowledge/pair/test/records/jev_responses.jsonl'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'{broken\n'+json.dumps(dict(split='test',state_sha256='state',response={})).encode()+b'\n')
            with self.assertRaises(json.JSONDecodeError): mod.resume_test_cache(root,'knowledge')
