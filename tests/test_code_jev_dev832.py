import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('code_jev_dev832',ROOT/'scripts/code_jev_dev832.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def response(a,b):return {'model':'judge-version','answers':{'correct':{'probabilities':{'A':a,'B':b}}}}

class CodeJudgeTests(unittest.TestCase):
    def test_sample_zero_is_not_first_available(self):
        row={'outputs':[{'sample_index':1,'raw_output':'other'},{'sample_index':0,'raw_output':None}]}
        self.assertEqual(mod.sample_zero(row),'')
        with self.assertRaises(ValueError):mod.sample_zero({'outputs':[row['outputs'][0]]})

    def test_orders_follow_identity_and_failures_fall_back(self):
        self.assertEqual(mod.choose({'GV':{'response':response(.2,.8)},'VG':{'response':response(.2,.8)}}),(True,.5))
        self.assertFalse(mod.choose({'GV':{'response':response(.8,.2)},'VG':{'response':response(.1,.9)}})[0])
        self.assertEqual(mod.choose({'GV':{'response':response(.2,.8)},'VG':{'error':'timeout'}}),(True,None))
        with self.assertRaises(ValueError):mod.probability(response(float('nan'),.5),'GV')
        with self.assertRaises(ValueError):mod.probability(response(.9,.9),'GV')

    def test_hash_includes_rules_and_request_only_contains_allowed_content(self):
        payload=mod.body('task','first','second')
        self.assertEqual(set(payload),{'model','state','questions'})
        self.assertEqual(payload['state'],'Programming task:\ntask\n\nResponse A:\nfirst\n\nResponse B:\nsecond')
        original=mod.digest(payload)
        payload['questions']['correct']['instructions']='another rule'
        self.assertNotEqual(mod.digest(payload),original)

    def test_cache_preserves_bytes_and_rejects_earlier_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.jsonl'
            raw=(json.dumps(dict(request_sha256='sha',order='GV',response=response(.2,.8)))+'\n').encode()
            path.write_bytes(raw+b'{"partial":')
            self.assertEqual(mod.cache(path)['sha'][0],response(.2,.8))
            self.assertEqual(path.read_bytes(),raw+b'{"partial":')
            path.write_bytes(b'{broken\n'+raw)
            with self.assertRaises(ValueError):mod.cache(path)

    def test_smoke_decisions_precede_label_reads(self):
        import jev_domain_sweep as api
        import yaml
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'pairs').mkdir()
            for name in ['bigcodebench_jev_dev832.yaml','bigcodebench_screen_dev832.yaml']:
                (root/'pairs'/name).write_bytes((ROOT/'pairs'/name).read_bytes())
            (root/'SOURCE_COMMIT.txt').write_text('synthetic-commit')
            out=root/'run';key=root/'key';key.write_text('synthetic-test-key')
            pair=yaml.safe_load((root/'pairs/bigcodebench_screen_dev832.yaml').read_text())['pairs'][0]
            gc,vc=pair['g']['condition'],pair['v']['condition'];ids=[str(i) for i in range(832)]
            solos={c:{pid:{'problem':'task text','raw_output':f'code {c}'} for pid in ids} for c in [f'c{i}' for i in range(100,110)]}
            configs={c:{'model':{'model_id':c}} for c in (gc,vc)};runs={c:root/c for c in (gc,vc)}
            seen=[]
            class FakeArchives:
                sources={}
                def __init__(self,*args):pass
                def grades(self,run):
                    assert len((out/pair['id']/'records/decisions.jsonl').read_text().splitlines())==3
                    seen.append(run.name)
                    return {pid:int(run.name==vc) for pid in ids}
            def request(key,path,payload=None):
                if path=='models':return {'models':['synthetic']}
                self.assertEqual(set(payload),{'model','state','questions'})
                return response(.2,.8)
            exists=Path.exists
            def server_exists(path):
                return True if str(path).replace('\\','/')=='/path/to/project' else exists(path)
            with patch.object(mod,'ROOT',root),patch.object(mod,'Archives',FakeArchives), \
                 patch.object(mod,'load_inputs',return_value=(solos,configs,runs,ids)), \
                 patch.object(Path,'exists',server_exists),patch.object(api.api,'KEY_FILE',key), \
                 patch.object(api,'request',side_effect=request), \
                 patch.object(sys,'argv',['code_jev','--out',str(out),'--smoke','3']):mod.main()
            self.assertEqual(seen,[gc,vc])
            self.assertTrue((out/'checksums.sha256').exists())
            self.assertEqual(json.loads((out/'reports/jev_all.json').read_text())['pairs'][0]['JEV'],3)

if __name__=='__main__':unittest.main()
