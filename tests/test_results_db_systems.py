"""Decision ingestion checks use synthetic grades, never registered server data."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test_results_db import FakeArchives, LABELS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('db_system_tests', ROOT / 'scripts/build_results_db.py')
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)


class ResultsDatabaseSystems(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.artifacts = self.base / 'artifacts'
        fa = FakeArchives(self.artifacts)
        g = fa.make('c001', 'ds-math', 'test', 'Small', {'a': 1, 'b': 0, 'c': 0})
        v = fa.make('c002', 'ds-math', 'test', 'Big', {'c': 1, 'a': 0, 'b': 1})
        fa.make('c003', 'ds-math', 'dev', 'Small', {'d': 0})
        fa.make('c004', 'ds-math', 'dev', 'Big', {'d': 1})
        self.run = self.base / 'systems/pair'
        (self.run / 'records').mkdir(parents=True)
        (self.run / 'reports').mkdir()
        self.report = dict(domain='math', g='org/Small', v='org/Big', n=3,
                           runs=['unused-development-g', 'unused-development-v', g.name, v.name],
                           G=1, V=2, oracle=3, selectors={'primary': {'correct': 3}, 'secondary': {'correct': 2}})
        # Deliberately different row order from both grade files; joining by position gives a wrong score.
        self.primary = [dict(problem_id=p, p_g=pg, p_v=pv, pick_v=pv >= pg)
                        for p, pg, pv in [('b', .1, .8), ('c', .1, .8), ('a', .8, .1)]]
        self.secondary = [dict(problem_id=p, p_g=.5, p_v=.5, pick_v=True) for p in ('a', 'b', 'c')]
        self.group = dict(kind='selector', domain='math', evaluation='test', n=3, pairs=1,
                          root=str(self.run.parent), report_glob='*/reports/selector.json')
        self.save()

    def save(self):
        (self.run / 'reports/selector.json').write_text(json.dumps(self.report))
        (self.run / 'reports/summary.md').write_text('# Synthetic summary\n')
        for name, records in [('primary', self.primary), ('secondary', self.secondary)]:
            (self.run / f'records/{name}_decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        self.group['file_sha256'] = {str(p.relative_to(self.run.parent)).replace('\\', '/'):
                                    hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.rglob('*') if p.is_file()}

    def ingest(self):
        return db.ingest_systems(db.Archives(self.artifacts), LABELS, {'groups': [self.group]})

    def test_join_u0_by_id_and_label_design_and_evaluation(self):
        rows, summaries = self.ingest()
        self.assertEqual([s['correct'] for s in summaries], [3, 2])
        self.assertTrue(all(r['evaluation'] == 'test' and r['split'] == 'test' and r['in_design_sample'] for r in rows))
        self.assertTrue(all(r['dataset_id'] == 'ds-math' for r in rows))
        result = db.build(self.artifacts, LABELS, {'groups': [self.group]})
        self.assertEqual(len(result['pairs']['math']), 1)  # includes a system-only pair without review
        self.assertEqual(result['pairs']['math'][0]['stronger'], 'Big')
        out = self.base / 'output'
        db.write(result, out)
        self.assertTrue((out / 'db/systems.jsonl').exists())
        self.assertNotIn('problem_id', (out / 'reports/systems.json').read_text())

    def test_report_mismatch_stops(self):
        self.report['selectors']['primary']['correct'] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'differ from archive'):
            self.ingest()

    def test_duplicate_and_missing_ids_stop(self):
        self.primary[-1]['problem_id'] = 'b'
        self.save()
        with self.assertRaisesRegex(ValueError, 'missing/duplicate/extra'):
            self.ingest()

    def test_registered_checksum_stops_changes(self):
        with (self.run / 'records/primary_decisions.jsonl').open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.ingest()

    def test_model_role_and_upstream_split_checked(self):
        self.report['g'], self.report['v'] = self.report['v'], self.report['g']
        self.save()
        with self.assertRaisesRegex(ValueError, 'upstream model/dataset/split'):
            self.ingest()

    def test_tie_choice_is_not_silently_repaired(self):
        self.secondary[0]['pick_v'] = False
        self.save()
        with self.assertRaisesRegex(ValueError, 'choice differs'):
            self.ingest()

    def test_per_seed_summary_checked(self):
        fa = FakeArchives(self.artifacts)
        fa.n = 100
        g = fa.make('c101', 'ds-code', 'dev', 'Small', {'a': 1, 'b': 0, 'c': 0})
        v = fa.make('c102', 'ds-code', 'dev', 'Big', {'c': 1, 'a': 0, 'b': 1})
        self.report['runs'] = [g.name, v.name]
        self.report.update(domain='code', split='dev', seeds=[10, 11])
        self.group.update(domain='code', evaluation='cv')
        for records in (self.primary, self.secondary):
            for r in records:
                r['per_seed'] = [dict(seed=s, fold=0, p_g=r['p_g'], p_v=r['p_v']) for s in (10, 11)]
        self.report['selectors']['primary']['per_seed'] = [3, 3]
        self.report['selectors']['secondary']['per_seed'] = [2, 2]
        self.save()
        _, summaries = self.ingest()
        self.assertEqual(summaries[0]['per_seed'], [3, 3])
        self.report['selectors']['primary']['per_seed'][0] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'accuracy/per-seed counts differ'):
            self.ingest()

    def test_evaluation_labels(self):
        self.assertEqual(db.evaluation_set('math', 'dev'), 'dev')
        self.assertEqual(db.evaluation_set('knowledge', 'test'), 'test')
        self.assertEqual(db.evaluation_set('nl_reasoning', 'dev'), 'cv')


if __name__ == '__main__':
    unittest.main()
