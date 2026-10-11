#!/usr/bin/env python3
"""One real, knowledge-prompt JEV request; stop unless its version is 1.13.0.

Server-only. The successful response is a reusable upstream artifact for the
full NL run. No grades, gold fields, or identifiers enter the request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from gv_domain_tables import Archives
from code_jev_dev832 import digest, probability, read_rows, sample_zero
import mc_jev_judge_test1500 as mc
import jev_domain_sweep as api

EXPECTED_MODEL = 'jev-1.13.0'
REGISTRY = ROOT / 'pairs/nl_selector_jev_eval901.yaml'


class VersionMismatch(RuntimeError):
    pass


def check_version(response):
    actual = response.get('model')
    if actual != EXPECTED_MODEL:
        raise VersionMismatch(f'expected {EXPECTED_MODEL}; returned {actual!r}')


def body(question, options, g, v):
    return {'model': mc.MODEL, 'state': mc.state(question, options, g, v),
            'questions': {'correct': {'type': 'choice', 'instructions': mc.INSTRUCTIONS,
                                     'criteria': dict(mc.CRITERIA)}}}


def checked_file(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('registered source checksum mismatch')
    return path


def source_questions(path, ids):
    """Project onto question/options immediately; never retain a gold field."""
    out = {}
    wanted = set(ids)
    for row in read_rows(path):
        pid = str(row['problem_id'])
        if pid in wanted:
            if pid in out:
                raise ValueError('duplicate source row')
            out[pid] = {'question': row['question'], 'options': list(row['options'])}
    if set(out) != wanted:
        raise ValueError('source membership mismatch')
    return out


def inputs(artifacts):
    reg = yaml.safe_load(REGISTRY.read_text(encoding='utf-8'))
    labels = yaml.safe_load((ROOT / 'claims/labels.yaml').read_text(encoding='utf-8'))
    design = labels['domains']['nl_reasoning']['design_pairs']
    if [p['design'] for p in reg['pairs']] != design or len(design) != 12:
        raise ValueError('registry differs from the corrected 12 design pairs')
    ds = yaml.safe_load((ROOT / f"datasets/{reg['dataset']}/dataset.yaml").read_text(encoding='utf-8'))
    m = ds['manifests']['eval']
    manifest = checked_file(Path(m['server_path']), m['sha256'])
    ids = [str(r['problem_id']) for r in read_rows(manifest)]
    if len(ids) != 901 or len(set(ids)) != 901:
        raise ValueError('expected all 901 unique registered rows')
    src = ds['source_rows']
    path = checked_file(Path(src['server_path']), src['sha256'])
    questions = source_questions(path, ids)
    ar = Archives(artifacts)
    solos, runs = {}, {}
    for cid, frozen in reg['solos'].items():
        run = artifacts / frozen['run']
        cfg = ar.config(run)
        if cfg['condition_id'] != cid or cfg['model']['model_id'] != frozen['model']:
            raise ValueError('wrong upstream identity')
        if (cfg['dataset']['id'], cfg['dataset']['manifest_sha256']) != (reg['dataset'], m['sha256']):
            raise ValueError('wrong upstream dataset')
        for rel in ('condition.resolved.yaml', 'records/pass1.jsonl'):
            checked_file(ar.checked(run, rel), frozen['sha256'][rel])
        rows = {}
        for row in read_rows(ar.checked(run, 'records/pass1.jsonl')):
            pid = str(row['problem_id'])
            if pid in rows or row['condition_id'] != cid:
                raise ValueError('duplicate or wrong solo row')
            sample_zero(row)
            output = next(o for o in row['outputs'] if o['sample_index'] == 0)
            rows[pid] = {'text': sample_zero(row), 'letter': output.get('extracted_answer')}
            if row['problem_text'] != questions[pid]['question']:
                raise ValueError('archived question differs from registered source')
        if set(rows) != set(ids):
            raise ValueError('solo membership mismatch')
        solos[cid], runs[cid] = rows, run
    return reg, ds, ar, ids, questions, solos, runs


def probe_call(key, payload, record, response_path, request_fn=api.request):
    """Persist the actual response before checking its version and probabilities."""
    start = time.perf_counter()
    record = dict(record, called_at_utc=datetime.now(timezone.utc).isoformat())
    response = request_fn(key, 'systemone', payload)
    record.update(response=response, latency_ms=1000 * (time.perf_counter() - start))
    with response_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
    check_version(response)
    probability(response, record['order'])
    if not isinstance(response.get('usage'), dict):
        raise ValueError('missing per-call token usage')
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--artifacts', type=Path, default=Path('/path/to/project/artifacts'))
    args = ap.parse_args()
    if not Path('/path/to/project').exists():
        raise SystemExit('server only')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'records').mkdir()
    (args.out / 'reports').mkdir()
    report = {'status': 'failed', 'required_model': EXPECTED_MODEL, 'calls': 0,
              'source_commit': (ROOT / 'SOURCE_COMMIT.txt').read_text().strip(),
              'registry_sha256': hashlib.sha256(REGISTRY.read_bytes()).hexdigest()}
    try:
        reg, ds, ar, ids, questions, solos, runs = inputs(args.artifacts)
        report.update(pairs=12, rows=901, upstream_runs={c:r.name for c,r in runs.items()},
                      source_sha256=ar.sources, manifest_sha256=ds['manifests']['eval']['sha256'])
        for pair in reg['pairs']:
            gc, vc = pair['g']['condition'], pair['v']['condition']
            pid = next((pid for pid in ids if not (solos[gc][pid]['letter'] is not None and
                        solos[gc][pid]['letter'] == solos[vc][pid]['letter'])), None)
            if pid is not None:
                break
        else:
            raise ValueError('no disagreement available for a real version probe')
        q = questions[pid]
        payload = body(q['question'], q['options'], solos[gc][pid]['text'], solos[vc][pid]['text'])
        record = {'pair': pair['id'], 'split': 'development', 'problem_id': pid, 'order': 'GV',
                  'request_sha256': digest(payload)}
        response_path = args.out / 'records/jev_responses.jsonl'
        result = probe_call(mc.KEY_FILE.read_text().strip(), payload, record, response_path)
        report.update(status='passed', returned_model=result['response']['model'], calls=1,
                      usage=result['response']['usage'], pair=pair['id'])
    except Exception as exc:
        report.update(error_type=type(exc).__name__)
        path = args.out / 'records/jev_responses.jsonl'
        if path.exists():
            result = next(read_rows(path))
            report.update(calls=1, returned_model=result['response'].get('model'),
                          usage=result['response'].get('usage'))
        # Do not expose request text, row identifiers, credentials or network error bodies.
    (args.out / 'reports/version_probe.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    text = (f"# NL JEV version gate\n\nStatus: {report['status']}. Required: {EXPECTED_MODEL}; "
            f"returned: {report.get('returned_model', 'no response')}. Real calls: {report['calls']}.\n\n"
            "No selector fit or bulk JEV calls were started. Question/options and two solo outputs only were sent. "
            "The raw response and token usage remain server-only.\n")
    (args.out / 'reports/summary.md').write_text(text, encoding='utf-8')
    files = [p for p in sorted(args.out.rglob('*')) if p.is_file()]
    (args.out / 'checksums.sha256').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(args.out).as_posix()}\n' for p in files), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
