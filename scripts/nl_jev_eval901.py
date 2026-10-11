#!/usr/bin/env python3
"""Knowledge-prompt JEV on twelve NL pairs; every response must be 1.13.0."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from code_jev_dev832 import digest, probability, choose, read_rows, now
from gv_domain_tables import paired
from nl_jev_version_probe import inputs, body, check_version, checked_file, EXPECTED_MODEL
from nl_selector_crossfit import dump, checksums, registration
import mc_jev_judge_test1500 as mc
import jev_domain_sweep as api


def validate_response(rec):
    if 'response' not in rec:
        raise RuntimeError('judge call failed')
    check_version(rec['response'])
    probability(rec['response'],rec['order'])
    usage = rec['response'].get('usage')
    if not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0
                                         for k in ('input_tokens','output_tokens')):
        raise ValueError('missing/invalid per-call token accounting')


def call_requests(todo, key, stream, current, workers=4, request_fn=api.request):
    """Bound in-flight calls. On a failed gate, stop scheduling and retain in-flight returns."""
    def call(item):
        rec,payload = item
        rec = dict(rec,called_at_utc=now())
        start = time.perf_counter()
        try:
            rec['response'] = request_fn(key,'systemone',payload)
        except Exception as exc:
            rec['error'] = type(exc).__name__
        rec['latency_ms'] = (time.perf_counter()-start)*1000
        return rec
    def save(rec):
        stream.write(json.dumps(rec,ensure_ascii=False)+'\n'); stream.flush()
        current[rec['problem_id'],rec['order']] = rec
    iterator = iter(todo)
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    pending = set()
    done_count = 0
    try:
        for _ in range(workers):
            item = next(iterator,None)
            if item is not None: pending.add(pool.submit(call,item))
        while pending:
            done,pending = cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            records = [future.result() for future in done]
            for rec in records: save(rec)
            for rec in records: validate_response(rec)
            done_count += len(records)
            if done_count%100<len(records): print(f'calls {done_count}/{len(todo)} complete',flush=True)
            for _ in records:
                item = next(iterator,None)
                if item is not None: pending.add(pool.submit(call,item))
    except BaseException:
        # Already issued calls cannot be recalled. Save their responses and token usage,
        # but never submit the next request after a failed version/parse/network gate.
        for future in pending:
            if not future.cancel(): save(future.result())
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


def probe_cache(root):
    report = json.loads((root/'reports/version_probe.json').read_text(encoding='utf-8'))
    if report['status']!='passed' or report['returned_model']!=EXPECTED_MODEL or report['calls']!=1:
        raise ValueError('a passed one-call version probe is required')
    hashes = {}
    for line in (root/'checksums.sha256').read_text().splitlines():
        sha,rel=line.split(maxsplit=1); hashes[rel]=sha
    path = checked_file(root/'records/jev_responses.jsonl',hashes['records/jev_responses.jsonl'])
    rows = list(read_rows(path))
    if len(rows)!=1: raise ValueError('expected one probe response')
    validate_response(rows[0])
    return {rows[0]['request_sha256']:rows[0]}, hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'reports').mkdir()
    report = {'status':'failed','required_model':EXPECTED_MODEL,'pairs':[],
              'source_commit':(ROOT/'SOURCE_COMMIT.txt').read_text().strip()}
    try:
        registration()
        reg,ds,ar,ids,questions,solos,runs=inputs(args.artifacts)
        cache,cache_hash=probe_cache(args.probe)
        report.update(probe_run=str(args.probe),probe_response_sha256=cache_hash,metric='u0',
            manifest_sha256=ds['manifests']['eval']['sha256'])
        dump(args.out/'RUN_MANIFEST.json',dict(run_id=args.out.name,source_commit=report['source_commit'],
            upstream_runs={c:r.name for c,r in runs.items()},dataset_id=reg['dataset'],
            pair_registry_sha256=hashlib.sha256((ROOT/'pairs/nl_selector_jev_eval901.yaml').read_bytes()).hexdigest(),
            judge=dict(requested_model=mc.MODEL,required_returned_model=EXPECTED_MODEL,
                       instructions=mc.INSTRUCTIONS,criteria=mc.CRITERIA,orders=['GV','VG'],workers=4),
            probe_run=str(args.probe),probe_response_sha256=cache_hash))
        key=mc.KEY_FILE.read_text().strip()
        dump(args.out/'models.json',{'time':now(),'response':api.request(key,'models')})
        for pair in reg['pairs']:
            out=args.out/f"nl_reasoning__{pair['id']}"
            out.mkdir(exist_ok=False); (out/'records').mkdir(); (out/'reports').mkdir()
            gc,vc=pair['g']['condition'],pair['v']['condition']
            current,todo,same={},[],set()
            report['active_pair']=pair['id']
            with (out/'records/jev_responses.jsonl').open('x',encoding='utf-8') as stream:
                for pid in ids:
                    g,v=solos[gc][pid],solos[vc][pid]
                    if g['letter'] is not None and g['letter']==v['letter']:
                        same.add(pid); continue
                    q=questions[pid]
                    for order,a,b in [('GV',g,v),('VG',v,g)]:
                        payload=body(q['question'],q['options'],a['text'],b['text'])
                        sha=digest(payload)
                        rec=dict(pair=pair['id'],split='development',problem_id=pid,order=order,
                                 request_sha256=sha,state_sha256=hashlib.sha256(payload['state'].encode()).hexdigest())
                        if sha in cache:
                            rec.update(response=cache[sha]['response'],reused_from=str(args.probe))
                            validate_response(rec)
                            stream.write(json.dumps(rec)+'\n');current[pid,order]=rec
                        else: todo.append((rec,payload))
                stream.flush()
                print(f"{pair['id']}: {len(todo)} fresh calls; {len(same)} equal letters",flush=True)
                try:
                    call_requests(todo,key,stream,current)
                except BaseException:
                    (out/'reports/summary.md').write_text('# NL JEV FAILED\n\nStopped scheduling on first failed gate; any in-flight returns retained. No completed decisions claimed.\n',encoding='utf-8')
                    raise
            decisions=[]
            with (out/'records/decisions.jsonl').open('x',encoding='utf-8') as stream:
                for pid in ids:
                    chosen,p=(True,None) if pid in same else choose({o:current[pid,o] for o in ('GV','VG')})
                    if pid not in same and p is None: raise ValueError('missing judge decision')
                    rec=dict(problem_id=pid,pick_v=bool(chosen),prob_v=p)
                    stream.write(json.dumps(rec)+'\n');decisions.append(rec)
            # All decisions are closed before labels are read. Only archived u0 is evaluated.
            ug,uv=ar.grades(runs[gc],outcome='u0'),ar.grades(runs[vc],outcome='u0')
            for cid,labels in ((gc,ug),(vc,uv)):
                checked_file(ar.checked(runs[cid],'records/grades.jsonl'),reg['solos'][cid]['sha256']['records/grades.jsonl'])
                if set(labels)!=set(ids) or sum(labels.values())!=reg['solos'][cid]['correct']:
                    raise ValueError('primary u0 membership/count mismatch')
            yj=[uv[d['problem_id']] if d['pick_v'] else ug[d['problem_id']] for d in decisions]
            fresh=[r for r in current.values() if 'reused_from' not in r]
            usage={k:sum(r['response']['usage'][k] for r in fresh) for k in ('input_tokens','output_tokens')}
            pr=dict(domain='nl_reasoning',split='dev',pair=pair['id'],g=pair['g']['model'],v=pair['v']['model'],
                g_condition=gc,v_condition=vc,n=901,G=sum(ug.values()),V=sum(uv.values()),JEV=sum(yj),
                oracle=sum(ug[p] or uv[p] for p in ids),vs_baseline2=paired(yj,[uv[p] for p in ids]),
                failed_rows=0,failed_calls=0,missing_rows=0,unparseable_judge_rows=0,
                source_unparseable_g=sum(solos[gc][p]['letter'] is None for p in ids),
                source_unparseable_v=sum(solos[vc][p]['letter'] is None for p in ids),
                no_call_equal_letters=len(same),new_calls=len(fresh),reused_calls=len(current)-len(fresh),
                new_usage=usage,models=sorted({r['response']['model'] for r in current.values()}),
                runs=[runs[c].name for c in (gc,vc)],metric='u0')
            dump(out/'reports/jev.json',pr)
            (out/'reports/summary.md').write_text(f"# NL JEV\n\n{pair['id']}: JEV={pr['JEV']}/901; "
                f"stronger solo={pr['V']}/901. All returned versions are {EXPECTED_MODEL}; zero failed or missing rows. "
                "Knowledge prompt and double-order mean unchanged. Equal parseable letters deliver V without a call.\n",encoding='utf-8')
            checksums(out)
            report['pairs'].append(pr);dump(args.out/'reports/progress.json',report)
            print(f"{pair['id']}: JEV={pr['JEV']}/901; failed=0",flush=True)
        report.update(status='passed',source_sha256=ar.sources)
        report.pop('active_pair',None)
    except Exception as exc:
        report.update(error_type=type(exc).__name__)
        responses=[r for p in args.out.glob('*/records/jev_responses.jsonl') for r in read_rows(p)]
        report.update(observed_models=sorted({str(r['response'].get('model')) for r in responses if 'response' in r}),
                      failed_calls=sum('response' not in r for r in responses))
    dump(args.out/'reports/jev_all.json',report)
    lines=['# NL JEV backfill', '', f"Status: {report['status']}; required model: {EXPECTED_MODEL}.", '',
           '|Pair|N|Stronger solo|JEV|Fresh calls|Reused calls|', '|---|---:|---:|---:|---:|---:|']
    for r in report['pairs']:
        lines.append(f"|{r['pair']}|901|{r['V']}|{r['JEV']}|{r['new_calls']}|{r['reused_calls']}|")
    if report['status']!='passed':
        lines+=['',f"Stopped: {report.get('error_type')}; observed models {report.get('observed_models',[])}. No settings relaxed."]
    (args.out/'reports/summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    checksums(args.out)
    print(json.dumps({'status':report['status'],'completed_pairs':len(report['pairs']),
                      'error_type':report.get('error_type'),'observed_models':report.get('observed_models')}),flush=True)
    if report['status']!='passed':raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--artifacts',type=Path,default=Path('/path/to/project/artifacts'))
    args=ap.parse_args()
    if not Path('/path/to/project').exists():raise SystemExit('server only')
    run(args)


if __name__=='__main__':main()
