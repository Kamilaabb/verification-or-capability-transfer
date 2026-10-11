#!/usr/bin/env python3
"""Fixed JEV reference on all 45 code development pairs, using archived solo outputs.

Only question text and two independent outputs leave the server. Grades are read
after the pair's decisions are durably written. No new solve, grading or fit.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from gv_domain_tables import Archives, paired

DATASET='bigcodebench-b74c0d0-instruct-s20260917'
MANIFEST_HASH='5fc23a137b20357ca254f4a73fc9d72d4bca24adb05ca03b947a92d4917b4a30'
INSTRUCTIONS='Which response provides a correct Python solution that satisfies the programming task?'
CRITERIA={'A':"Response A's program correctly satisfies the task requirements",
          'B':"Response B's program correctly satisfies the task requirements"}
MODEL='jev-latest'

def now():return datetime.now(timezone.utc).isoformat()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def sample_zero(row):
    hits=[o for o in row['outputs'] if o['sample_index']==0]
    if len(hits)!=1:raise ValueError('one sample_index=0 is required')
    raw=hits[0].get('raw_output')
    if raw is not None and not isinstance(raw,str):raise ValueError('invalid raw output')
    return raw or ''

def body(problem,a,b):
    return {'model':MODEL,'state':f'Programming task:\n{problem}\n\nResponse A:\n{a}\n\nResponse B:\n{b}',
            'questions':{'correct':{'type':'choice','instructions':INSTRUCTIONS,'criteria':dict(CRITERIA)}}}

def digest(payload):
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def probability(response,order):
    ps=response['answers']['correct']['probabilities']
    values=[float(ps[k]) for k in ('A','B')]
    if any(not 0<=p<=1 for p in values) or abs(sum(values)-1)>1e-4:
        raise ValueError('invalid judge probability')
    return values[1 if order=='GV' else 0]

def choose(records):
    if set(records)!= {'GV','VG'} or any('response' not in r for r in records.values()):
        return True,None
    p=sum(probability(records[o]['response'],o) for o in ('GV','VG'))/2
    return p>=.5,p

def cache(path):
    result={}
    if not path.exists():return result
    raw=path.read_bytes()
    lines=raw.splitlines()
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if i==len(lines)-1 and not raw.endswith(b'\n'):break
            raise
        if 'response' in r:
            probability(r['response'],r['order'])
            result[r['request_sha256']]=(r['response'],str(path))
    return result

def load_inputs(ar,registry):
    solos={};configs={};runs={}
    cids=sorted({p[side]['condition'] for p in registry['pairs'] for side in ('g','v')})
    if cids!=[f'c{i}' for i in range(100,110)]:raise ValueError('unexpected upstream conditions')
    for cid in cids:
        run=ar.find(cid)
        if run is None:raise ValueError(f'missing solo {cid}')
        cfg=ar.config(run);d=cfg['dataset']
        if (d['id'],d['manifest'],d['manifest_rows'],d['manifest_sha256'])!=(DATASET,'development',832,MANIFEST_HASH):
            raise ValueError('unexpected dataset identity')
        rows={}
        for r in read_rows(ar.checked(run,'records/pass1.jsonl')):
            pid=str(r['problem_id'])
            if pid in rows or r['condition_id']!=cid:raise ValueError('duplicate/wrong solo record')
            rows[pid]={'problem':r['problem_text'],'raw_output':sample_zero(r)}
        if len(rows)!=832:raise ValueError('unexpected solo membership')
        solos[cid]=rows;configs[cid]=cfg;runs[cid]=run
    ids=list(solos[cids[0]])
    for rows in solos.values():
        if set(rows)!=set(ids):raise ValueError('solo memberships differ')
        if any(rows[pid]['problem']!=solos[cids[0]][pid]['problem'] for pid in ids):
            raise ValueError('archived task text differs')
    actual={tuple(sorted([p['g']['condition'],p['v']['condition']])) for p in registry['pairs']}
    if len(registry['pairs'])!=45 or actual!=set(itertools.combinations(cids,2)):
        raise ValueError('registry is not all 45 unique pairs')
    return solos,configs,runs,ids

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--artifacts',type=Path,default=Path('/path/to/project/artifacts'))
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--smoke',type=int,default=0,help='first N rows of first frozen pair, reused by full run')
    ap.add_argument('--resume-cache',type=Path)
    args=ap.parse_args()
    if not Path('/path/to/project').exists():raise SystemExit('server only')
    if args.smoke<0 or args.smoke>832:raise SystemExit('invalid smoke size')
    out=args.out
    if (out/'reports/summary.md').exists():raise SystemExit('completed run is immutable')
    (out/'reports').mkdir(parents=True,exist_ok=True)
    (out/'records').mkdir(exist_ok=True)
    frozen=ROOT/'pairs/bigcodebench_jev_dev832.yaml'
    config=yaml.safe_load(frozen.read_text(encoding='utf-8'))
    if config['instructions']!=INSTRUCTIONS or config['criteria']!=CRITERIA or config['model']!=MODEL or config['workers']!=2:
        raise ValueError('judge settings differ from frozen plan')
    registry_path=ROOT/config['pair_registry']
    registry=yaml.safe_load(registry_path.read_text(encoding='utf-8'))
    if registry['dataset']!=DATASET or registry['manifest']!='dev832':raise ValueError('wrong pair registry')
    ar=Archives(args.artifacts)
    solos,configs,runs,ids=load_inputs(ar,registry)
    source_commit=(ROOT/'SOURCE_COMMIT.txt').read_text().strip()
    dump(out/'RUN_MANIFEST.json',dict(run_id=out.name,analysis_id=config['analysis_id'],
         dataset_id=DATASET,split='development',protocol_reference='code_solve_v1',
         replicate_reference='r015',source_commit=source_commit,upstream_runs={c:r.name for c,r in runs.items()},
         judge_config=config,judge_config_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
         pair_registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),started_at_utc=now(),smoke=args.smoke))
    differing=sum(solos[p['g']['condition']][pid]['raw_output']!=solos[p['v']['condition']][pid]['raw_output']
                  for p in registry['pairs'] for pid in ids)
    dump(out/'reports/preflight.json',dict(pairs=45,rows=832,max_calls=2*differing,source_sha256=ar.sources,
         test_read=False,source='checked solo archives only'))
    if args.preflight:
        (out/'reports/summary.md').write_text(f'# Code JEV preflight\n\n45 pairs × 832 development rows verified; at most {2*differing} requests. No API call; no grade or test read.\n',encoding='utf-8')
        print(json.dumps({'preflight':'passed','pairs':45,'max_calls':2*differing}),flush=True)
        return
    # Reuse the existing network/retry implementation, without loading any new dataset.
    import jev_domain_sweep as api
    key=api.api.KEY_FILE.read_text().strip()
    dump(out/'models.json',dict(time=now(),response=api.request(key,'models')))
    reusable={};upstream_cache={}
    if args.resume_cache:
        for path in sorted(args.resume_cache.glob('*/records/jev_responses.jsonl')):
            reusable.update(cache(path));upstream_cache[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        dump(out/'upstream_cache.json',upstream_cache)
    reports=[];total_tokens=0;new_calls=0
    pairs=registry['pairs'][:1] if args.smoke else registry['pairs']
    target_ids=ids[:args.smoke] if args.smoke else ids
    for pair in pairs:
        gc,vc=pair['g']['condition'],pair['v']['condition']
        pairout=out/pair['id'];(pairout/'records').mkdir(parents=True,exist_ok=True)
        response_path=pairout/'records/jev_responses.jsonl'
        if response_path.exists():raise ValueError('partial output: resume into a new run with --resume-cache')
        current={};todo=[];identical=set();reused=0
        with response_path.open('w',encoding='utf-8') as stream:
            for pid in target_ids:
                g,v=solos[gc][pid],solos[vc][pid]
                if g['raw_output']==v['raw_output']:
                    identical.add(pid);continue
                for order,aa,bb in [('GV',g,v),('VG',v,g)]:
                    payload=body(g['problem'],aa['raw_output'],bb['raw_output'])
                    sha=digest(payload)
                    rec=dict(pair=pair['id'],split='development',problem_id=pid,order=order,request_sha256=sha)
                    if sha in reusable:
                        response,origin=reusable[sha];rec.update(response=response,reused_from=origin)
                        stream.write(json.dumps(rec)+'\n');current[pid,order]=rec;reused+=1
                    else:todo.append((rec,payload))
            stream.flush()
            def call(item):
                rec,payload=item;start=time.perf_counter();rec['called_at_utc']=now()
                try:
                    response=api.request(key,'systemone',payload)
                    probability(response,rec['order']);rec['response']=response
                except api.ServiceBlocked:raise
                except Exception as exc:rec['error']=type(exc).__name__
                rec['latency_ms']=(time.perf_counter()-start)*1000
                return rec
            print(f"{pair['id']}: {len(todo)} new calls, {reused} reused",flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(call,item) for item in todo]
                for i,future in enumerate(concurrent.futures.as_completed(futures),1):
                    try:rec=future.result()
                    except BaseException:
                        for pending in futures:pending.cancel()
                        raise
                    stream.write(json.dumps(rec)+'\n');stream.flush()
                    current[rec['problem_id'],rec['order']]=rec
                    if 'response' in rec:reusable.setdefault(rec['request_sha256'],(rec['response'],str(response_path)))
                    if i%100==0:print(f"{pair['id']}: {i}/{len(todo)} calls",flush=True)
        decisions=[];failed=0
        for pid in target_ids:
            if pid in identical:chosen,p=True,None
            else:
                chosen,p=choose({o:current[pid,o] for o in ('GV','VG')});failed+=p is None
            decisions.append(dict(problem_id=pid,pick_v=chosen,prob_v=p,identical_raw=pid in identical))
        (pairout/'records/decisions.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in decisions),encoding='utf-8')
        # Decision file is closed before any evaluation labels are loaded.
        ug,uv=ar.grades(runs[gc]),ar.grades(runs[vc])
        if set(ug)!=set(ids) or set(uv)!=set(ids) or sum(ug.values())>sum(uv.values()):
            raise ValueError('grade membership or frozen G/V ordering mismatch')
        yg,yv=[ug[p] for p in target_ids],[uv[p] for p in target_ids]
        yj=[uv[d['problem_id']] if d['pick_v'] else ug[d['problem_id']] for d in decisions]
        oracle=sum(a or b for a,b in zip(yg,yv));baseline=max(sum(yg),sum(yv));headroom=oracle-baseline
        fresh=[r for r in current.values() if 'reused_from' not in r]
        tokens=sum(int(r.get('response',{}).get('usage',{}).get('input_tokens',0)) for r in fresh)
        versions=sorted({r['response'].get('model','unknown') for r in current.values() if 'response' in r})
        report=dict(domain='code',pair=pair['id'],split='development',g_condition=gc,v_condition=vc,
            g=configs[gc]['model']['model_id'],v=configs[vc]['model']['model_id'],n=len(target_ids),
            G=sum(yg),V=sum(yv),baseline2=baseline,JEV=sum(yj),oracle=oracle,headroom=headroom,
            recovered=(sum(yj)-baseline)/headroom if headroom else None,vs_baseline2=paired(yj,yv),
            failed_rows=failed,failed_calls=sum('response' not in r for r in current.values()),
            new_calls=len(todo),reused_calls=reused,new_input_tokens=tokens,models=versions,
            runs=[runs[gc].name,runs[vc].name])
        (pairout/'reports').mkdir(exist_ok=True);dump(pairout/'reports/jev.json',report)
        (pairout/'reports/summary.md').write_text(f"# {pair['id']} — code development JEV\n\nN={len(target_ids)}; G={sum(yg)}; V={sum(yv)}; JEV={sum(yj)}; oracle={oracle}; failed rows={failed}.\n",encoding='utf-8')
        reports.append(report);total_tokens+=tokens;new_calls+=len(todo)
        dump(out/'reports/progress.json',dict(pairs=reports,new_calls=new_calls,new_input_tokens=total_tokens))
        print(f"{pair['id']}: JEV={sum(yj)}/{len(target_ids)}, failed={failed}",flush=True)
    dump(out/'reports/jev_all.json',dict(pairs=reports,source_sha256=ar.sources,new_calls=new_calls,
         new_input_tokens=total_tokens,estimated_new_usd=total_tokens*.042/1e6,upstream_cache_sha256=upstream_cache))
    lines=['# Code JEV: development-only external-judge reference','',
           'Fixed double-order mean; ties/failures deliver registry V. Decisions precede evaluation labels. No tests or gold sent. JEV does not execute code.','',
           '|pair|N|G|V|max|Oracle|JEV|JEV − max|failed rows|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:lines.append(f"|{r['pair']}|{r['n']}|{r['G']}|{r['V']}|{r['baseline2']}|{r['oracle']}|{r['JEV']}|{r['JEV']-r['baseline2']:+d}|{r['failed_rows']}|")
    lines+=['',f'New calls: {new_calls}; new input tokens: {total_tokens}; estimated cost ${total_tokens*.042/1e6:.4f} (not an invoice).',
            'Development exploratory results; test300 remains unread. No new solve or selector fit.']
    (out/'reports/summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    checks=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name!='checksums.sha256':checks.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out).as_posix()}')
    (out/'checksums.sha256').write_text('\n'.join(checks)+'\n',encoding='utf-8')

if __name__=='__main__':main()
