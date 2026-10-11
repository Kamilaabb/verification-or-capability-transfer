#!/usr/bin/env python3
"""Independently re-evaluate saved NL decisions; no fit, inference or new grading."""
import argparse
import hashlib
import json
from pathlib import Path
from nl_selector_crossfit import registration, manifest_ids, grades, dump, checksums, ROOT
from nl_jev_version_probe import inputs, body, checked_file
from nl_jev_eval901 import validate_response
from code_jev_dev832 import digest, choose, read_rows
from code_selector_backfill import assignments, validate_decisions
from gv_domain_tables import Archives


def checked_run(path):
    hashes={}
    for line in (path/'checksums.sha256').read_text().splitlines():
        h,rel=line.split(maxsplit=1);hashes[rel]=h
    for rel,h in hashes.items():checked_file(path/rel,h)
    if not (path/'reports/summary.md').exists():raise ValueError('missing human report')


def verify(args):
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'reports').mkdir()
    result={'status':'failed','pairs':[]}
    try:
        reg,ds=registration();ids=manifest_ids(ds)
        reg,ds,ar,other_ids,questions,solos,runs=inputs(ROOT/'artifacts')
        if other_ids!=ids:raise ValueError('changed manifest order')
        if len(list(args.selectors.glob('nl_reasoning__*/reports/selector_dev.json')))!=12:
            raise ValueError('expected 12 complete selector reports')
        jr=json.loads((args.jev/'reports/jev_all.json').read_text())
        if jr['status']!='passed' or len(jr['pairs'])!=12:raise ValueError('JEV incomplete')
        checked_run(args.jev)
        for pair in reg['pairs']:
            sp=args.selectors/f"nl_reasoning__{pair['id']}";jp=args.jev/f"nl_reasoning__{pair['id']}"
            checked_run(sp);checked_run(jp)
            gc,vc=pair['g']['condition'],pair['v']['condition']
            yg,yv=grades(reg,pair,ids,{s:runs[pair[s]['condition']] for s in ('g','v')},ar)
            folds=assignments(yg,yv)
            sr=json.loads((sp/'reports/selector_dev.json').read_text())
            cfg=__import__('yaml').safe_load((sp/'condition.resolved.yaml').read_text())
            cm=json.loads((sp/'RUN_MANIFEST.json').read_text())
            for rel,h in cm['referenced_file_sha256'].items():checked_file(ROOT/rel,h)
            if digest({'behavior':cfg,'referenced_file_sha256':cm['referenced_file_sha256']})!=cm['condition_sha256']:
                raise ValueError('resolved behavioral hash mismatch')
            selector_counts={};file_hashes={}
            for name in ('primary','secondary'):
                path=sp/f'records/{name}_decisions.jsonl'
                yy,seedcounts=validate_decisions(path,ids,yg,yv,folds)
                if int(yy.sum())!=sr['selectors'][name]['correct'] or seedcounts!=sr['selectors'][name]['per_seed']:
                    raise ValueError('selector report differs from saved per-seed decisions')
                selector_counts[name]=int(yy.sum());file_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            records={}
            for r in read_rows(jp/'records/jev_responses.jsonl'):
                validate_response(r)
                pid,order=r['problem_id'],r['order']
                if pid not in set(ids) or order not in ('GV','VG') or (pid,order) in records:
                    raise ValueError('wrong/duplicate judge response membership')
                g,v=solos[gc][pid],solos[vc][pid];q=questions[pid]
                a,b=(g,v) if order=='GV' else (v,g)
                payload=body(q['question'],q['options'],a['text'],b['text'])
                if digest(payload)!=r['request_sha256']:raise ValueError('judge request cache differs')
                records[pid,order]=r
            decisions=list(read_rows(jp/'records/decisions.jsonl'))
            if [r['problem_id'] for r in decisions]!=ids:raise ValueError('missing/duplicate/reordered JEV rows')
            yj=[];expected_keys=set();same_count=0
            for i,d in enumerate(decisions):
                if set(d)!={'problem_id','pick_v','prob_v'} or type(d['pick_v']) is not bool:
                    raise ValueError('invalid decision schema')
                pid=ids[i];g,v=solos[gc][pid],solos[vc][pid]
                if g['letter'] is not None and g['letter']==v['letter']:
                    selected,p=True,None;same_count+=1
                else:
                    selected,p=choose({o:records[pid,o] for o in ('GV','VG')})
                    expected_keys.update((pid,o) for o in ('GV','VG'))
                if d['pick_v']!=selected or d['prob_v']!=p:raise ValueError('saved judge choice differs from raw probabilities')
                yj.append(int(yv[i] if selected else yg[i]))
            if set(records)!=expected_keys:raise ValueError('unexpected or missing response rows')
            pr=json.loads((jp/'reports/jev.json').read_text())
            fresh=[r for r in records.values() if 'reused_from' not in r]
            usage={k:sum(r['response']['usage'][k] for r in fresh) for k in ('input_tokens','output_tokens')}
            expected={'n':901,'G':int(yg.sum()),'V':int(yv.sum()),'JEV':sum(yj),
                      'new_calls':len(fresh),'reused_calls':len(records)-len(fresh),
                      'new_usage':usage,'no_call_equal_letters':same_count,'models':['jev-1.13.0'],
                      'source_unparseable_g':sum(solos[gc][p]['letter'] is None for p in ids),
                      'source_unparseable_v':sum(solos[vc][p]['letter'] is None for p in ids)}
            for k,v in expected.items():
                if pr[k]!=v:raise ValueError('JEV aggregate differs from response/decision recomputation')
            for k in ('failed_rows','failed_calls','missing_rows','unparseable_judge_rows'):
                if pr[k]!=0:raise ValueError('nonzero judge failures')
            result['pairs'].append(dict(pair=pair['id'],g=pair['g']['model'],v=pair['v']['model'],n=901,
                G=int(yg.sum()),stronger_correct=int(yv.sum()),selectors=selector_counts,JEV=sum(yj),
                per_seed={k:sr['selectors'][k]['per_seed'] for k in ('primary','secondary')},
                failed_rows=0,missing_rows=0,unparseable_judge_rows=0,
                source_unparseable_g=pr['source_unparseable_g'],source_unparseable_v=pr['source_unparseable_v'],
                new_calls=len(fresh),reused_calls=len(records)-len(fresh),new_usage=usage,
                selector_path=str(sp),jev_path=str(jp),decision_sha256=dict(file_hashes,
                    jev=hashlib.sha256((jp/'records/decisions.jsonl').read_bytes()).hexdigest())))
        result.update(status='passed',selector_files=24,selector_rows=21624,jev_files=12,jev_rows=10812,
            source_sha256=ar.sources,selectors_root=str(args.selectors),jev_root=str(args.jev),metric='u0')
    except Exception as exc:
        result.update(error_type=type(exc).__name__)
        dump(args.out/'reports/verification.json',result)
        (args.out/'reports/summary.md').write_text(f'# NL verification FAILED\n\n{type(exc).__name__}; no gate relaxed.\n')
        raise
    dump(args.out/'reports/verification.json',result)
    lines=['# NL selectors and JEV: verified', '',
        '12 design pairs × 901 rows. Every saved decision, seed/fold assignment and judge probability was rechecked. Primary u0 only.', '',
        '|G / stronger V|V solo|Primary|Secondary|JEV|', '|---|---:|---:|---:|---:|']
    for r in result['pairs']:
        lines.append(f"|{r['g']} / {r['v']}|{r['stronger_correct']}|{r['selectors']['primary']}|{r['selectors']['secondary']}|{r['JEV']}|")
    lines+=['','Zero failed, unparseable judge or missing decision rows. Every actual JEV response is jev-1.13.0. Solo unparseable answers remain included as originally graded. No new generation or grading.']
    (args.out/'reports/summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');checksums(args.out)
    print(json.dumps({'status':result['status'],'pairs':12,'selector_rows':21624,'jev_rows':10812}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selectors',type=Path,required=True);ap.add_argument('--jev',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args())
