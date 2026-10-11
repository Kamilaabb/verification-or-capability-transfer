#!/usr/bin/env python3
"""Frozen JEV judge on math/knowledge test1500 by default, server only.

Reuses successful requests by (domain, exact request-state SHA256). Sends only
problem/options and two independent raw responses. Writes decisions before
reading that split's grades for evaluation. Does not train or tune a judge.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
import mc_jev_judge_test1500 as api
import run_c002_dev30 as base
from gv_domain_tables import Archives, MODELS, paired

INSTRUCTIONS = {
    "math": "Which response gives the correct final answer to the math problem?",
    "knowledge": api.INSTRUCTIONS,
}


class ServiceBlocked(RuntimeError):
    pass


def request(key, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    for attempt in range(4):
        req = urllib.request.Request(f"{api.API}/{path}", data=data,
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (401,402,403):
                raise ServiceBlocked(f"service_http_{exc.code}") from None
            if exc.code in (413,422) or attempt==3:
                raise RuntimeError(f"http_{exc.code}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt==3: raise RuntimeError("network_timeout_or_unavailable") from None
        time.sleep(2**attempt)


def pick_v(records):
    if len(records)!=2 or any("response" not in r for r in records):
        return True,None
    ps=[]
    for r in records:
        p=r["response"]["answers"]["correct"]["probabilities"]
        value=float(p["B" if r["order"]=="GV" else "A"])
        if not 0<=value<=1: raise ValueError("invalid JEV probability")
        ps.append(value)
    probability=sum(ps)/2
    return probability>=0.5,probability


def load_source(domain, ids):
    from datasets import load_dataset
    wanted=set(ids)
    if domain=="math":
        ds=load_dataset("nlile/hendrycks-MATH-benchmark",
            revision="465bcdb36f5962aa3512891498966df785fc3c18",split="train")
        ds=ds.select_columns(["unique_id","problem"])
        out={str(r["unique_id"]):{"problem":r["problem"]} for r in ds if str(r["unique_id"]) in wanted}
    else:
        ds=load_dataset("TIGER-Lab/MMLU-Pro",
            revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",split="test")
        ds=ds.select_columns(["question_id","question","options"])
        out={str(r["question_id"]):{"question":r["question"],"options":list(r["options"])}
             for r in ds if str(r["question_id"]) in wanted}
    if set(out)!=wanted: raise ValueError("source membership mismatch")
    return out


def state(domain, q, a, b):
    if domain=="knowledge": return api.state(q["question"],q["options"],a,b)
    return f"Problem:\n{q['problem']}\n\nResponse A:\n{a}\n\nResponse B:\n{b}"


def cache_from_archives(artifacts, domain):
    cache={}
    pattern="*__mc-jev*__posthoc" if domain=="knowledge" else "*__math_jev*__posthoc"
    for run in sorted(artifacts.glob(pattern)):
        path=run/"records/jev_responses.jsonl"
        if not path.exists() or not (run/"checksums.sha256").exists(): continue
        # Verify before reusing; never replace the old immutable archive.
        Archives(artifacts).checked(run,"records/jev_responses.jsonl")
        for r in api.read_jsonl(path):
            if "response" in r:
                cache.setdefault(r["state_sha256"],(r,str(path)))
    return cache


def resume_test_cache(root, domain):
    """Read prior interrupted test-only requests without changing the old run."""
    cache, hashes = {}, {}
    for path in sorted((root/domain).glob('*/test/records/jev_responses.jsonl')):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if not line.strip(): continue
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # SIGTERM may interrupt the final write; never hide an earlier bad row.
                if i == len(lines)-1 and not raw.endswith(b'\n'): break
                raise
            if r.get('split') != 'test': raise ValueError('non-test row in resume cache')
            if 'response' in r: cache[r['state_sha256']] = (r, str(path))
    return cache, hashes


def parse_math(outputs):
    parsed={}
    for pid,o in outputs.items():
        text=base.answer_text(o)
        try: parsed[pid]=base.math_parse(text)[0] if text else []
        except (Exception,base.MathVerifyTimeout): parsed[pid]=[]
    return parsed


def equivalent(domain, g, v, pg=None, pv=None):
    if domain=="knowledge":
        a,b=g.get("extracted_answer"),v.get("extracted_answer")
        return a is not None and b is not None and a==b
    if not pg or not pv: return False
    from math_verify import verify
    try:
        return bool(verify(pg,pv,float_rounding=6,numeric_precision=15,strict=True,
            allow_set_relation_comp=False,timeout_seconds=10,raise_on_error=True))
    except (Exception,base.MathVerifyTimeout): return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--domain",choices=["math","knowledge","both"],default="both")
    ap.add_argument("--split",choices=["test","dev","both"],default="test")
    ap.add_argument("--resume-cache",type=Path,help="prior stopped run root; reuses test requests only")
    args=ap.parse_args()
    if not Path("/path/to/project").exists():
        raise SystemExit("Server only")
    if (args.out/"reports/summary.md").exists(): raise SystemExit("completed run is immutable")
    (args.out/"reports").mkdir(parents=True,exist_ok=True)
    key=api.KEY_FILE.read_text().strip()
    model_info=request(key,"models")
    (args.out/"models.json").write_text(json.dumps({"time":datetime.now(timezone.utc).isoformat(),"response":model_info},indent=2))
    ar=Archives(ROOT/"artifacts"); reports=[]; new_tokens_total=0; upstream_hashes={}
    splits=("test","dev") if args.split=="both" else (args.split,)
    for domain,mapping in MODELS.items():
        if args.domain not in (domain,"both"): continue
        cache=cache_from_archives(ROOT/"artifacts",domain)
        if args.resume_cache:
            resumed, hashes = resume_test_cache(args.resume_cache, domain)
            cache.update(resumed); upstream_hashes.update(hashes)
            (args.out/'upstream_cache.json').write_text(json.dumps(dict(
                root=str(args.resume_cache), sha256=upstream_hashes),indent=2)+'\n')
        outputs={}; runs={}; accuracy={}; model_names={}
        for dc,tc in mapping.items():
            runs[dc]={}
            for split,cid in (("dev",dc),("test",tc)):
                run=ar.find(cid)
                if run is None: raise ValueError(f"missing solo {cid}")
                runs[dc][split]=run
                outputs[dc,split]={str(r["problem_id"]):r["outputs"][0]
                    for r in api.read_jsonl(ar.checked(run,"records/pass1.jsonl"))}
            accuracy[dc]=sum(ar.grades(runs[dc]["dev"]).values())
            model_names[dc]=ar.config(runs[dc]["dev"])["model"]["model_id"]
        ids_by_split={s:list(outputs[next(iter(mapping)),s]) for s in ("dev","test")}
        if len(ids_by_split["dev"])!=1000 or len(ids_by_split["test"])!=1500:
            raise ValueError("unexpected membership")
        if set(ids_by_split["dev"])&set(ids_by_split["test"]): raise ValueError("dev/test overlap")
        for (dc,split),rows in outputs.items():
            if set(rows)!=set(ids_by_split[split]): raise ValueError("solo membership mismatch")
        source=load_source(domain,[pid for s in splits for pid in ids_by_split[s]])
        parsed={k:parse_math(v) for k,v in outputs.items() if k[1] in splits} if domain=="math" else {}
        for split in splits:
            ids=ids_by_split[split]
            for a,b in itertools.combinations(mapping,2):
                a,b=sorted((a,b),key=lambda c:(accuracy[c],c))
                pair=f"{domain}__{a}__{b}"
                out=args.out/pair/split
                if (out/"reports/summary.md").exists():
                    report=json.loads((out/"reports/jev.json").read_text())
                    reports.append(report); new_tokens_total+=report["new_input_tokens"]
                    continue
                (out/"records").mkdir(parents=True,exist_ok=True); (out/"reports").mkdir(exist_ok=True)
                g,v=outputs[a,split],outputs[b,split]
                dis=[p for p in ids if not equivalent(domain,g[p],v[p],
                    parsed.get((a,split),{}).get(p),parsed.get((b,split),{}).get(p))]
                path=out/"records/jev_responses.jsonl"
                current={(r["problem_id"],r["order"]):r for r in api.read_jsonl(path)} if path.exists() else {}
                todo=[]; new_tokens=0; reused=0
                with path.open("a") as f:
                    for pid in dis:
                        for order,aa,bb in (("GV",g[pid],v[pid]),("VG",v[pid],g[pid])):
                            st=state(domain,source[pid],aa.get("raw_output") or "",bb.get("raw_output") or "")
                            digest=hashlib.sha256(st.encode()).hexdigest()
                            if (pid,order) in current:
                                if current[pid,order]["state_sha256"]!=digest: raise ValueError("cache state changed")
                                continue
                            rec=dict(pair=pair,split=split,problem_id=pid,order=order,state_sha256=digest)
                            if digest in cache:
                                old,origin=cache[digest]
                                rec.update(response=old["response"],reused_from=origin)
                                f.write(json.dumps(rec)+"\n"); current[pid,order]=rec
                            else: todo.append((rec,st))
                    f.flush()
                    def call(item):
                        rec,st=item; start=time.perf_counter()
                        rec["called_at_utc"]=datetime.now(timezone.utc).isoformat()
                        try:
                            rec["response"]=request(key,"systemone",{"model":api.MODEL,"state":st,
                                "questions":{"correct":{"type":"choice","instructions":INSTRUCTIONS[domain],"criteria":api.CRITERIA}}})
                            pick_v([rec,{**rec,"order":"VG" if rec['order']=='GV' else 'GV'}])
                        except ServiceBlocked: raise
                        except Exception as exc:
                            rec.pop("response",None); rec["error"]=type(exc).__name__
                        rec["latency_ms"]=(time.perf_counter()-start)*1000
                        return rec
                    print(f"{pair} {split}: {len(dis)} differing/unparseable rows, {len(todo)} new calls",flush=True)
                    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                        for i,r in enumerate(pool.map(call,todo),1):
                            f.write(json.dumps(r)+"\n"); f.flush(); current[r['problem_id'],r['order']]=r
                            if 'response' in r: cache.setdefault(r['state_sha256'],(r,str(path)))
                            if i%200==0: print(f"{pair} {split}: {i}/{len(todo)}",flush=True)
                decisions=[]; failed=0; rejected=0; input_tokens=0; versions=set()
                for pid in ids:
                    records=[current[pid,o] for o in ("GV","VG")] if pid in set(dis) else []
                    chosen,p=pick_v(records) if records else (True,None)
                    failed+=bool(records) and p is None
                    for r in records:
                        if "response" in r:
                            tok=int(r['response'].get('usage',{}).get('input_tokens',0)); input_tokens+=tok
                            versions.add(r['response'].get('model','unknown'))
                            if 'reused_from' in r: reused+=1
                            else: new_tokens+=tok
                        else: rejected+=1
                    decisions.append(dict(problem_id=pid,pick_v=chosen,prob_v=p))
                (out/"records/decisions.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in decisions))
                ug,uv=ar.grades(runs[a][split]),ar.grades(runs[b][split])
                yg,yv=[ug[p] for p in ids],[uv[p] for p in ids]
                yj=[uv[r['problem_id']] if r['pick_v'] else ug[r['problem_id']] for r in decisions]
                oracle=sum(x or y for x,y in zip(yg,yv)); h=oracle-sum(yv)
                report=dict(domain=domain,pair=pair,split=split,g_condition=a,v_condition=b,
                    g=model_names[a],v=model_names[b],runs=[runs[a][split].name,runs[b][split].name],
                    n=len(ids),G=sum(yg),V=sum(yv),JEV=sum(yj),oracle=oracle,headroom=h,
                    recovered=(sum(yj)-sum(yv))/h if h else None,vs_baseline2=paired(yj,yv),
                    disagreement_rows=len(dis),failed_rows=failed,failed_calls=rejected,reused_calls=reused,
                    input_tokens=input_tokens,new_input_tokens=new_tokens,models=sorted(versions),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                (out/"reports/jev.json").write_text(json.dumps(report,indent=2)+'\n')
                (out/"reports/summary.md").write_text(f"# {pair} {split}\n\nN={len(ids)}; G={sum(yg)}, V={sum(yv)}, JEV={sum(yj)}, oracle={oracle}; failed rows={failed}; new calls={len(todo)}, reused={reused}.\n")
                reports.append(report); new_tokens_total+=new_tokens
                (args.out/"reports/progress.json").write_text(json.dumps(dict(pairs=reports,new_input_tokens=new_tokens_total,estimated_new_usd=new_tokens_total*.042/1e6),indent=2)+'\n')
                print(f"{pair} {split}: JEV={sum(yj)}, failed={failed}, cumulative estimated new USD={new_tokens_total*.042/1e6:.3f}",flush=True)
    lines=['# JEV math and knowledge complete sweep','',
        f"Evaluated splits: {', '.join(splits)}; fixed double-order average, ties/failures deliver development-stronger V. Only the evaluated split's problems/options and raw independent responses were sent. Development solo accuracy fixes roles only. Decisions precede target-label evaluation.",'',
        '|domain|pair|split|N|G|V|JEV|oracle|failed rows|new input tokens|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        lines.append(f"|{r['domain']}|{r['g'].split('/')[-1]} / {r['v'].split('/')[-1]}|{r['split']}|{r['n']}|{r['G']}|{r['V']}|{r['JEV']}|{r['oracle']}|{r['failed_rows']}|{r['new_input_tokens']}|")
    lines+=['',f'New input tokens {new_tokens_total}; estimated new API cost ${new_tokens_total*.042/1e6:.4f}, not an invoice.']
    (args.out/"reports/summary.md").write_text('\n'.join(lines)+'\n')
    (args.out/"reports/jev_all.json").write_text(json.dumps(dict(pairs=reports,splits=splits,
        upstream_cache_sha256=upstream_hashes,source_sha256=ar.sources,new_input_tokens=new_tokens_total,estimated_new_usd=new_tokens_total*.042/1e6),indent=2)+'\n')


if __name__=='__main__': main()
