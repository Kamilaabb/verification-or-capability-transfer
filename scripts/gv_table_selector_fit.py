#!/usr/bin/env python3
"""Frozen pair selectors on test1500; math trains on testpool_dev1000 only.

One array task per pair. No parameter or model choice uses test outcomes.
The existing thirteen knowledge-pair results can be reused by the report;
--missing-only skips those pairs. Primary/secondary definitions are unchanged.
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import numpy as np
import selector_dev1000 as sel
import test_mathfull_v2_analysis as mathsel
import mc_pair_selector_dev1000 as mc
import mc_pair_selector_test1500 as mcsel
import mc_pair_selector_test_pairs as existing_mc
from gv_domain_tables import MODELS, Archives, paired


def tasks(missing_only):
    existing = {frozenset((p[1][1],p[2][1])) for p in existing_mc.PAIRS}
    return [(domain,a,b) for domain,models in MODELS.items() for a,b in itertools.combinations(models,2)
            if not (missing_only and domain == "knowledge" and frozenset((a,b)) in existing)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index",type=int,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--missing-only",action="store_true")
    args = ap.parse_args()
    domain,a,b = tasks(args.missing_only)[args.index]
    ar = Archives(ROOT/"artifacts")
    dr = {c:ar.find(c) for c in (a,b)}
    a,b = sorted((a,b),key=lambda c:(sum(ar.grades(dr[c]).values()),c))
    tg,tv = MODELS[domain][a],MODELS[domain][b]
    out = args.out_dir/f"{domain}__{a}__{b}"
    if (out/"reports/summary.md").exists():
        print("Already complete",out); return
    (out/"records").mkdir(parents=True,exist_ok=True); (out/"reports").mkdir(exist_ok=True)
    runs = [dr[a],dr[b],ar.find(tg),ar.find(tv)]
    for run in runs:
        for name in ("condition.resolved.yaml","records/grades.jsonl","records/pass1.jsonl",
                     "records/native_logit_trace.jsonl","records/scalar_evidence.jsonl"):
            ar.checked(run,name)
    if domain == "math":
        manifest_root = ROOT/"data/manifests/mathfull-r465bcdb-l3to5-s20260924"
        def build(g,v,split):
            sel.G_RUN,sel.V_RUN = g.name,v.name
            return sel.build(ROOT/"artifacts",manifest_root/f"{split}.jsonl",set())
        ids_d,rows_d,yg_d,yv_d = build(runs[0],runs[1],"testpool_dev1000")
        ids_t,rows_t,yg,yv = build(runs[2],runs[3],"test1500")
        fs_d,fs_t = sel.feature_sets(rows_d),sel.feature_sets(rows_t)
        dev,test = (rows_d,ids_d,yg_d,yv_d,fs_d),(rows_t,ids_t,fs_t)
        def fit(members): return mathsel.fit_predict(dev,test,members)
        configs = [("primary",mathsel.PRIMARY),("secondary",mathsel.SECONDARY)]
    else:
        ids_d,rows_d,yg_d,yv_d = mc.build(ROOT/"artifacts",a,b)
        ids_t,rows_t,yg,yv = mc.build(ROOT/"artifacts",tg,tv,existing_mc.TEST_MANIFEST,existing_mc.TEST_TAG)
        dev,test = (rows_d,ids_d,yg_d,yv_d),(rows_t,ids_t,yg,yv)
        def fit(members): return mcsel.fit_predict(dev,test,mc.feature_sets(),members)[:2]
        configs = list(mc.MEMBERS.items())
    if set(ids_d)&set(ids_t): raise ValueError("training/test overlap")
    if len(ids_d)!=1000 or len(ids_t)!=1500: raise ValueError("unexpected membership")
    if sum(yg_d)>sum(yv_d): raise ValueError("baseline2 not selected on development")
    report = dict(domain=domain,g=ar.config(runs[0])["model"]["model_id"],
                  v=ar.config(runs[1])["model"]["model_id"],runs=[r.name for r in runs],
                  n=len(ids_t),dev_G=int(sum(yg_d)),dev_V=int(sum(yv_d)),
                  G=int(sum(yg)),V=int(sum(yv)),oracle=int(np.maximum(yg,yv).sum()),selectors={})
    for name,members in configs:
        pg,pv = fit(members)
        # Explicit decisions contain no evaluation labels.
        with (out/f"records/{name}_decisions.jsonl").open("w") as f:
            for pid,gp,vp in zip(ids_t,pg,pv):
                f.write(json.dumps(dict(problem_id=pid,p_g=float(gp),p_v=float(vp),pick_v=bool(vp>=gp)))+"\n")
        yy = np.where(pg>pv,yg,yv).astype(int).tolist()
        report["selectors"][name] = dict(correct=sum(yy),vs_baseline2=paired(yy,yv.tolist()))
    report["source_sha256"] = ar.sources
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/"reports/selector.json").write_text(json.dumps(report,indent=2)+"\n")
    lines = [f"# {domain}: {report['g']} / {report['v']}","",
             "Fit only on the registered 1000 development rows; frozen feature sets, models, grids, five seeds, probability averaging and V tie-break. Test 1500 used only for evaluation.","",
             f"G {report['G']}; baseline2 V {report['V']}; independent oracle {report['oracle']}.","",
             "|selector|correct|minus B2|recovered headroom|","|---|---:|---:|---:|"]
    h=report['oracle']-report['V']
    for k,r in report['selectors'].items():
        delta=r['correct']-report['V']
        lines.append(f"|{k}|{r['correct']}|{delta:+d}|{delta/h:.1%}|")
    (out/"reports/summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__ == "__main__": main()
