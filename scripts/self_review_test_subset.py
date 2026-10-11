#!/usr/bin/env python3
"""Self-review on test1500, first 500 questions (c386-c389; docs/plan/self_review_test1500.md). Server only.

    python scripts/self_review_test_subset.py --db /path/to/scratch/results_db/20261007 \
        --runs /path/to/scratch/authority-allocation/runs --out /path/to/scratch/claims/<new dir>

The four self-review runs were stopped after row 500 and are not archived. For each, this reads the
first 500 review rows from the run directory, checks they are the first 500 manifest rows in order,
and grades them with the condition's own runner (`grade` stage, into the output directory). The
cross reviews come from the results database (directions.jsonl, built from the archives).

For every design direction on test1500 whose reviewer has a self-review, on the same 500 questions:
on the questions the reviewer gets right alone and G gets wrong, how many are still right after
self-review and after reviewing G, the same on all
questions the reviewer gets right alone; and how often self-review keeps the reviewer's own answer.
Counts only; no fit, no threshold. Writes a new directory and never overwrites.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 500
SELF = {  # condition: (domain, reviewer as in the results database, runner, manifest, solo run reviewed)
    "c386": ("math", "Qwen3-8B", "run_mathfull_review.py", "data/manifests/mathfull-r465bcdb-l3to5-s20260924/test1500.jsonl", "c092"),
    "c387": ("math", "Qwen3-4B", "run_mathfull_review.py", "data/manifests/mathfull-r465bcdb-l3to5-s20260924/test1500.jsonl", "c062"),
    "c388": ("knowledge", "Qwen3-8B", "run_mmlu_pro.py", "data/manifests/mmlupro-rb189ec7-s20260927/test1500.jsonl", "c082"),
    "c389": ("knowledge", "Qwen2.5-7B-Instruct", "run_mmlu_pro.py", "data/manifests/mmlupro-rb189ec7-s20260927/test1500.jsonl", "c122"),
}


def jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a row cut by the time limit
            break
    return out


def answer(row: dict):
    a = row["outputs"][0].get("extracted_answer")
    return a.strip() if isinstance(a, str) and a.strip() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reports are never overwritten")
    (args.out / "graded").mkdir(parents=True)

    selfs = {}
    for cid, (domain, reviewer, runner, manifest, _solo) in SELF.items():
        runs = [p for p in args.runs.glob(f"*__{cid}__*") if p.is_dir() and not p.name.endswith("__shards")]
        if len(runs) != 1:
            raise SystemExit(f"{cid}: expected one run directory, found {runs}")
        run = runs[0]
        ids = [str(json.loads(l)["problem_id"]) for l in (ROOT / manifest).read_text().splitlines()][:N]
        rows = jsonl(run / "records/review.jsonl")[:N]
        if [str(r["problem_id"]) for r in rows] != ids:
            raise SystemExit(f"{cid}: the first {N} review rows are not the first {N} manifest rows")
        own = {str(r["problem_id"]): r for r in jsonl(run / "records/g_pass1.jsonl")}
        sub = args.out / "graded" / cid
        sub.mkdir()
        (sub / "review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (sub / "g_pass1.jsonl").write_text("".join(json.dumps(own[p]) + "\n" for p in ids), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / runner), "--condition-id", cid, "grade",
               "--g-records", str(sub / "g_pass1.jsonl"), "--output", str(sub / "grades.jsonl")]
        cmd += (["--review-records", str(sub / "review.jsonl")] if domain == "math"
                else ["--records", str(sub / "review.jsonl")])
        subprocess.run(cmd, check=True, cwd=ROOT)
        gv = {str(g["problem_id"]): g["u"] for g in jsonl(sub / "grades.jsonl") if g["outcome"] == "u_gv"}
        kept_own = {p: int(answer(r) is not None and answer(r) == answer(own[p])) for p, r in zip(ids, rows)}
        hw = (run / "logs/execution_hardware.txt").read_text().strip().splitlines()
        selfs[cid] = (domain, reviewer, ids, gv, kept_own, run.name, hw)

    dirs = defaultdict(dict)  # (domain, G, V) -> pid -> row, test split only
    for line in (args.db / "db/directions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] == "test" and r["domain"] in ("math", "knowledge"):
            dirs[(r["domain"], r["generator"], r["reviewer"])][str(r["problem_id"])] = r
    pairs = json.loads((args.db / "db/pairs.json").read_text())

    rows, skipped, lines = [], [], [
        "# Self-review on test1500, first 500 questions (c386-c389)", "",
        "Same reviewer, same questions, same frozen review prompt; only whose answer is shown differs.", "",
        "|domain|reviewer|reviewed G|reviewer is|only reviewer right|kept, self-review|kept, reviewing G|"
        "reviewer right|kept, self-review|kept, reviewing G|", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.1f}%)" if b else "—"
    for cid, (domain, reviewer, ids, gv_self, kept_own, run_name, _hw) in selfs.items():
        for p in pairs[domain]:
            if not p["in_design_sample"] or reviewer not in p["models"]:
                continue
            g_model = next(m for m in p["models"] if m != reviewer)
            d = dirs[(domain, g_model, reviewer)]
            missing = [i for i in ids if i not in d]
            if missing:
                raise SystemExit(f"{domain} {reviewer} reviews {g_model}: {len(missing)} test rows missing")
            # The reviewer's solo answers in this direction must be the ones the self-review was shown.
            if {d[i]["v_condition"] for i in ids} != {SELF[cid][4]}:
                skipped.append(f"{domain} {reviewer} reviewing {g_model} ({d[ids[0]]['condition']}): reviewer's solo "
                               f"answers from {d[ids[0]]['v_condition']}, self-review shown {SELF[cid][4]}")
                continue
            keys = [i for i in ids if gv_self[i] is not None and None not in (d[i]["g"], d[i]["v"], d[i]["gv"])]
            only_v = [k for k in keys if d[k]["v"] == 1 and d[k]["g"] == 0]
            v_right = [k for k in keys if d[k]["v"] == 1]
            r = {"self_condition": cid, "domain": domain, "reviewer": reviewer, "g": g_model,
                 "cross_condition": d[keys[0]]["condition"],
                 "kind": "stronger reviews" if p["stronger"] == reviewer else "weaker reviews", "n": len(keys),
                 "only_v": len(only_v), "only_v_kept_self": sum(gv_self[k] for k in only_v),
                 "only_v_kept_cross": sum(d[k]["gv"] for k in only_v),
                 "v_right": len(v_right), "v_right_kept_self": sum(gv_self[k] for k in v_right),
                 "v_right_kept_cross": sum(d[k]["gv"] for k in v_right),
                 "acc_v": sum(d[k]["v"] for k in keys), "acc_self": sum(gv_self[k] for k in keys),
                 "acc_g": sum(d[k]["g"] for k in keys), "acc_cross": sum(d[k]["gv"] for k in keys),
                 "self_keeps_own_answer": sum(kept_own[k] for k in keys)}
            rows.append(r)
            lines.append(f"|{domain}|{reviewer}|{g_model}|{r['kind'].split()[0]}|{r['only_v']}|"
                         f"{pct(r['only_v_kept_self'], r['only_v'])}|{pct(r['only_v_kept_cross'], r['only_v'])}|"
                         f"{r['v_right']}|{pct(r['v_right_kept_self'], r['v_right'])}|"
                         f"{pct(r['v_right_kept_cross'], r['v_right'])}|")
    lines += ["", "Not compared (the direction uses another solo run of the reviewer than the one self-reviewed):", ""]
    lines += [f"- {s}" for s in skipped] or ["- none"]
    lines += ["", "Correct out of the questions used (reviewer alone / self-review / G alone / reviewer reviewing G):", ""]
    lines += [f"- {r['domain']} {r['reviewer']} reviewing {r['g']} ({r['cross_condition']}): "
              f"{r['acc_v']} / {r['acc_self']} / {r['acc_g']} / {r['acc_cross']} of {r['n']}; "
              f"self-review keeps its own extracted answer on {r['self_keeps_own_answer']}" for r in rows]
    meta = {cid: {"run": s[5], "rows_graded": len(s[2]), "hardware": s[6]} for cid, s in selfs.items()}
    lines += ["", "Hardware (the cross reviews ran on H100):", ""] + [f"- {cid}: {m['hardware']}" for cid, m in meta.items()]
    (args.out / "reports").mkdir()
    (args.out / "reports/result.json").write_text(json.dumps({"rows": rows, "skipped": skipped, "self_runs": meta}, indent=1) + "\n")
    (args.out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
