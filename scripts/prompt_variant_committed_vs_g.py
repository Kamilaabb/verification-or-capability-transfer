"""When the commit-first reviewer's committed answer differs from its own solo answer, how
often does it equal G's answer? Reads the same records as scripts/prompt_variant_test_subset.py."""
import json
import sys
from pathlib import Path

ROOT = Path("/path/to/project")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import prompt_variant_test_subset as pv  # noqa: E402

RUNS = Path("/path/to/scratch/authority-allocation/runs")
DB = Path("/path/to/scratch/results_db/20261007/db/directions.jsonl")
dirs = {}
for line in DB.read_text().splitlines():
    r = json.loads(line)
    if r["split"] == "test":
        dirs.setdefault(r["condition"], {})[str(r["problem_id"])] = r

out = {}
for cid, (domain, g_model, reviewer, orig, runner, manifest) in pv.VARIANT.items():
    run = [p for p in RUNS.glob(f"*__{cid}__*") if p.is_dir() and not p.name.endswith("__shards")][0]
    ids = [str(json.loads(l)["problem_id"]) for l in (ROOT / manifest).read_text().splitlines()][:pv.N]
    rows = {str(r["problem_id"]): r for r in pv.jsonl(run / "records/review.jsonl")[:pv.N]}
    g_rows = {str(r["problem_id"]): r for r in pv.jsonl(run / "records/g_pass1.jsonl")}
    d = dirs[orig]
    v_cond = {d[i]["v_condition"] for i in ids}.pop()
    v_rows = {str(r["problem_id"]): r for r in pv.jsonl(pv.archive(ROOT / "artifacts", v_cond) / "records/pass1.jsonl")}
    reader = (pv.MathCommitted if domain == "math" else pv.McCommitted)(ids, cid)
    res = {"present": 0, "differs_from_solo": 0, "differs_equals_g": 0, "differs_equals_g_and_g_wrong": 0,
           "differs_right": 0, "solo_differs_from_g": 0}
    for p in ids:
        c, c_right = reader.read(p, pv.committed_text(rows[p]["outputs"][0]["raw_output"]))
        v, g = pv.answer(v_rows[p]), pv.answer(g_rows[p])
        res["solo_differs_from_g"] += int(v is not None and g is not None and v != g)
        if c is None:
            continue
        res["present"] += 1
        if v is not None and c == v:
            continue
        res["differs_from_solo"] += 1
        res["differs_right"] += c_right
        if g is not None and c == g:
            res["differs_equals_g"] += 1
            res["differs_equals_g_and_g_wrong"] += int(d[p]["g"] == 0)
    out[f"{cid} {domain} {reviewer} reviews {g_model}"] = res
print(json.dumps(out, indent=1))
