"""For each thinking direction, where review loses against the reviewer alone (by which model
is right alone), a paired bootstrap interval for review minus reviewer alone, and whether lost answers become G's."""
import json
import random
import sys
from pathlib import Path

ROOT = Path("/path/to/project")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import thinking_check_test500 as t  # noqa: E402

REC = Path("/path/to/scratch/authority-allocation/runs/20261010T_thinking_check_test500/records")
ART = ROOT / "artifacts"

t.review.configure("c236")
base = t.review.base
ids = [p for p in t.manifest_ids() if p not in t.review.solo.ungradeable_ids()]
gold = {p: r["gold"] for p, r in base.load_source_rows(ids).items()}


def grade(row, fallback=None):
    text = base.answer_text(row["outputs"][0])
    if text is None and fallback is not None:
        text = base.answer_text(fallback["outputs"][0])
    return int(base.grade_prediction(gold[row["problem_id"]], text)[0] or 0)


def ans(row):
    a = row["outputs"][0].get("extracted_answer")
    return a.strip() if isinstance(a, str) and a.strip() else None


def rows_of(path, key="problem_id"):
    return {str(r[key]): r for r in t.jsonl(path)}


think = {c: rows_of(REC / f"{c}.jsonl") for c in t.CALLS}
solo_rows = {c: rows_of(t.archive(ART, c) / "records/pass1.jsonl") for c in ("c092", "c068", "c097")}
solo_u = {c: {str(r["problem_id"]): int(r["u"] or 0) for r in t.jsonl(t.archive(ART, c) / "records/grades.jsonl")
              if r["outcome"] == "u0"} for c in ("c092", "c068", "c097")}
v_on = {p: grade(think["solo"][p]) for p in ids}
rng = random.Random(20261010)
out = {}
for call, (orig, g) in t.CALLS.items():
    if g is None:
        continue
    old = t.archive(ART, orig)
    old_rows = rows_of(old / "records/review.jsonl")
    old_u = {str(r["problem_id"]): int(r["u"] or 0) for r in t.jsonl(old / "records/grades.jsonl") if r["outcome"] == "u_gv"}
    new_u = {p: grade(think[call][p], solo_rows[g][p]) for p in ids}
    for mode, vu, vrows, ru, rrows in (("off", solo_u["c092"], solo_rows["c092"], old_u, old_rows),
                                       ("on", v_on, think["solo"], new_u, think[call])):
        gu = solo_u[g]
        cells = {"both": [], "only_v": [], "only_g": [], "neither": []}
        for p in ids:
            cells[{(1, 1): "both", (1, 0): "only_v", (0, 1): "only_g", (0, 0): "neither"}[(vu[p], gu[p])]].append(p)
        res = {"cells": {k: [len(v), sum(ru[p] for p in v)] for k, v in cells.items()}}
        lost_v = [p for p in cells["only_v"] if ru[p] == 0]
        res["only_v_lost"] = len(lost_v)
        res["only_v_lost_to_g_answer"] = sum(ans(rrows[p]) is not None and ans(rrows[p]) == ans(solo_rows[g][p])
                                             for p in lost_v)
        lost_both = [p for p in cells["both"] if ru[p] == 0]
        res["both_lost"] = len(lost_both)
        d = [ru[p] - vu[p] for p in ids]
        res["review_minus_v_pp"] = 100 * sum(d) / len(ids)
        boots = sorted(100 * sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
        res["ci95_pp"] = [boots[249], boots[9749]]
        out[f"{call}_{mode}"] = res
print(json.dumps(out, indent=1))
