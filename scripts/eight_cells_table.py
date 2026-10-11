"""Eight cells for the paper's 96 review directions: questions split by whether G alone, V alone and the final answer
are right, as a share of all questions. Averaged over the 12 directions of each kind within a domain, then over the
four domains equally, as in the paper. Math and knowledge pool development and test (2,500 questions); code and
reasoning use all their questions. Reads results_db/20261007 on the server and writes aggregates only.

The cells reproduce the three parts of the gain over G in Section 4.2: 011 is the V-only part, 100 the G-only part,
001 - 110 the same-outcome part."""
import json
import sys
from collections import defaultdict
from pathlib import Path

DB = Path(sys.argv[1] if len(sys.argv) > 1 else "/path/to/scratch/results_db/20261007/db")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
DOMAINS = ("math", "knowledge", "code", "nl_reasoning")
CELLS = [f"{g}{v}{f}" for g in "10" for v in "10" for f in "10"]

pairs = json.loads((DB / "pairs.json").read_text())
direction = {}  # condition -> (domain, kind, pair, G)
for dom in DOMAINS:
    chosen = [p for p in pairs[dom] if p["in_design_sample"]]
    assert len(chosen) == 12, (dom, len(chosen))
    for p in chosen:
        for gkey, conds in p["directions"].items():
            g = gkey.split("=", 1)[1]
            reviewer = next(m for m in p["models"] if m != g)
            kind = "stronger_reviews" if reviewer == p["stronger"] else "weaker_reviews"
            for cond in conds.values():
                direction[cond] = (dom, kind, tuple(p["models"]), g)

counts = defaultdict(lambda: defaultdict(int))
seen = defaultdict(set)
with (DB / "directions.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        key = direction.get(r["condition"])
        if key is None:
            continue
        assert key[0] == r["domain"]
        assert r["problem_id"] not in seen[key], key
        seen[key].add(r["problem_id"])
        counts[key][f"{r['g']}{r['v']}{r['gv']}"] += 1
assert len(counts) == 96, len(counts)

per = defaultdict(list)
for key, cnt in counts.items():
    n = sum(cnt.values())
    per[key[:2]].append({c: 100 * cnt[c] / n for c in CELLS} | {"n": n})

result = {"source": str(DB), "unit": "percent of all questions", "cells": "G alone, V alone, final (1 = right)"}
for kind in ("stronger_reviews", "weaker_reviews"):
    block = {}
    for dom in DOMAINS:
        rows = per[dom, kind]
        assert len(rows) == 12, (dom, kind, len(rows))
        block[dom] = {c: sum(r[c] for r in rows) / 12 for c in CELLS} | {"n": sorted({r["n"] for r in rows})}
    block["mean"] = {c: sum(block[d][c] for d in DOMAINS) / 4 for c in CELLS}
    m = block["mean"]
    block["check"] = {"v_only_part": m["011"], "g_only_part": -m["100"], "same_outcome_part": m["001"] - m["110"],
                      "delta_g": m["011"] - m["100"] + m["001"] - m["110"]}
    result[kind] = block

text = json.dumps(result, indent=1)
if OUT:
    OUT.write_text(text + "\n")
print(text)
