#!/usr/bin/env python3
"""Anchoring in the BigCodeBench GV directions (docs/analysis/code_anchoring_dev832.md).

Pure arithmetic over archived grades and outputs; no model, no fit. For each direction it compares
the reviewed model's solo result with (a) the reviewer's own solo result and (b) GV, and on the rows
where the two solo results differed in correctness it counts whose correctness GV follows. It also
counts final programs identical (whitespace-normalized last fenced Python block) to the reviewed one.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"```[ \t]*(?:python3?|py)[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
DIRECTIONS = [  # (GV condition, reviewer's solo condition, reviewed solo condition, label)
    ("c111", "c100", "c105", "Qwen3-4B reviews Ministral-8B"),
    ("c110", "c105", "c100", "Ministral-8B reviews Qwen3-4B"),
    ("c114", "c100", "c101", "Qwen3-4B reviews Qwen3-8B"),
    ("c112", "c101", "c100", "Qwen3-8B reviews Qwen3-4B"),
]


def archive(artifacts: Path, cid: str) -> Path:
    hits = sorted(h for h in glob.glob(str(artifacts / f"*__{cid}__bcb-dev832__*"))
                  if (Path(h) / "checksums.sha256").exists())
    return Path(hits[-1])


def grades(run: Path, outcome: str) -> dict[str, int]:
    rows = (json.loads(line) for line in (run / "records/grades.jsonl").read_text(encoding="utf-8").splitlines())
    return {r["problem_id"]: int(r["u"]) for r in rows if r["outcome"] == outcome}


def programs(run: Path, name: str) -> dict[str, str | None]:
    out = {}
    for line in (run / "records" / name).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        blocks = BLOCK.findall(r["outputs"][0]["raw_output"] or "")
        out[r["problem_id"]] = re.sub(r"\s+", " ", blocks[-1]).strip() if blocks else None
    return out


def table(a: dict, b: dict, ids: list[str]) -> dict[str, int]:
    both = sum(a[p] and b[p] for p in ids)
    only_a = sum(a[p] and not b[p] for p in ids)
    only_b = sum(b[p] and not a[p] for p in ids)
    return {"both": both, "only_reviewed": only_a, "only_other": only_b,
            "neither": len(ids) - both - only_a - only_b, "disagree": only_a + only_b,
            "union": both + only_a + only_b}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    for gv_cid, reviewer_cid, reviewed_cid, label in DIRECTIONS:
        gv_run, rv_run, rd_run = (archive(artifacts, c) for c in (gv_cid, reviewer_cid, reviewed_cid))
        reviewer, reviewed, gv = grades(rv_run, "u0"), grades(rd_run, "u0"), grades(gv_run, "u_gv")
        ids = sorted(gv)
        differ = [p for p in ids if reviewed[p] != reviewer[p]]
        reviewer_only = [p for p in ids if reviewer[p] and not reviewed[p]]
        reviewed_only = [p for p in ids if reviewed[p] and not reviewer[p]]
        prog_rd, prog_rv, prog_gv = (programs(rd_run, "pass1.jsonl"), programs(rv_run, "pass1.jsonl"),
                                     programs(gv_run, "review.jsonl"))
        result = {
            "direction": f"{gv_cid}: {label}", "rows": len(ids),
            "reviewed_alone": sum(reviewed.values()), "reviewer_alone": sum(reviewer.values()), "gv": sum(gv[p] for p in ids),
            "independent": table(reviewed, reviewer, ids), "after_review": table(reviewed, gv, ids),
            "differ_rows": len(differ),
            "gv_follows_reviewed": sum(gv[p] == reviewed[p] for p in differ),
            "gv_keeps_reviewer_alone": sum(gv[p] == reviewer[p] for p in differ),
            "reviewer_only_right": len(reviewer_only), "reviewer_only_right_kept": sum(gv[p] for p in reviewer_only),
            "reviewed_only_right": len(reviewed_only), "reviewed_only_right_taken": sum(gv[p] for p in reviewed_only),
            "identical_program_alone": sum(prog_rd[p] is not None and prog_rd[p] == prog_rv[p] for p in ids),
            "identical_program_after": sum(prog_rd[p] is not None and prog_rd[p] == prog_gv[p] for p in ids),
        }
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
