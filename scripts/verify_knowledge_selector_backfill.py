"""Verify backfilled test decisions against the immutable thirteen-pair summary.

Run on the server. Reads archived grades, never problems or manifests. Does not fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from gv_domain_tables import Archives


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def conditions(report):
    return tuple(Path(r).name.split("__")[1] for r in report["runs"][:2])


def compare_report(old, new):
    if new["domain"] != "knowledge" or new["n"] != 1500:
        raise ValueError("Expected knowledge test1500")
    if conditions(new) != (old["g"][1], old["v"][1]):
        raise ValueError("Upstream model roles changed")
    for field, expected in (("G", old["g"][3]), ("V", old["v"][3]), ("oracle", old["oracle"])):
        if new[field] != expected:
            raise ValueError(f"{old['pair']}: {field} mismatch")
    for kind in ("primary", "secondary"):
        actual, expected = new["selectors"][kind]["correct"], old["selectors"][kind]["correct"]
        if actual != expected:
            raise ValueError(f"{old['pair']} {kind}: correct {actual} != archived {expected}")


def validate_decisions(path, yg, yv):
    seen, correct = set(), 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if set(r) != {"problem_id", "p_g", "p_v", "pick_v"}:
                raise ValueError("Decision schema changed or contains labels")
            pid = r["problem_id"]
            if pid in seen or pid not in yg or pid not in yv:
                raise ValueError("Duplicate or unexpected problem ID")
            seen.add(pid)
            for field in ("p_g", "p_v"):
                if not isinstance(r[field], (int, float)) or not math.isfinite(r[field]) or not 0 <= r[field] <= 1:
                    raise ValueError("Invalid probability")
            if type(r["pick_v"]) is not bool or r["pick_v"] != (r["p_v"] >= r["p_g"]):
                raise ValueError("Tie/delivery rule changed")
            correct += yv[pid] if r["pick_v"] else yg[pid]
    if seen != set(yg) or seen != set(yv) or len(seen) != 1500:
        raise ValueError("Decisions do not cover the same 1500 upstream rows")
    return correct


def verify(old_path, new_root):
    ar = Archives(ROOT / "artifacts")
    ar.checked(old_path.parent.parent, "reports/" + old_path.name)
    olds = read_json(old_path)
    if len(olds) != 13:
        raise ValueError("Expected archived thirteen pairs")
    expected = {(r["g"][1], r["v"][1]): r for r in olds}
    rows, seen = [], set()
    reports = sorted(new_root.glob("knowledge__*/reports/selector.json"))
    for path in reports:
        new = read_json(path)
        pair = conditions(new)
        if pair not in expected or pair in seen:
            raise ValueError("Unexpected or repeated pair")
        old = expected[pair]
        compare_report(old, new)
        runs = [ROOT / "artifacts" / Path(r).name for r in new["runs"]]
        if tuple(r.name.split("__")[1] for r in runs[2:]) != (old["g"][2], old["v"][2]):
            raise ValueError("Test source conditions changed")
        yg, yv = ar.grades(runs[2]), ar.grades(runs[3])
        files = {}
        for kind in ("primary", "secondary"):
            decision = path.parent.parent / "records" / f"{kind}_decisions.jsonl"
            correct = validate_decisions(decision, yg, yv)
            if correct != new["selectors"][kind]["correct"]:
                raise ValueError("Recorded decisions disagree with summary")
            files[kind] = {"path": str(decision), "sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
                           "correct": correct, "old_correct": old["selectors"][kind]["correct"]}
        if not (path.parent / "summary.md").is_file():
            raise ValueError("Missing per-pair human report")
        seen.add(pair)
        rows.append({"pair": old["pair"], "n": 1500, "files": files, "runs": new["runs"],
                     "source_sha256": new["source_sha256"]})
    if seen != set(expected):
        raise ValueError(f"Incomplete batch: {len(seen)}/13 pairs")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit("Validation output exists; use a new directory")
    args.out_dir.mkdir(parents=True)
    reports = args.out_dir / "reports"
    reports.mkdir()
    try:
        rows = verify(args.old_summary, args.new_root)
    except Exception as exc:
        (reports / "summary.md").write_text(f"# Knowledge selector backfill FAILED\n\n{exc}\n\nStop: no count alignment or selector changes are permitted.\n", encoding="utf-8")
        raise
    result = {"status": "passed", "pairs": rows,
              "old_summary": str(args.old_summary),
              "old_summary_sha256": hashlib.sha256(args.old_summary.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (reports / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Knowledge selector backfill: PASSED", "",
             "13 test1500 pairs; primary and secondary counts exactly reproduce the old summary.",
             "All 26 decision files pass schema, unique membership, finite probability, V tie-break and delivery checks.",
             "No inference, prompt changes, hyperparameter changes or new research condition.", "",
             "|pair|primary old/new|secondary old/new|", "|---|---:|---:|"]
    for r in rows:
        a, b = r["files"]["primary"], r["files"]["secondary"]
        lines.append(f"|{r['pair']}|{a['old_correct']}/{a['correct']}|{b['old_correct']}/{b['correct']}|")
    (reports / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("PASSED: 13 pairs, 26 decision files; exact old/new counts")


if __name__ == "__main__":
    main()
