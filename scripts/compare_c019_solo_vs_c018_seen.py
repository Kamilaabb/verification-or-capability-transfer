#!/usr/bin/env python3
"""Pair c017 G, c019 Qwen3-8B solo and c018 Qwen3-8B seeing G on math dev300.

Read-only over three archives; nothing is written into them. Every source is graded from
its own text through the fixed path (430d65c): G and the solo run from raw output, the
verifier from its FINAL ANSWER. For each pair it reports both correct, both wrong and
only-one-correct, and for the verifier it reports whether its answer equals G's or its
own solo answer, to show how much seeing G moves it.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c002_dev30 as base  # noqa: E402

C017 = "20260922T193921Z__c017__pilot300__r009_mathscalars"
C018 = "20260923T192748Z__c018__pilot300__r010_vscalars"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["problem_id"]: row for row in rows}


def pair_table(name_a: str, a: dict[str, int | None], name_b: str,
               b: dict[str, int | None]) -> dict:
    ids = [p for p in a if a[p] is not None and b.get(p) is not None]
    cells = collections.Counter((a[p], b[p]) for p in ids)
    n = len(ids)
    table = {"A": name_a, "B": name_b, "n": n,
             "both_correct": cells[(1, 1)], "both_wrong": cells[(0, 0)],
             f"only_{name_a}": cells[(1, 0)], f"only_{name_b}": cells[(0, 1)]}
    table["disagreement_rate"] = (cells[(1, 0)] + cells[(0, 1)]) / n if n else None
    return table


def equivalent(left: str | None, right: str | None) -> bool | None:
    if left is None or right is None:
        return None
    return base.math_equivalent(left, right)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--c019-run", required=True, help="c019 archive directory name")
    parser.add_argument("--out", help="optional JSON output path")
    args = parser.parse_args()
    art = Path(args.artifacts)

    g = by_id(read_jsonl(art / C017 / "records/pass1.jsonl"))
    solo = by_id(read_jsonl(art / args.c019_run / "records/pass1.jsonl"))
    seen = by_id(read_jsonl(art / C018 / "records/v.jsonl"))
    ids = sorted(g)
    if set(ids) != set(solo) or set(ids) != set(seen) or len(ids) != 300:
        raise SystemExit("the three archives do not cover the same 300 problems")
    sources = base.load_source_rows(ids)
    base.audit_gold_answers(sources)

    def grade(text: str | None, pid: str) -> int | None:
        return base.grade_prediction(sources[pid]["gold"], text)[0]

    u_g = {p: grade(g[p]["outputs"][0]["raw_output"], p) for p in ids}
    u_solo = {p: grade(solo[p]["outputs"][0]["raw_output"], p) for p in ids}
    u_seen = {p: (None if seen[p]["v_report"] is None
                  else grade(seen[p]["v_report"]["final_answer"], p)) for p in ids}

    accuracy = {}
    for name, u in (("G_qwen3_4b", u_g), ("V_solo", u_solo), ("V_seen_G", u_seen)):
        graded = [v for v in u.values() if v is not None]
        accuracy[name] = {"correct": sum(graded), "graded": len(graded),
                          "missing": len(ids) - len(graded)}

    tables = [pair_table("G", u_g, "V_solo", u_solo),
              pair_table("G", u_g, "V_seen", u_seen),
              pair_table("V_solo", u_solo, "V_seen", u_seen)]

    # Where does the verifier's answer land when it has seen G?
    anchoring = collections.Counter()
    for p in ids:
        report = seen[p]["v_report"]
        if report is None:
            anchoring["no_report"] += 1
            continue
        v_text = report["final_answer"]
        g_text = base.answer_text(g[p]["outputs"][0])
        s_text = base.answer_text(solo[p]["outputs"][0])
        same_g, same_solo = equivalent(v_text, g_text), equivalent(v_text, s_text)
        solo_eq_g = equivalent(s_text, g_text)
        anchoring[(f"solo==G:{solo_eq_g}", f"seen==G:{same_g}", f"seen==solo:{same_solo}",
                   f"verdict:{report['verdict']}")] += 1

    print("accuracy (raw-text grading)")
    for name, row in accuracy.items():
        print(f"  {name:11s} {row['correct']}/{row['graded']}  missing {row['missing']}")
    print("\npaired outcomes")
    for t in tables:
        keys = [k for k in t if k.startswith("only_")]
        print(f"  {t['A']} vs {t['B']}  n={t['n']}  both correct {t['both_correct']}  "
              f"both wrong {t['both_wrong']}  {keys[0]} {t[keys[0]]}  {keys[1]} {t[keys[1]]}  "
              f"disagree {t['disagreement_rate']:.1%}")
    print("\nverifier answer after seeing G, against G and against its own solo answer")
    for key in sorted(anchoring, key=str):
        print(f"  {key}: {anchoring[key]}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"accuracy": accuracy, "pairs": tables,
             "anchoring": {str(k): v for k, v in anchoring.items()}}, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
