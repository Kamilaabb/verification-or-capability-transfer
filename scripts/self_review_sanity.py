#!/usr/bin/env python3
"""Self-review as a control for the reviewer following G (paper appendix, Further checks). Server only.

    python scripts/self_review_sanity.py --artifacts /path/to/project/artifacts --out <new dir>

Reads finished archives only (2026-09-26, mathfull-dev1000, frozen review prompt v2, docs/plan/gv_vs_max_g_v_study.md):
the solo runs c026 (Qwen3-8B), c032 (Ministral-8B), c029 (Llama-3.2-3B); the self-reviews c051 (Qwen3-8B
reviews c026) and c050 (Ministral-8B reviews c032); and the cross reviews c052 (Qwen3-8B reviews c032),
c048 (Qwen3-8B reviews c029) and c053 (Ministral-8B reviews c026). Same questions, same prompt, same
reviewer: only whose answer is shown differs.

For each cross review, on the questions the reviewer answers correctly alone and G gets wrong, and on all
questions the reviewer answers correctly alone: how often the final answer is right after self-review and
after reviewing G, and how often a final answer that turned wrong is G's answer (same extracted answer).
Rows with any missing grade are left out. Only counts leave the server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from gv_domain_tables import Archives  # noqa: E402

SOLO = {"Qwen3-8B": "c026", "Ministral-8B": "c032", "Llama-3.2-3B": "c029"}
SELF = {"Qwen3-8B": "c051", "Ministral-8B": "c050"}
CROSS = [("Qwen3-8B", "Ministral-8B", "c052"), ("Qwen3-8B", "Llama-3.2-3B", "c048"),
         ("Ministral-8B", "Qwen3-8B", "c053")]  # (reviewer, G, condition)


def grades(ar: Archives, cid: str, outcome: str) -> dict:
    run = ar.find(cid)
    out = {}
    for line in ar.checked(run, "records/grades.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["outcome"] == outcome:
            out[str(r["problem_id"])] = r["u"] if r["u"] in (0, 1) else None
    return out


def answers(ar: Archives, cid: str, rel: str) -> dict:
    run = ar.find(cid)
    out = {}
    for line in ar.checked(run, rel).read_text().splitlines():
        r = json.loads(line)
        a = r["outputs"][0].get("extracted_answer")
        out[str(r["problem_id"])] = a.strip() if isinstance(a, str) and a.strip() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reports are never overwritten")
    ar = Archives(args.artifacts)
    solo = {m: grades(ar, c, "u0") for m, c in SOLO.items()}
    rows, lines = [], ["# 自审对照：同一个审阅者、同一批题、同一个提示词，只换被审的答案", "",
                       "数据：mathfull-dev1000（2026-09-26 的旧开发集，MATH 3–5 级），冻结的 review 提示词 v2。", "",
                       "|审阅者|被审者|题数|只有审阅者对的题|自审后仍对|审别人后仍对|变错的里面变成 G 的答案|"
                       "审阅者对的全部题|自审后仍对|审别人后仍对|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v, g, cid in CROSS:
        self_gv = grades(ar, SELF[v], "u_gv")
        cross_gv = grades(ar, cid, "u_gv")
        final = answers(ar, cid, "records/review.jsonl")
        g_ans = answers(ar, cid, "records/g_pass1.jsonl")
        keys = [k for k in solo[v] if all(d.get(k) is not None for d in (solo[v], solo[g], self_gv, cross_gv))]
        only_v = [k for k in keys if solo[v][k] == 1 and solo[g][k] == 0]
        v_right = [k for k in keys if solo[v][k] == 1]
        lost = [k for k in only_v if cross_gv[k] == 0]
        to_g = sum(1 for k in lost if final.get(k) is not None and final.get(k) == g_ans.get(k))
        r = {"reviewer": v, "g": g, "condition": cid, "self_condition": SELF[v], "n": len(keys),
             "only_v": len(only_v), "only_v_kept_self": sum(self_gv[k] for k in only_v),
             "only_v_kept_cross": sum(cross_gv[k] for k in only_v), "only_v_lost_cross": len(lost),
             "only_v_lost_to_g": to_g, "v_right": len(v_right), "v_right_kept_self": sum(self_gv[k] for k in v_right),
             "v_right_kept_cross": sum(cross_gv[k] for k in v_right),
             "acc_v": sum(solo[v][k] for k in keys), "acc_self": sum(self_gv[k] for k in keys),
             "acc_cross": sum(cross_gv[k] for k in keys)}
        rows.append(r)
        pct = lambda a, b: f"{a}/{b} ({100 * a / b:.1f}%)" if b else "—"
        lines.append(f"|{v}|{g}|{r['n']}|{r['only_v']}|{pct(r['only_v_kept_self'], r['only_v'])}|"
                     f"{pct(r['only_v_kept_cross'], r['only_v'])}|{pct(r['only_v_lost_to_g'], r['only_v_lost_cross'])}|"
                     f"{r['v_right']}|{pct(r['v_right_kept_self'], r['v_right'])}|{pct(r['v_right_kept_cross'], r['v_right'])}|")
    lines += ["", "整体正确数（同一批题）：", "", "|审阅者|被审者|审阅者单独|自审|审别人|", "|---|---|---:|---:|---:|"]
    lines += [f"|{r['reviewer']}|{r['g']}|{r['acc_v']}|{r['acc_self']}|{r['acc_cross']}|" for r in rows]
    (args.out / "reports").mkdir(parents=True)
    (args.out / "reports/result.json").write_text(json.dumps({"rows": rows, "sources": ar.sources}, indent=1) + "\n")
    (args.out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
