#!/usr/bin/env python3
"""Rerun H01-H11 and the RQ2 summaries with the audited intervals; compare with the old reports.

Statistics audit of 2026-10-05 (docs/analysis/stats_audit_20261005):
- each direction gets a paired question bootstrap;
- means keep the evaluated directions fixed, with one shared question resample per domain within
  fixed strata (split everywhere, dataset in reasoning);
- relations across pairs resample model pairs with both directions together;
- pair resampling of the means is reported as a sensitivity analysis.

Same cards, database, cost rows, seeds and replicate counts as the old reports. Server only:

    python scripts/rerun_claims_audited.py --db /path/to/scratch/results_db/20261004b \
        --cost /path/to/scratch/cost/20261004 --out /path/to/scratch/claims/20261005_audit

Writes a new directory, never overwrites, and copies only aggregate reports:
- H01-H11 results and summaries;
- rq2_cells.json and rq2_cells_audited.json;
- comparison.json;
- reports/summary.md, a before/after table of the audit.
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from authority_allocation.outcome_levels import outcome_level  # noqa: E402

MAIN = "主分析：设计配对"
OLD = ROOT / "docs/analysis/claims"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ci(c):
    return "—" if c is None else f"[{c[0]:+.2f}, {c[1]:+.2f}]"


def run_all(db: Path, cost: Path, out: Path) -> None:
    for i in range(1, 12):
        h = f"H{i:02d}"
        cards = list((ROOT / "claims").glob(f"{h}_*.yaml"))
        if len(cards) != 1:
            raise ValueError(f"{h}: expected one registered card")
        if h == "H09":
            continue  # reads H08's per-direction report; run after H08
        target = out / "runs" / h
        if h in ("H10", "H11"):
            subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_cost_claims.py"), "--cost", str(cost),
                            "--claim", str(cards[0]), "--out-dir", str(target)], check=True)
            for name in (f"{h}_result.json", f"{h}_summary.md"):
                shutil.copyfile(target / name, out / name)
            continue
        script = "explain_conditions.py" if h == "H08" else "evaluate_claim.py"
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--db", str(db), "--claim", str(cards[0]),
                        "--out-dir", str(target)], check=True)
        for name, dest in (("result.json", f"{h}_result.json"), ("summary.md", f"{h}_summary.md")):
            shutil.copyfile(target / "reports" / name, out / dest)
    dirs = load(out / "H08_result.json")["analyses"][MAIN]["directions"]
    counts = dict(Counter(d["domain"] for d in dirs))
    if counts != {d: 24 for d in ("math", "knowledge", "code", "nl_reasoning")}:
        raise ValueError(f"expected 24 design review directions per domain; found {counts}")
    subprocess.run([sys.executable, str(ROOT / "scripts/explain_outcome_levels.py"), "--h08",
                    str(out / "H08_result.json"), "--out-dir", str(out)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/rq2_cells.py"), str(out)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/rq2_cells_bootstrap.py"), "--db", str(db),
                    "--cells", str(out / "rq2_cells.json"), "--out", str(out / "rq2_cells_audited.json")], check=True)


def compare(out: Path, old_b: Path, old_c: Path, old_d: Path) -> None:
    rows, lines = [], ["# 统计审计后重跑：改前 / 改后（2026-10-05）", "",
                       "点估计不变，只换区间。改后：单个方向用逐题配对 bootstrap；平均值固定方向、每个域共用一次按题重抽"
                       "（开发/测试分层，推理按数据集分层）；回归按模型对重抽。"
                       "“按对重抽”一栏是附录敏感性分析。单位 pp。", ""]

    def add(section, name, old_point, old_ci, new_point, new_ci, pairs_ci=None):
        rows.append(dict(section=section, name=name, old_point=old_point, old_ci=old_ci,
                         new_point=new_point, new_ci=new_ci, pairs_ci=pairs_ci))

    for i in range(1, 8):
        h = f"H{i:02d}"
        old_path = old_b / f"{h}_result.json"
        if not old_path.exists():
            old_path = old_b.parent / "20261003b" / f"{h}_result.json"
        old, new = load(old_path)["analyses"], load(out / f"{h}_result.json")["analyses"]
        for analysis in new:
            if analysis not in old or new[analysis] is None or old[analysis] is None:
                continue
            ob, nb = old[analysis]["report"]["bootstrap"], new[analysis]["report"]["bootstrap"]
            npairs = new[analysis]["report"].get("bootstrap_pairs")
            for scope in ["overall", *nb["per_domain"]]:
                o = ob["overall"] if scope == "overall" else ob["per_domain"].get(scope)
                n = nb["overall"] if scope == "overall" else nb["per_domain"][scope]
                p = None if npairs is None else (npairs["overall"] if scope == "overall" else npairs["per_domain"][scope])
                add(h, f"{analysis} · {'合计' if scope == 'overall' else scope}", o and o["mean_diff_pp"],
                    o and o["ci95_pp"], n["mean_diff_pp"], n["ci95_pp"], p and p["ci95_pp"])

    old8, new8 = load(old_b / "H08_result.json")["analyses"][MAIN], load(out / "H08_result.json")["analyses"][MAIN]
    for kind, fit in new8["fits"].items():
        for key, s in fit["statistics"].items():
            if key.startswith("合计|"):
                o = old8["fits"][kind]["statistics"].get(key, {})
                add("H08 斜率（每 10pp 差距）", f"{kind} · {key}", o.get("point"), o.get("ci95"), s["point"], s["ci95"])
    old_dirs = {d["id"]: d for d in old8["directions"]}
    flips, widths = [], []
    for d in new8["directions"]:
        o = old_dirs[d["id"]]
        a, b = outcome_level(o["dG_ci95"], o["dC_ci95"]), outcome_level(d["dG_ci95"], d["dC_ci95"])
        if a != b:
            flips.append(dict(id=d["id"], old_level=a, new_level=b, dG=d["dG"], dC=d["dC"]))
        for k in ("dG", "dC"):
            widths.append(((d[f"{k}_ci95"][1] - d[f"{k}_ci95"][0]) / (o[f"{k}_ci95"][1] - o[f"{k}_ci95"][0]),
                           d["id"], k, d[k], o[f"{k}_ci95"], d[f"{k}_ci95"]))

    old9, new9 = load(old_c / "H09_result.json")["analyses"][MAIN], load(out / "H09_result.json")["analyses"][MAIN]
    levels = {k: (old9["kinds"][k]["levels"], new9["kinds"][k]["levels"],
                  old9["kinds"][k]["test"], new9["kinds"][k]["test"]) for k in new9["kinds"]}

    old_rq, new_rq = load(old_b / "rq2_cells.json")["summary"], load(out / "rq2_cells_audited.json")
    for key, s in new_rq["statistics"].items():
        kind, q = key.split("|")
        o = old_rq.get(kind, {}).get(q)
        add("第 4.2 节（RQ2）", key, o and o["mean"], o and o["ci95"], s["point"], s["ci95"],
            new_rq["sensitivity_pairs"][key]["ci95"])

    for h in ("H10", "H11"):
        old, new = load(old_d / f"{h}_result.json"), load(out / f"{h}_result.json")
        for o, n in zip(old["comparisons"], new["comparisons"]):
            if (o["a"], o["b"]) != (n["a"], n["b"]):
                raise ValueError(f"{h}: comparisons out of order")
            add(h, f"{n['a']} − {n['b']}（以上/持平/以下 {o['counts']} → {n['counts']}）",
                o["bootstrap"]["overall"]["mean_diff_pp"], o["bootstrap"]["overall"]["ci95_pp"],
                n["bootstrap"]["overall"]["mean_diff_pp"], n["bootstrap"]["overall"]["ci95_pp"],
                n["bootstrap_pairs"]["overall"]["ci95_pp"])

    for section in dict.fromkeys(r["section"] for r in rows):
        lines += [f"## {section}", "", "|量|改前|改后|按对重抽（附录）|", "|---|---|---|---|"]
        for r in (r for r in rows if r["section"] == section):
            fmt = lambda p, c: "—" if p is None else f"{p:+.2f} {ci(c)}"
            lines.append(f"|{r['name']}|{fmt(r['old_point'], r['old_ci'])}|{fmt(r['new_point'], r['new_ci'])}|{ci(r['pairs_ci'])}|")
        lines.append("")
    widths.sort()
    lines += ["## 96 个方向的五档", "",
              f"档位变化的方向：{len(flips)} 个。" + "".join(f"\n- {f['id']}：{f['old_level']} → {f['new_level']}" for f in flips), "",
              f"单方向区间宽度（改后 ÷ 改前）：最小 {widths[0][0]:.2f}，中位 {widths[len(widths) // 2][0]:.2f}，最大 {widths[-1][0]:.2f}"
              f"（{widths[-1][1]} 的 {widths[-1][2]} = {widths[-1][3]:+.1f}：{ci(widths[-1][4])} → {ci(widths[-1][5])}）。", "",
              "|谁审谁|五档计数 改前|改后|秩检验 改前|改后（正文将删去）|", "|---|---|---|---|---|"]
    for k, (a, b, ta, tb) in levels.items():
        lines.append(f"|{k}|{a}|{b}|{ta['point']:+.2f}, p={ta['p_two_sided']:.4f}|{tb['point']:+.2f}, p={tb['p_two_sided']:.4f}|")
    (out / "comparison.json").write_text(json.dumps({"rows": rows, "level_flips": flips, "levels": levels,
                                                     "width_ratio": [w[:4] for w in widths]},
                                                    ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--cost", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old-b", type=Path, default=OLD / "20261004b", help="H01-H08 and rq2_cells before the audit")
    ap.add_argument("--old-c", type=Path, default=OLD / "20261004c", help="H09 before the audit")
    ap.add_argument("--old-d", type=Path, default=OLD / "20261004d", help="H10, H11 before the audit")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("reports are immutable; choose a new output directory")
    args.out.mkdir(parents=True)
    try:
        if not load(args.db / "reports/systems.json")["all_archived_counts_match"]:
            raise ValueError("system accuracy validation did not pass")
        run_all(args.db, args.cost, args.out)
        # The pre-audit reports are not part of the code release; without them only the comparison is skipped.
        if all(p.exists() for p in (args.old_b, args.old_c, args.old_d)):
            compare(args.out, args.old_b, args.old_c, args.old_d)
        else:
            (args.out / "reports").mkdir(exist_ok=True)
            (args.out / "reports/summary.md").write_text(
                "# H01-H11 rerun\n\nThe reports before the audit were not found, so no before/after comparison.\n",
                encoding="utf-8")
    except Exception as exc:
        (args.out / "reports").mkdir(exist_ok=True)
        (args.out / "reports/summary.md").write_text(f"# 重跑失败\n\n{type(exc).__name__}: {exc}\n\n未放宽检查。\n",
                                                    encoding="utf-8")
        raise
    print(args.out / "reports/summary.md")


if __name__ == "__main__":
    main()
