#!/usr/bin/env python3
"""Selectors and GV per model pair on mathfull dev1000 (pairs/mathfull_v2_dev1000.yaml).

For every registered pair: cross-fit the test plan's two selectors on the pair's solo archives
with selector_dev1000's features, folds, seeds and grids (primary: the mean of F2-A-l2 and
F3-A-gbt; secondary: F2-A-l2), then compare each finished GV direction with the selectors and
with max(G alone, V alone, GV), and report the anchoring statistics. Development exploration;
nothing here is registered as confirmatory.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import selector_dev1000 as sel  # noqa: E402
from test_mathfull_v2_analysis import joint, paired  # noqa: E402

MEMBERS = {"primary": [("F2", "A", "l2"), ("F3", "A", "gbt")], "secondary": [("F2", "A", "l2")]}


def archive(artifacts: Path, condition: str) -> str | None:
    hits = sorted(glob.glob(str(artifacts / f"*__{condition}__mathfull-dev1000__*")))
    hits = [h for h in hits if (Path(h) / "checksums.sha256").exists()]
    return Path(hits[-1]).name if hits else None


def gv_u(artifacts: Path, run: str, ids: list[str]) -> np.ndarray:
    u = {r["problem_id"]: r["u"] for r in sel.read_jsonl(artifacts / run / "records" / "grades.jsonl")
         if r.get("outcome") == "u_gv"}
    return np.asarray([u[p] for p in ids])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--registry", default=str(ROOT / "pairs" / "mathfull_v2_dev1000.yaml"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    listed = {r["problem_id"] for r in yaml.safe_load(
        (ROOT / "datasets" / sel.DATASET / "ungradeable_gold.yaml").read_text(encoding="utf-8"))["ungradeable"]}
    manifest = ROOT / "data/manifests" / sel.DATASET / f"{registry['manifest']}.jsonl"
    report, lines = [], ["# Selectors and GV per model pair (mathfull dev1000, prompts v2)", "",
                         "Development exploration; selectors cross-fitted (5 outer folds x 5 seeds) on each pair's "
                         "solo archives. Paired comparisons: exact two-sided sign test on discordant rows.", ""]
    for pair in registry["pairs"]:
        g_run, v_run = archive(artifacts, pair["g"]["condition"]), archive(artifacts, pair["v"]["condition"])
        sel.G_RUN, sel.V_RUN = g_run, v_run
        ids, rows, yg, yv = sel.build(artifacts, manifest, listed)
        fsets = sel.feature_sets(rows)
        delivered = {}
        for name, members in MEMBERS.items():
            pg_all, pv_all = [], []
            for seed in sel.SEEDS:
                pgs, pvs = [], []
                for fs, form, fam in members:
                    x_g, x_v = sel.matrix(rows, ids, fsets[fs]["g"]), sel.matrix(rows, ids, fsets[fs]["v"])
                    pg, pv = sel.crossfit(x_g, x_v, yg, yv, fam, form, seed)
                    pgs.append(pg); pvs.append(pv)
                pg_all.append(np.mean(pgs, axis=0)); pv_all.append(np.mean(pvs, axis=0))
            per_seed = [int(np.where(a > b, yg, yv).sum()) for a, b in zip(pg_all, pv_all)]
            mpg, mpv = np.mean(pg_all, axis=0), np.mean(pv_all, axis=0)
            delivered[name] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                               "rows": np.where(mpg > mpv, yg, yv)}
            print(f"{pair['id']} {name} {per_seed}", flush=True)
        g_name, v_name = pair["g"]["model"], pair["v"]["model"]
        entry = {"pair": pair["id"], "g_run": g_run, "v_run": v_run, "rows": len(ids),
                 "g": int(yg.sum()), "v": int(yv.sum()), "oracle": int(np.maximum(yg, yv).sum()),
                 "selector": {k: {"per_seed": d["per_seed"], "mean": d["mean"],
                                  "seed_mean_prob": int(d["rows"].sum())} for k, d in delivered.items()},
                 "vs_max_gv": paired(delivered["primary"]["rows"], yv if yv.sum() >= yg.sum() else yg),
                 "directions": []}
        lines += [f"## {pair['id']}: G = {g_name} {entry['g']}, V = {v_name} {entry['v']}, oracle {entry['oracle']}", "",
                  f"- Selector primary (L2 + gbt): mean {delivered['primary']['mean']:.1f} {delivered['primary']['per_seed']}; "
                  f"secondary (F2-A-l2): mean {delivered['secondary']['mean']:.1f} {delivered['secondary']['per_seed']}",
                  "", "|direction|GV|max(G,V,GV)|primary vs that max (sel:max, p)|secondary vs max|"
                  "reviewed vs reviewer: disagree indep -> exposed|union indep -> exposed|",
                  "|---|---:|---:|---|---|---|---|"]
        for d in pair["gv"]:
            run = archive(artifacts, d["condition"])
            if run is None:
                lines.append(f"|{d['condition']} {d['note']}|not finished||||||")
                continue
            u = gv_u(artifacts, run, ids)
            three = {g_name: yg, v_name: yv, "GV": u}
            best = max(int(x.sum()) for x in three.values())
            tied = [k for k, x in three.items() if int(x.sum()) == best]
            cmp = {}
            for name in ("primary", "secondary"):
                res = {k: paired(delivered[name]["rows"], three[k]) for k in tied}
                cmp[name] = res[min(res, key=lambda k: (res[k]["diff_rows"], -res[k]["p_sign"]))]
            reviewed = yg if d["reviewer"] == "v" else yv
            reviewer_alone = yv if d["reviewer"] == "v" else yg
            ind, exp = joint(reviewed, reviewer_alone), joint(reviewed, u)
            entry["directions"].append({"condition": d["condition"], "run": run, "note": d["note"],
                                        "gv": int(u.sum()), "max_of_three": best, "comparators": tied,
                                        "selector_vs_max": cmp, "anchoring": {"independent": ind, "exposed": exp}})
            pr, se = cmp["primary"], cmp["secondary"]
            lines.append(f"|{d['condition']} {d['note']}|{int(u.sum())}|{best} ({', '.join(tied)})|"
                         f"{pr['x_only']}:{pr['y_only']}, {pr['diff_rows']:+d}, p={pr['p_sign']:.3f}|"
                         f"{se['x_only']}:{se['y_only']}, {se['diff_rows']:+d}, p={se['p_sign']:.3f}|"
                         f"{ind['disagree']} -> {exp['disagree']}|{ind['union']} -> {exp['union']}|")
        lines.append("")
        report.append(entry)
    (out / "reports" / "pair_selector.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
