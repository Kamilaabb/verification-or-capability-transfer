#!/usr/bin/env python3
"""The MMLU-Pro selector on the sealed test1500 prefix (docs/plan/mmlu_pro_study.md section 10).

Fixed before any test row ran. It uses the configuration that was best on development:
F3-B-l2 (mean 654.6 of 1000 over five seeds, job 65146561). It is refitted once on all of dev1000
(c070/c071, mc_pair_scalar_v1) with each of the five seeds' inner grid search, the five test
probabilities are averaged, and the averaged probabilities pick one model's answer per test row
(c081 Qwen3-4B = G, c082 Qwen3-8B = V; a tie delivers V). The math pair selectors' primary
ensemble (F2-A-l2 + F3-A-gbt), fitted the same way, is reported beside it for reference only.
Nothing on test is used to fit, tune or choose anything. No decision rule was set: the
report gives counts, paired discordant rows with exact sign-test p and a 95% interval, and the
anchoring tables for the two GV directions (c083: Qwen3-8B reviews c081; c084: the reverse).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mc_pair_selector_dev1000 as dev  # noqa: E402
import selector_dev1000 as sel  # noqa: E402
from test_mathfull_v2_analysis import joint, paired  # noqa: E402

TEST_G, TEST_V = "c081", "c082"
TEST_TAG = "mmlupro-test1500"
TEST_MANIFEST = ROOT / "data/manifests/mmlupro-rb189ec7-s20260927/test1500.jsonl"
CHOSEN = [("F3", "B", "l2")]
REFERENCE = [("F2", "A", "l2"), ("F3", "A", "gbt")]
DIRECTIONS = [{"condition": "c083", "reviewer": "v", "note": "Qwen3-8B reviews Qwen3-4B"},
              {"condition": "c084", "reviewer": "g", "note": "Qwen3-4B reviews Qwen3-8B"}]


def fit_predict(dev_data, test_data, fsets, members) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Refit each member on all development rows per seed; average test probabilities."""
    (rows_d, ids_d, yg_d, yv_d), (rows_t, ids_t, yg_t, yv_t) = dev_data, test_data
    pg_seeds, pv_seeds = [], []
    for seed in sel.SEEDS:
        pgs, pvs = [], []
        for fs, form, family in members:
            xg_d, xv_d = sel.matrix(rows_d, ids_d, fsets[fs]["g"]), sel.matrix(rows_d, ids_d, fsets[fs]["v"])
            xg_t, xv_t = sel.matrix(rows_t, ids_t, fsets[fs]["g"]), sel.matrix(rows_t, ids_t, fsets[fs]["v"])
            if form == "A":
                out = []
                for x_d, y, x_t in ((xg_d, yg_d, xg_t), (xv_d, yv_d, xv_t)):
                    model = sel.search(family, seed)
                    model.fit(x_d, y)
                    out.append(model.predict_proba(x_t)[:, 1])
                pg, pv = out
            else:
                disagree = yg_d != yv_d
                model = sel.search(family, seed)
                model.fit(xv_d[disagree], yv_d[disagree])
                pv = model.predict_proba(xv_t)[:, 1]
                pg = 1 - pv
            pgs.append(pg); pvs.append(pv)
        pg_seeds.append(np.mean(pgs, axis=0)); pv_seeds.append(np.mean(pvs, axis=0))
    per_seed = [int(np.where(a > b, yg_t, yv_t).sum()) for a, b in zip(pg_seeds, pv_seeds)]
    return np.mean(pg_seeds, axis=0), np.mean(pv_seeds, axis=0), per_seed


def gv_u(run: Path, ids: list[str]) -> np.ndarray:
    u = {r["problem_id"]: r["u"] for r in sel.read_jsonl(run / "records" / "grades.jsonl")
         if r.get("outcome") == "u_gv"}
    return np.asarray([u[p] for p in ids])


def fmt(p: dict) -> str:
    lo, hi = p["ci95"]
    return (f"{p['x_only']}:{p['y_only']}, {p['diff_rows']:+d} rows ({100 * p['diff']:+.1f} pt, "
            f"95% [{100 * lo:+.1f}, {100 * hi:+.1f}]), p={p['p_sign']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    ids_d, rows_d, yg_d, yv_d = dev.build(artifacts)
    ids_t, rows_t, yg_t, yv_t = dev.build(artifacts, TEST_G, TEST_V, TEST_MANIFEST, TEST_TAG)
    fsets = dev.feature_sets()
    dev_data, test_data = (rows_d, ids_d, yg_d, yv_d), (rows_t, ids_t, yg_t, yv_t)
    delivered = {}
    for name, members in (("chosen F3-B-l2", CHOSEN), ("reference F2-A-l2 + F3-A-gbt", REFERENCE)):
        pg, pv, per_seed = fit_predict(dev_data, test_data, fsets, members)
        use_g = pg > pv
        delivered[name] = {"rows": np.where(use_g, yg_t, yv_t), "chose_g": int(use_g.sum()), "per_seed": per_seed}
        print(name, int(delivered[name]["rows"].sum()), per_seed, flush=True)
    max_side = yv_t if yv_t.sum() >= yg_t.sum() else yg_t
    n = len(ids_t)
    report = {"rows": n, "g": TEST_G, "v": TEST_V, "g_correct": int(yg_t.sum()), "v_correct": int(yv_t.sum()),
              "max_gv": int(max_side.sum()), "oracle": int(np.maximum(yg_t, yv_t).sum()),
              "independent": joint(yg_t, yv_t), "selectors": {}, "directions": []}
    lines = ["# MMLU-Pro test1500: Qwen3-4B x Qwen3-8B, selector and GV", "",
             "Selector fixed on development before any test row ran (F3-B-l2, the best of sixteen on dev1000, "
             "refitted on all of dev1000; five seeds' probabilities averaged). No decision rule was set; "
             "paired comparisons give discordant rows (first:second), the difference with a 95% interval, and "
             "the exact two-sided sign-test p.", "",
             f"- Rows {n}. G = Qwen3-4B ({TEST_G}) {report['g_correct']}; V = Qwen3-8B ({TEST_V}) "
             f"{report['v_correct']}; max(G, V) {report['max_gv']}; oracle {report['oracle']}; "
             f"independent disagreement {report['independent']['disagree']}", ""]
    for name, d in delivered.items():
        cmp = paired(d["rows"], max_side)
        report["selectors"][name] = {"correct": int(d["rows"].sum()), "chose_g": d["chose_g"],
                                     "per_seed_fits": d["per_seed"], "vs_max_gv": cmp}
        lines.append(f"- Selector {name}: {int(d['rows'].sum())} correct (delivered G on {d['chose_g']} rows; "
                     f"single-seed fits {d['per_seed']}); vs max(G, V): {fmt(cmp)}")
    chosen = delivered["chosen F3-B-l2"]["rows"]
    lines += ["", "|direction|GV|GV vs max(G,V)|selector vs GV|reviewed vs reviewer: disagree indep -> exposed|"
              "union indep -> exposed|", "|---|---:|---|---|---|---|"]
    for d in DIRECTIONS:
        run = dev.archive(artifacts, d["condition"], TEST_TAG)
        if run is None:
            lines.append(f"|{d['condition']} {d['note']}|not finished|||||")
            continue
        u = gv_u(run, ids_t)
        reviewed = yg_t if d["reviewer"] == "v" else yv_t
        reviewer_alone = yv_t if d["reviewer"] == "v" else yg_t
        ind, exp = joint(reviewed, reviewer_alone), joint(reviewed, u)
        gv_max, sel_gv = paired(u, max_side), paired(chosen, u)
        report["directions"].append({"condition": d["condition"], "run": run.name, "gv": int(u.sum()),
                                     "gv_vs_max_gv": gv_max, "selector_vs_gv": sel_gv,
                                     "anchoring": {"independent": ind, "exposed": exp}})
        lines.append(f"|{d['condition']} {d['note']}|{int(u.sum())}|{fmt(gv_max)}|{fmt(sel_gv)}|"
                     f"{ind['disagree']} -> {exp['disagree']}|{ind['union']} -> {exp['union']}|")
    lines.append("")
    (out / "reports" / "mc_pair_selector_test1500.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
