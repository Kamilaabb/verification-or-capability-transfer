#!/usr/bin/env python3
"""Selectors per model pair on MMLU-Pro dev1000: all 36 pairs of the nine solo models
(pairs/mmlupro_screen_dev1000.yaml, docs/plan/mmlu_pro_study.md section 14).

The knowledge-domain counterpart of scripts/pair_selector_dev1000.py's screen. For every pair it
builds mc_pair_scalar_v1 features from the two solo archives exactly as
scripts/mc_pair_selector_dev1000.py does, cross-fits that script's two selectors (primary: the mean
of F2-A-l2 and F3-A-gbt; secondary: F2-A-l2; 5 outer folds x 5 seeds, grids of
scripts/selector_dev1000.py) and reports headroom (oracle - max(G, V)) and each selector against
max(G, V). The three pairs with finished GV directions also get the GV comparison and the anchoring
counts. Development exploration; nothing here is registered as confirmatory.
Nothing is written into any archive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mc_pair_selector_dev1000 as mps  # noqa: E402
from test_mathfull_v2_analysis import joint, paired  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--registry", default=str(ROOT / "pairs" / "mmlupro_screen_dev1000.yaml"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    if registry["manifest"] != "dev1000":
        raise SystemExit("the screen runs on dev1000 only")
    fsets = mps.feature_sets()

    report, table, sections = [], [], []
    for pair in registry["pairs"]:
        g_cond, v_cond = pair["g"]["condition"], pair["v"]["condition"]
        g_run, v_run = mps.archive(artifacts, g_cond), mps.archive(artifacts, v_cond)
        if g_run is None or v_run is None:
            raise SystemExit(f"{pair['id']}: a solo archive is missing")
        ids, rows, yg, yv = mps.build(artifacts, g_cond, v_cond)
        if yv.sum() < yg.sum():
            raise SystemExit(f"{pair['id']}: v must be the model with the higher solo accuracy")
        delivered = {name: mps.ensemble(rows, ids, yg, yv, fsets, members)
                     for name, members in mps.MEMBERS.items()}
        print(f"{pair['id']} primary {delivered['primary']['per_seed']} "
              f"secondary {delivered['secondary']['per_seed']}", flush=True)
        g_name, v_name = pair["g"]["model"], pair["v"]["model"]
        oracle = int(np.maximum(yg, yv).sum())
        vs_max = {name: paired(d["rows"], yv) for name, d in delivered.items()}
        entry = {"pair": pair["id"], "g": g_cond, "v": v_cond, "g_run": g_run.name, "v_run": v_run.name,
                 "rows": len(ids), "g_correct": int(yg.sum()), "v_correct": int(yv.sum()),
                 "max_gv": int(yv.sum()), "oracle": oracle, "headroom": oracle - int(yv.sum()),
                 "independent": joint(yg, yv),
                 "selector": {k: {"per_seed": d["per_seed"], "mean": d["mean"],
                                  "seed_mean_prob": int(d["rows"].sum()), "vs_max_gv": vs_max[k]}
                              for k, d in delivered.items()},
                 "directions": []}
        pm, sm = vs_max["primary"], vs_max["secondary"]
        table.append(f"|{pair['id']}|{g_name} {entry['g_correct']}|{v_name} {entry['v_correct']}|{oracle}|"
                     f"{entry['headroom']}|{delivered['primary']['mean']:.1f}|"
                     f"{delivered['primary']['mean'] - entry['max_gv']:+.1f}|"
                     f"{pm['x_only']}:{pm['y_only']}, p={pm['p_sign']:.3f}|"
                     f"{delivered['secondary']['mean']:.1f}|"
                     f"{delivered['secondary']['mean'] - entry['max_gv']:+.1f}|"
                     f"{sm['x_only']}:{sm['y_only']}, p={sm['p_sign']:.3f}|")
        if pair["gv"]:
            sections += [f"## {pair['id']}: G = {g_name} {entry['g_correct']}, V = {v_name} {entry['v_correct']}, "
                         f"oracle {oracle}", "",
                         "|direction|GV|max(G,V,GV)|primary vs that max (sel:max, p)|secondary vs max|"
                         "reviewed vs reviewer: disagree indep -> exposed|union indep -> exposed|",
                         "|---|---:|---:|---|---|---|---|"]
        for d in pair["gv"]:
            run = mps.archive(artifacts, d["condition"])
            if run is None:
                sections.append(f"|{d['condition']} {d['note']}|not finished||||||")
                continue
            u = mps.gv_u(run, ids)
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
            entry["directions"].append({"condition": d["condition"], "run": run.name, "note": d["note"],
                                        "gv": int(u.sum()), "max_of_three": best, "comparators": tied,
                                        "gv_vs_max_gv": paired(u, yv), "selector_vs_max": cmp,
                                        "anchoring": {"independent": ind, "exposed": exp}})
            pr, se = cmp["primary"], cmp["secondary"]
            sections.append(f"|{d['condition']} {d['note']}|{int(u.sum())}|{best} ({', '.join(tied)})|"
                            f"{pr['x_only']}:{pr['y_only']}, {pr['diff_rows']:+d}, p={pr['p_sign']:.3f}|"
                            f"{se['x_only']}:{se['y_only']}, {se['diff_rows']:+d}, p={se['p_sign']:.3f}|"
                            f"{ind['disagree']} -> {exp['disagree']}|{ind['union']} -> {exp['union']}|")
        if pair["gv"]:
            sections.append("")
        report.append(entry)

    lines = ["# Selectors per model pair: all 36 pairs of the MMLU-Pro dev1000 solo models", "",
             "Development exploration; selectors cross-fitted (5 outer folds x 5 seeds) on each pair's solo "
             "archives with mc_pair_scalar_v1, as scripts/mc_pair_selector_dev1000.py. V is the model with the "
             "higher solo accuracy, so max(G, V) = V. Paired comparisons use the selector's seed-mean "
             "probabilities against V: exact two-sided sign test on discordant rows. Choosing the best of 36 "
             "pairs on these numbers overstates its gain (winner's curse).", "",
             "|pair|G|V = max(G,V)|oracle|headroom|primary mean|primary - max|primary vs max (sel:V, p)|"
             "secondary mean|secondary - max|secondary vs max|",
             "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|"] + table + [""] + sections
    (out / "reports" / "mc_pair_screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
