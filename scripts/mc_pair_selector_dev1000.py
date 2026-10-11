#!/usr/bin/env python3
"""Selectors and GV for the MMLU-Pro pair Qwen3-4B x Qwen3-8B on dev1000
(docs/plan/mmlu_pro_study.md section 9).

Development exploration, fixed before it was first run. G is Qwen3-4B
(c070) and V is Qwen3-8B (c071) as registered in pairs/mmlupro_dev1000.yaml, so a tie delivers V
as in the math selectors. Features are mc_pair_scalar_v1 (authority_allocation.mc_pair_features),
computed offline from the two solo archives and each model's own tokenizer.json (to read which
top-20 token ids spell an option letter); labels are their archived u0. The sixteen
configurations, folds, seeds, grids and the primary/secondary ensembles are those of
scripts/selector_dev1000.py and scripts/pair_selector_dev1000.py, imported unchanged:
F1 = the 22 registered evidence scalars per model, F2 = the 38 per-model features plus the 18
subject/source indicators, F3 = both models' features, the indicators and the cross features.
Each GV direction (c072: Qwen3-8B reviews Qwen3-4B; c073: the reverse) is then compared with the
selectors and with max(G alone, V alone, GV), with the anchoring tables. Nothing is written into
any archive.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import selector_dev1000 as sel  # noqa: E402
from test_mathfull_v2_analysis import joint, paired  # noqa: E402

from authority_allocation import mc_pair_features as mpf  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402

G_COND, V_COND = "c070", "c071"
NAMES = {"c070": "Qwen3-4B", "c071": "Qwen3-8B"}
DIRECTIONS = [{"condition": "c072", "reviewer": "v", "note": "Qwen3-8B reviews Qwen3-4B"},
              {"condition": "c073", "reviewer": "g", "note": "Qwen3-4B reviews Qwen3-8B"}]
MANIFEST = ROOT / "data/manifests/mmlupro-rb189ec7-s20260927/dev1000.jsonl"
MEMBERS = {"primary": [("F2", "A", "l2"), ("F3", "A", "gbt")], "secondary": [("F2", "A", "l2")]}
FAMILIES = ["l2", "tree", "forest", "gbt"]


def archive(artifacts: Path, condition: str, tag: str = "mmlupro-dev1000") -> Path | None:
    hits = sorted(glob.glob(str(artifacts / f"*__{condition}__{tag}__*")))
    hits = [h for h in hits if (Path(h) / "checksums.sha256").exists()]
    return Path(hits[-1]) if hits else None


def load(run: Path) -> dict[str, dict[str, dict]]:
    rec = run / "records"
    return {name: {r["problem_id"]: r for r in sel.read_jsonl(rec / f"{name}.jsonl")
                   if name != "grades" or r["outcome"] == "u0"}
            for name in ("pass1", "native_logit_trace", "scalar_evidence", "grades")}


def letters_for(condition: str) -> dict[int, str]:
    model = resolve_condition(ROOT / "conditions" / condition / "condition.yaml", repository_root=ROOT)["model"]
    vocab = json.loads((Path(model["server_path"]) / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    return mpf.letter_map({tid: tok for tok, tid in vocab.items()})


def build(artifacts: Path, g_cond: str = G_COND, v_cond: str = V_COND, manifest: Path = MANIFEST,
          tag: str = "mmlupro-dev1000") -> tuple[list[str], dict, np.ndarray, np.ndarray]:
    """Rows of the pair's two solo archives on one manifest (development by default; the test
    script passes the test1500 conditions)."""
    ids = [r["problem_id"] for r in sel.read_jsonl(manifest)]
    runs = {"g": load(archive(artifacts, g_cond, tag)), "v": load(archive(artifacts, v_cond, tag))}
    letters = {"g": letters_for(g_cond), "v": letters_for(v_cond)}
    rows, y_g, y_v = {}, [], []
    for pid in ids:
        row: dict[str, float | None] = {}
        feats, extras = {}, {}
        for side in ("g", "v"):
            r = runs[side]
            feats[side], extras[side] = mpf.model_features(r["pass1"][pid], r["native_logit_trace"][pid],
                                                           r["scalar_evidence"][pid], letters[side])
            row.update({f"{side}_{k}": v for k, v in feats[side].items()})
        row.update(mpf.problem_features(runs["v"]["pass1"][pid]))
        row.update(mpf.cross_features(feats["g"], feats["v"], extras["g"], extras["v"]))
        rows[pid] = row
        y_g.append(int(runs["g"]["grades"][pid]["u"]))
        y_v.append(int(runs["v"]["grades"][pid]["u"]))
    return ids, rows, np.asarray(y_g), np.asarray(y_v)


def feature_sets() -> dict[str, dict[str, list[str]]]:
    single = lambda side: [f"{side}_{n}" for n in mpf.MODEL_FEATURES] + list(mpf.PROBLEM_FEATURES)  # noqa: E731
    both = [f"g_{n}" for n in mpf.MODEL_FEATURES] + [f"v_{n}" for n in mpf.MODEL_FEATURES] \
        + list(mpf.PROBLEM_FEATURES) + list(mpf.CROSS_FEATURES)
    evidence = lambda side: [f"{side}_{n}" for n in mpf.EVIDENCE_FEATURES]  # noqa: E731
    return {"F1": {"g": evidence("g"), "v": evidence("v")},
            "F2": {"g": single("g"), "v": single("v")},
            "F3": {"g": both, "v": both}}


def ensemble(rows, ids, yg, yv, fsets, members) -> dict:
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
    return {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "rows": np.where(mpg > mpv, yg, yv)}


def gv_u(run: Path, ids: list[str]) -> np.ndarray:
    u = {r["problem_id"]: r["u"] for r in sel.read_jsonl(run / "records" / "grades.jsonl")
         if r.get("outcome") == "u_gv"}
    return np.asarray([u[p] for p in ids])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    ids, rows, yg, yv = build(artifacts)
    fsets = feature_sets()
    missing = {n: sum(rows[p][n] is None for p in ids) for n in fsets["F3"]["g"]}
    print(json.dumps({"rows": len(ids), "G": int(yg.sum()), "V": int(yv.sum()),
                      "features": {k: len(v["g"]) for k, v in fsets.items()}}), flush=True)

    configs = [(fs, "A", fam) for fs in ("F1", "F2", "F3") for fam in FAMILIES] + \
              [("F3", "B", fam) for fam in FAMILIES]
    main_results = []
    for config in configs:
        t0 = time.time()
        res = sel.evaluate(rows, ids, yg, yv, fsets, config)
        main_results.append(res)
        print(f"{res['config']:<14} seeds {res['per_seed_correct']} mean {res['mean_correct']:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    delivered = {name: ensemble(rows, ids, yg, yv, fsets, members) for name, members in MEMBERS.items()}
    max_side = yv if yv.sum() >= yg.sum() else yg

    g_name, v_name = NAMES[G_COND], NAMES[V_COND]
    report = {"g": G_COND, "v": V_COND, "rows": len(ids), "g_correct": int(yg.sum()), "v_correct": int(yv.sum()),
              "max_gv": int(max_side.sum()), "oracle": int(np.maximum(yg, yv).sum()),
              "independent": joint(yg, yv), "feature_missing": {k: v for k, v in missing.items() if v},
              "features_per_head": {k: len(v["g"]) for k, v in fsets.items()},
              "configs": [{k: v for k, v in r.items() if k != "seed_mean_use_g"} for r in main_results],
              "selector": {k: {"per_seed": d["per_seed"], "mean": d["mean"], "seed_mean_prob": int(d["rows"].sum())}
                           for k, d in delivered.items()},
              "primary_vs_max_gv": paired(delivered["primary"]["rows"], max_side), "directions": []}
    lines = ["# Selectors and GV: Qwen3-4B x Qwen3-8B on MMLU-Pro dev1000", "",
             "Development exploration, fixed before its first run; selectors cross-fitted "
             "(5 outer folds x 5 seeds) on the two solo archives with mc_pair_scalar_v1. Paired comparisons: "
             "exact two-sided sign test on discordant rows.", "",
             f"- G = {g_name} ({G_COND}) {report['g_correct']}; V = {v_name} ({V_COND}) {report['v_correct']}; "
             f"max(G, V) {report['max_gv']}; oracle {report['oracle']}; independent disagreement "
             f"{report['independent']['disagree']}",
             f"- Features per head: {report['features_per_head']}", "",
             "|config|features/head|per-seed correct|mean|vs max(G,V)|seed-mean-prob|repair/harm vs V|",
             "|---|---:|---|---:|---:|---:|---|"]
    for r in main_results:
        sm = r["seed_mean_prob"]
        lines.append(f"|{r['config']}|{r['features_per_head']}|{r['per_seed_correct']}|{r['mean_correct']:.1f}|"
                     f"{r['mean_correct'] - r['max_gv']:+.1f}|{sm['correct']}|{sm['repair_vs_v']}/{sm['harm_vs_v']}|")
    pm = report["primary_vs_max_gv"]
    lines += ["", f"- Selector primary (F2-A-l2 + F3-A-gbt): mean {delivered['primary']['mean']:.1f} "
              f"{delivered['primary']['per_seed']}; secondary (F2-A-l2): mean {delivered['secondary']['mean']:.1f} "
              f"{delivered['secondary']['per_seed']}",
              f"- Primary (seed-mean probabilities) vs max(G, V): {pm['x_only']}:{pm['y_only']}, "
              f"{pm['diff_rows']:+d} rows, p={pm['p_sign']:.3f}", "",
              "|direction|GV|max(G,V,GV)|primary vs that max (sel:max, p)|secondary vs max|"
              "reviewed vs reviewer: disagree indep -> exposed|union indep -> exposed|",
              "|---|---:|---:|---|---|---|---|"]
    for d in DIRECTIONS:
        run = archive(artifacts, d["condition"])
        if run is None:
            lines.append(f"|{d['condition']} {d['note']}|not finished||||||")
            continue
        u = gv_u(run, ids)
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
        report["directions"].append({"condition": d["condition"], "run": run.name, "note": d["note"],
                                     "gv": int(u.sum()), "max_of_three": best, "comparators": tied,
                                     "gv_vs_max_gv": paired(u, max_side), "selector_vs_max": cmp,
                                     "anchoring": {"independent": ind, "exposed": exp}})
        pr, se = cmp["primary"], cmp["secondary"]
        lines.append(f"|{d['condition']} {d['note']}|{int(u.sum())}|{best} ({', '.join(tied)})|"
                     f"{pr['x_only']}:{pr['y_only']}, {pr['diff_rows']:+d}, p={pr['p_sign']:.3f}|"
                     f"{se['x_only']}:{se['y_only']}, {se['diff_rows']:+d}, p={se['p_sign']:.3f}|"
                     f"{ind['disagree']} -> {exp['disagree']}|{ind['union']} -> {exp['union']}|")
    lines.append("")
    (out / "reports" / "mc_pair_selector.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
