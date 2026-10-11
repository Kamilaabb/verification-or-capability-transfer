#!/usr/bin/env python3
"""The frozen confirmatory analysis of docs/plan/test_qwen3_4b_qwen25_7b.md, written before any test run.

The Qwen2.5-7B / Qwen3-4B pair on the first 1,500 test rows. Fits the frozen selectors once
on all 999 development rows of the pair (c031 as G, c025 as V), applies them to the test-set
solo runs (c063/c062), and compares them with both GV directions (c064: Qwen2.5-7B reviews
Qwen3-4B; c065: Qwen3-4B reviews Qwen2.5-7B). Selectors, features, members, seeds, delivery rule and
the paired statistics are those of scripts/test_mathfull_v2_analysis.py, imported unchanged;
only the runs and the hypothesis family differ. Test grades are read only to score.

Primary family: H1a (selector > max(G, V, GV-A)) and H1b (selector > max(G, V, GV-B)), each an
exact two-sided sign test with a positive difference, Holm-corrected at 0.05 over the two.

--dry-run fits on c031/c025 and scores the same development runs with c060/c061, only to
exercise the code before the test archives exist; its numbers mean nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import selector_dev1000 as sel  # noqa: E402
from test_mathfull_v2_analysis import (ALPHA, PRIMARY, SECONDARY, fit_predict, gv_labels,  # noqa: E402
                                       joint, paired, solo_tokens)

DEV = {"g": "20260926T060236Z__c031__mathfull-dev1000__r012_solo4096",
       "v": "20260925T172823Z__c025__mathfull-dev1000__r012_solo4096"}
STEM = "20260927T055111Z"
TEST = {"g": f"{STEM}__c063__mathfull-test1500__r012_solo4096",
        "v": f"{STEM}__c062__mathfull-test1500__r012_solo4096",
        "gv_a": f"{STEM}__c064__mathfull-test1500__r012_review",
        "gv_b": f"{STEM}__c065__mathfull-test1500__r012_review",
        "manifest": "test1500"}
G_NAME, V_NAME = "G (Qwen2.5-7B alone)", "V (Qwen3-4B alone)"
# (key, hypothesis, label, the side the reviewer reviews)
DIRECTIONS = (("gv_a", "H1a", "GV-A (Qwen2.5-7B reviews Qwen3-4B)", "v"),
              ("gv_b", "H1b", "GV-B (Qwen3-4B reviews Qwen2.5-7B)", "g"))


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    rejected, still = {}, True
    for rank, (key, p) in enumerate(sorted(pvalues.items(), key=lambda kv: kv[1])):
        still = still and p < alpha / (len(pvalues) - rank)
        rejected[key] = still
    return rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-gv", nargs=2, metavar=("GV_A", "GV_B"),
                        help="with --dry-run: the c060 and c061 development archives")
    args = parser.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    target = dict(DEV, gv_a=args.dry_gv[0], gv_b=args.dry_gv[1], manifest="dev1000") if args.dry_run else TEST
    listed = {r["problem_id"] for r in yaml.safe_load(
        (ROOT / "datasets" / sel.DATASET / "ungradeable_gold.yaml").read_text(encoding="utf-8"))["ungradeable"]}

    sel.G_RUN, sel.V_RUN = DEV["g"], DEV["v"]
    ids_d, rows_d, yg_d, yv_d = sel.build(artifacts, ROOT / "data/manifests" / sel.DATASET / "dev1000.jsonl", listed)
    sel.G_RUN, sel.V_RUN = target["g"], target["v"]
    ids_t, rows_t, yg, yv = sel.build(artifacts, ROOT / "data/manifests" / sel.DATASET / f"{target['manifest']}.jsonl",
                                      listed)
    dev = (rows_d, ids_d, yg_d, yv_d, sel.feature_sets(rows_d))
    test = (rows_t, ids_t, sel.feature_sets(rows_t))

    delivered, choices = {}, {}
    for name, members in (("primary", PRIMARY), ("secondary", SECONDARY)):
        pg, pv = fit_predict(dev, test, members)
        delivered[name] = np.where(pg > pv, yg, yv)
        choices[name] = (pg > pv).astype(int).tolist()

    id_set = set(ids_t)
    g_tok, g_inv = solo_tokens(artifacts, target["g"], id_set)
    v_tok, v_inv = solo_tokens(artifacts, target["v"], id_set)
    systems = {G_NAME: yg, V_NAME: yv}
    tokens = {G_NAME: g_tok, V_NAME: v_tok, "selector": g_tok + v_tok}
    gvs, fallbacks = {}, {}
    for key, _, label, reviewed in DIRECTIONS:
        u, fb, rev_tok = gv_labels(artifacts, target[key], ids_t)
        gvs[key], systems[label], fallbacks[label] = u, u, fb
        tokens[label] = (g_tok if reviewed == "g" else v_tok) + rev_tok

    report = {"dry_run": args.dry_run, "rows": len(ids_t), "excluded_ungradeable": sorted(listed & id_set),
              "dev_runs": DEV, "target_runs": target,
              "accuracy": {k: int(v.sum()) for k, v in systems.items()},
              "selector_primary": int(delivered["primary"].sum()),
              "selector_secondary": int(delivered["secondary"].sum()),
              "oracle_g_or_v": int(np.maximum(yg, yv).sum()),
              "output_tokens": tokens, "gv_fallback_rows": fallbacks,
              "extraction_timeouts": {"G": g_inv, "V": v_inv},
              "selector_choices": {"ids": ids_t, **choices}, "directions": {}}

    for key, hyp, label, reviewed_side in DIRECTIONS:
        reviewed, other = (yg, yv) if reviewed_side == "g" else (yv, yg)
        three = {G_NAME: yg, V_NAME: yv, label: gvs[key]}
        best = max(int(v.sum()) for v in three.values())
        tied = [k for k, v in three.items() if int(v.sum()) == best]
        d = {"hypothesis": hyp, "max_of_three": best, "comparators": tied, "selector": {}}
        for name in ("primary", "secondary"):
            results = {k: paired(delivered[name], three[k]) for k in tied}
            # With several systems tied at the maximum, the least favourable comparison counts.
            worst = min(results, key=lambda k: (results[k]["diff_rows"], -results[k]["p_sign"]))
            d["selector"][name] = {"against": worst, **results[worst], "all_tied": results}
        d["gv_vs_reviewed_alone"] = paired(gvs[key], reviewed)
        d["gv_vs_max_single"] = paired(gvs[key], yv if yv.sum() >= yg.sum() else yg)
        d["anchoring"] = {"independent": joint(reviewed, other), "exposed": joint(reviewed, gvs[key])}
        report["directions"][label] = d

    for name in ("primary", "secondary"):
        ps = {d["hypothesis"]: (d["selector"][name]["p_sign"] if d["selector"][name]["diff_rows"] > 0 else 1.0)
              for d in report["directions"].values()}
        report[f"holm_{name}"] = {"p": ps, "rejected": holm(ps)}

    (out / "reports" / "test_qwen3_4b_qwen25_7b.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# {'DRY RUN (development archives; numbers meaningless) - ' if args.dry_run else ''}"
             "Confirmatory test, Qwen2.5-7B / Qwen3-4B, test1500", "",
             "Pre-registered in docs/plan/test_qwen3_4b_qwen25_7b.md; analysis code frozen before any test run.", "",
             f"- Rows {len(ids_t)}; excluded as ungradeable gold: {report['excluded_ungradeable']}",
             f"- Oracle (G or V right): {report['oracle_g_or_v']}", "",
             "|system|correct|accuracy|output tokens|", "|---|---:|---:|---:|"]
    for k, v in systems.items():
        lines.append(f"|{k}|{int(v.sum())}|{v.mean():.4f}|{tokens[k]:,}|")
    for name in ("primary", "secondary"):
        lines.append(f"|selector ({name})|{int(delivered[name].sum())}|{delivered[name].mean():.4f}|"
                     f"{tokens['selector']:,}|")
    for label, d in report["directions"].items():
        lines += ["", f"## {d['hypothesis']}: {label}", "",
                  f"- max(G, V, GV) = {d['max_of_three']} ({', '.join(d['comparators'])})"]
        for name in ("primary", "secondary"):
            s = d["selector"][name]
            rejected = report[f"holm_{name}"]["rejected"][d["hypothesis"]]
            lines.append(f"- selector ({name}) against {s['against']}: {s['x_only']}:{s['y_only']} discordant, "
                         f"diff {s['diff']:+.4f} (95% CI {s['ci95'][0]:+.4f} to {s['ci95'][1]:+.4f}), "
                         f"sign test p = {s['p_sign']:.4f}, Holm over H1a/H1b: "
                         f"{'significant' if rejected else 'not significant'}")
        g = d["gv_vs_reviewed_alone"]
        lines.append(f"- GV against the reviewed model alone: {g['x_only']}:{g['y_only']}, diff {g['diff']:+.4f} "
                     f"(95% CI {g['ci95'][0]:+.4f} to {g['ci95'][1]:+.4f}), p = {g['p_sign']:.4f}")
        m = d["gv_vs_max_single"]
        lines.append(f"- GV against max(G, V) alone: {m['x_only']}:{m['y_only']}, diff {m['diff']:+.4f} "
                     f"(95% CI {m['ci95'][0]:+.4f} to {m['ci95'][1]:+.4f}), p = {m['p_sign']:.4f}")
        ind, exp = d["anchoring"]["independent"], d["anchoring"]["exposed"]
        lines += ["", "|reviewed model against|independent other model|reviewer after seeing it|", "|---|---:|---:|"]
        for field in ("disagree", "both_wrong", "union", "second_wrong_given_first_wrong", "phi"):
            fmt = (lambda x: f"{x:.3f}") if field in ("second_wrong_given_first_wrong", "phi") else str
            lines.append(f"|{field}|{fmt(ind[field])}|{fmt(exp[field])}|")
    lines += ["", f"- GV rows delivered from the reviewed model because the reviewer gave no answer: {fallbacks}",
              f"- Solo rows whose answer extraction timed out (model_invalid with an output): {report['extraction_timeouts']}", ""]
    (out / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
