#!/usr/bin/env python3
"""The frozen confirmatory analysis of docs/plan/test_mathfull_v2.md, written before any test run.

Fits the frozen selectors once on all 999 development rows (c025/c026), applies them to the
test-set solo runs (c040/c041), and computes every pre-declared quantity against the test GV
runs (c042: Qwen3-8B reviews Qwen3-4B; c043: Qwen3-4B reviews Qwen3-8B). Features come from
scripts/selector_dev1000.py unchanged; the test gold answers and grades are read only to
score, never to fit or choose anything. Nothing is chosen after reading the test labels.

Selectors (frozen):
    primary    each head's probability is the mean over the members F2-A-l2 and F3-A-gbt and
               over the five selector seeds; each (member, seed, head) model is the member's
               grid search refitted on all development rows with that seed's inner folds
    secondary  F2-A-l2 alone, fitted the same way
    delivery   the side with the higher probability of being right; ties deliver V

--dry-run fits on c025/c026 and scores the development pair c022/c023 with c024 as the
8B-reviews-4B GV, only to exercise the code before the test archives exist. Its numbers
mean nothing: the same problems were used to fit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import selector_dev1000 as sel  # noqa: E402

DEV = {"g": "20260925T172823Z__c025__mathfull-dev1000__r012_solo4096",
       "v": "20260925T172823Z__c026__mathfull-dev1000__r012_solo4096"}
TEST = {"g": "20260926T091114Z__c040__mathfull-test__r012_solo4096",
        "v": "20260926T091114Z__c041__mathfull-test__r012_solo4096",
        "gv_a": "20260926T091114Z__c042__mathfull-test__r012_review",
        "gv_b": "20260926T091114Z__c043__mathfull-test__r012_review",
        "manifest": "test"}
DRY = {"g": "20260924T203442Z__c022__mathfull-dev1000__r012_solo4096",
       "v": "20260924T203442Z__c023__mathfull-dev1000__r012_solo4096",
       "gv_a": "20260925T105636Z__c024__mathfull-dev1000__r012_review",
       "gv_b": None, "manifest": "dev1000"}
PRIMARY = [("F2", "l2"), ("F3", "gbt")]
SECONDARY = [("F2", "l2")]
ALPHA = 0.05


def sign_p(a: int, b: int) -> float:
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(x: np.ndarray, y: np.ndarray) -> dict:
    """x against y on the same rows: discordant counts, exact two-sided sign test, and the
    accuracy difference with a 95% normal interval for paired proportions."""
    n = len(x)
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    d = (b - c) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    return {"x_only": b, "y_only": c, "p_sign": sign_p(b, c), "diff": d,
            "ci95": [d - 1.96 * se, d + 1.96 * se], "diff_rows": b - c}


def joint(a: np.ndarray, b: np.ndarray) -> dict:
    n11, n10 = int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n01, n00 = int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())
    ra, rb, n = n11 + n10, n11 + n01, len(a)
    den = math.sqrt(ra * (n - ra) * rb * (n - rb))
    return {"both_right": n11, "only_first": n10, "only_second": n01, "both_wrong": n00,
            "disagree": n10 + n01, "union": n11 + n10 + n01,
            "second_wrong_given_first_wrong": n00 / (n01 + n00) if n01 + n00 else None,
            "phi": (n11 * n00 - n10 * n01) / den if den else None}


def fit_predict(dev: tuple, test: tuple, members: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    (rows_d, ids_d, yg_d, yv_d, fs_d), (rows_t, ids_t, fs_t) = dev, test
    pg, pv = [], []
    for fs, family in members:
        names = fs_d[fs]
        if names != fs_t[fs]:
            raise SystemExit(f"{fs} feature names differ between development and test")
        for head, y, out in (("g", yg_d, pg), ("v", yv_d, pv)):
            x_d, x_t = sel.matrix(rows_d, ids_d, names[head]), sel.matrix(rows_t, ids_t, names[head])
            for seed in sel.SEEDS:
                model = sel.search(family, seed)
                model.fit(x_d, y)
                out.append(model.predict_proba(x_t)[:, 1])
    return np.mean(pg, axis=0), np.mean(pv, axis=0)


def gv_labels(artifacts: Path, run: str, ids: list[str]) -> tuple[np.ndarray, int, int]:
    u, fallback = {}, 0
    for r in sel.read_jsonl(artifacts / run / "records" / "grades.jsonl"):
        if r.get("outcome") == "u_gv":
            u[r["problem_id"]] = r["u"]
            fallback += r.get("delivered_from") == "g_fallback" and r["problem_id"] in ids
    missing = [p for p in ids if p not in u]
    if missing or any(u[p] is None for p in ids):
        raise SystemExit(f"{run}: missing or ungradeable GV rows {missing[:5]}")
    tokens = sum(int(r["outputs"][0]["usage"].get("output_tokens") or 0)
                 for r in sel.read_jsonl(artifacts / run / "records" / "review.jsonl") if r["problem_id"] in ids)
    return np.asarray([u[p] for p in ids]), fallback, tokens


def solo_tokens(artifacts: Path, run: str, ids: set[str]) -> tuple[int, int]:
    tokens = invalid = 0
    for r in sel.read_jsonl(artifacts / run / "records" / "pass1.jsonl"):
        if r["problem_id"] in ids:
            o = r["outputs"][0]
            tokens += int(o["usage"].get("output_tokens") or 0)
            invalid += o["parse_status"] == "model_invalid" and bool(o["raw_output"])
    return tokens, invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    target = DRY if args.dry_run else TEST
    import yaml
    listed = {r["problem_id"] for r in yaml.safe_load(
        (ROOT / "datasets" / sel.DATASET / "ungradeable_gold.yaml").read_text(encoding="utf-8"))["ungradeable"]}

    sel.G_RUN, sel.V_RUN = DEV["g"], DEV["v"]
    ids_d, rows_d, yg_d, yv_d = sel.build(artifacts, ROOT / "data/manifests" / sel.DATASET / "dev1000.jsonl", listed)
    sel.G_RUN, sel.V_RUN = target["g"], target["v"]
    ids_t, rows_t, yg, yv = sel.build(artifacts, ROOT / "data/manifests" / sel.DATASET / f"{target['manifest']}.jsonl",
                                      listed)
    fs_d, fs_t = sel.feature_sets(rows_d), sel.feature_sets(rows_t)
    dev, test = (rows_d, ids_d, yg_d, yv_d, fs_d), (rows_t, ids_t, fs_t)

    delivered = {}
    choices = {}
    for name, members in (("primary", PRIMARY), ("secondary", SECONDARY)):
        pg, pv = fit_predict(dev, test, members)
        use_g = pg > pv
        delivered[name] = np.where(use_g, yg, yv)
        choices[name] = use_g.astype(int).tolist()

    id_set = set(ids_t)
    g_tok, g_inv = solo_tokens(artifacts, target["g"], id_set)
    v_tok, v_inv = solo_tokens(artifacts, target["v"], id_set)
    systems = {"G (Qwen3-4B alone)": yg, "V (Qwen3-8B alone)": yv}
    tokens = {"G (Qwen3-4B alone)": g_tok, "V (Qwen3-8B alone)": v_tok,
              "selector": g_tok + v_tok}
    gvs, fallbacks = {}, {}
    for key, label, base in (("gv_a", "GV-A (8B reviews 4B)", g_tok), ("gv_b", "GV-B (4B reviews 8B)", v_tok)):
        if target[key]:
            u, fb, rev_tok = gv_labels(artifacts, target[key], ids_t)
            gvs[key] = u
            systems[label] = u
            tokens[label] = base + rev_tok
            fallbacks[label] = fb

    report = {"dry_run": args.dry_run, "rows": len(ids_t), "excluded_ungradeable": sorted(listed & id_set),
              "dev_runs": DEV, "target_runs": target,
              "accuracy": {k: int(v.sum()) for k, v in systems.items()},
              "selector_primary": int(delivered["primary"].sum()),
              "selector_secondary": int(delivered["secondary"].sum()),
              "oracle_g_or_v": int(np.maximum(yg, yv).sum()),
              "output_tokens": tokens, "gv_fallback_rows": fallbacks,
              "extraction_timeouts": {"G": g_inv, "V": v_inv},
              "selector_choices": {"ids": ids_t, **choices}, "directions": {}}

    for key, label, reviewed, reviewed_label in (
            ("gv_a", "GV-A (8B reviews 4B)", yg, "G (Qwen3-4B alone)"),
            ("gv_b", "GV-B (4B reviews 8B)", yv, "V (Qwen3-8B alone)")):
        if key not in gvs:
            continue
        three = {"G (Qwen3-4B alone)": yg, "V (Qwen3-8B alone)": yv, label: gvs[key]}
        best = max(int(v.sum()) for v in three.values())
        tied = [k for k, v in three.items() if int(v.sum()) == best]
        direction = {"max_of_three": best, "comparators": tied, "selector": {}}
        for name in ("primary", "secondary"):
            results = {k: paired(delivered[name], three[k]) for k in tied}
            # With several systems tied at the maximum, the least favourable comparison counts.
            worst = min(results, key=lambda k: (results[k]["diff_rows"], -results[k]["p_sign"]))
            direction["selector"][name] = {"against": worst, **results[worst], "all_tied": results}
        other = yv if reviewed is yg else yg
        direction["gv_vs_reviewed_alone"] = paired(gvs[key], reviewed)
        direction["gv_vs_max_single"] = paired(gvs[key], yv if yv.sum() >= yg.sum() else yg)
        direction["anchoring"] = {"independent": joint(reviewed, other),
                                  "exposed": joint(reviewed, gvs[key])}
        report["directions"][label] = direction

    (out / "reports" / "test_mathfull_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# {'DRY RUN (development archives; numbers meaningless) - ' if args.dry_run else ''}"
             "Confirmatory mathfull test, frozen prompt v2", "",
             "Pre-registered in docs/plan/test_mathfull_v2.md; analysis code frozen before any test run.", "",
             f"- Rows {len(ids_t)}; excluded as ungradeable gold: {report['excluded_ungradeable']}",
             f"- Oracle (G or V right): {report['oracle_g_or_v']}", "",
             "|system|correct|accuracy|output tokens|", "|---|---:|---:|---:|"]
    for k, v in systems.items():
        lines.append(f"|{k}|{int(v.sum())}|{v.mean():.4f}|{tokens[k]:,}|")
    for name in ("primary", "secondary"):
        lines.append(f"|selector ({name})|{int(delivered[name].sum())}|{delivered[name].mean():.4f}|"
                     f"{tokens['selector']:,}|")
    for label, d in report["directions"].items():
        lines += ["", f"## {label}", "",
                  f"- max(G, V, GV) = {d['max_of_three']} ({', '.join(d['comparators'])})"]
        for name in ("primary", "secondary"):
            s = d["selector"][name]
            verdict = "significant" if s["p_sign"] < ALPHA else "not significant"
            lines.append(f"- selector ({name}) against {s['against']}: {s['x_only']}:{s['y_only']} discordant, "
                         f"diff {s['diff']:+.4f} (95% CI {s['ci95'][0]:+.4f} to {s['ci95'][1]:+.4f}), "
                         f"sign test p = {s['p_sign']:.4f} ({verdict})")
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
