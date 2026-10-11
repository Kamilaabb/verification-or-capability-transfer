#!/usr/bin/env python3
"""Cross-fit G/V selectors on mathfull dev1000, as pre-registered in docs/plan/selector_dev1000.md.

Unregistered development analysis over a G and a V archive: c022/c023 (prompt v1) by default,
c025/c026 (frozen prompt v2) through --g-run/--v-run, where the configuration selected on v1,
F2-A-l2, is refitted and the other fifteen are reference only (docs/plan/gv_vs_max_g_v_study.md
section 4.7). Features are
computed offline from the archived outputs, native traces and scalar evidence; the gold
answer, the grades and the problem id never enter a feature. Labels are the archived u.
Nothing is written into either archive.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c002_dev30 as base  # noqa: E402
from authority_allocation.math_expression_spans import content_token_indices  # noqa: E402

G_RUN = "20260924T203442Z__c022__mathfull-dev1000__r012_solo4096"
V_RUN = "20260924T203442Z__c023__mathfull-dev1000__r012_solo4096"
DATASET = "mathfull-r465bcdb-l3to5-s20260924"
SEEDS = [20260925, 20260926, 20260927, 20260928, 20260929]
OUTER, INNER = 5, 5
LOW = math.log(0.5)
BUDGET = 4096
LEARNING_CURVE_N = [250, 500, 750, 999]
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
BOXED = "\\boxed{"

ORIG7 = ["math_expression_mean_max_logit", "math_expression_mean_logprob", "answer_parseable",
         "problem_char_count", "output_char_count", "output_mean_max_logit", "output_mean_logprob"]
SUBJECTS = ["Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra",
            "Number Theory", "Prealgebra", "Precalculus"]
PROBLEM = ["level"] + [f"subject_{i}" for i in range(len(SUBJECTS))]
DIFF_KEYS = ["out_lp_mean", "out_lp_min", "out_ent_mean", "out_margin_mean", "out_lowconf_frac",
             "math_lp_mean", "math_lp_min", "math_ent_mean", "math_margin_mean",
             "math_lowconf_frac", "ans_lp_mean", "ans_ent_mean", "n_output_tokens"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stats(prefix: str, lp: list[float], ent: list[float], mar: list[float]) -> dict:
    f: dict[str, float | None] = {}
    have = bool(lp)
    f[f"{prefix}_lp_mean"] = float(np.mean(lp)) if have else None
    f[f"{prefix}_lp_min"] = float(np.min(lp)) if have else None
    f[f"{prefix}_lp_p10"] = float(np.percentile(lp, 10)) if have else None
    f[f"{prefix}_lp_std"] = float(np.std(lp)) if have else None
    f[f"{prefix}_ent_mean"] = float(np.mean(ent)) if have else None
    f[f"{prefix}_ent_p90"] = float(np.percentile(ent, 90)) if have else None
    f[f"{prefix}_ent_max"] = float(np.max(ent)) if have else None
    f[f"{prefix}_margin_mean"] = float(np.mean(mar)) if have else None
    f[f"{prefix}_margin_p10"] = float(np.percentile(mar, 10)) if have else None
    low = [x < LOW for x in lp]
    f[f"{prefix}_lowconf_frac"] = float(np.mean(low)) if have else None
    run = best = 0
    for flag in low:
        run = run + 1 if flag else 0
        best = max(best, run)
    f[f"{prefix}_lowconf_run"] = float(best) if have else None
    return f


def boxed_char_span(text: str) -> tuple[int, int] | None:
    start = text.rfind(BOXED)
    if start < 0:
        return None
    depth, i = 0, start + len(BOXED) - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1
    return None


def answer_tokens(raw: str, trace: dict) -> list[int] | None:
    span = boxed_char_span(raw)
    if span is None or trace.get("status") != "ok":
        return None
    offset = int(trace.get("content_start_byte") or 0)
    b0 = offset + len(raw[:span[0]].encode("utf-8"))
    b1 = offset + len(raw[:span[1]].encode("utf-8"))
    return [int(t["index"]) for t in trace.get("tokens", [])
            if t["end_byte"] > b0 and t["start_byte"] < b1]


def answer_kind(text: str | None) -> str | None:
    if text is None:
        return None
    body = text.strip()
    if re.fullmatch(r"-?\d+", body):
        return "integer"
    if re.fullmatch(r"-?\d*\.\d+|-?\\d?frac\{-?\d+\}\{\d+\}|-?\d+/\d+", body):
        return "fraction"
    return "other"


def model_features(evidence: dict, pass1: dict, native: dict) -> tuple[dict, dict]:
    out = pass1["outputs"][0]
    raw = out["raw_output"] or ""
    trace = out.get("token_logprobs") or {}
    by_index = {int(p["index"]): p for p in native.get("positions", [])}
    f: dict[str, float | None] = {name: evidence["features"][f"g_{name}"] if f"g_{name}" in evidence["features"]
                                  else evidence["features"][name] for name in ORIG7}
    spans = {"out": content_token_indices(trace),
             "math": [int(i) for i in evidence["math_token_indices"]],
             "ans": answer_tokens(raw, trace) or []}
    for name, idx in spans.items():
        pos = [by_index[i] for i in idx if i in by_index]
        if len(pos) != len(idx):
            pos = []
        f.update(_stats(name, [p["sampled_logprob"] for p in pos], [p["entropy"] for p in pos],
                        [p["top1_top2_margin"] for p in pos]))
    tokens = int(out["usage"].get("output_tokens") or 0)
    f["n_output_tokens"] = float(tokens)
    f["truncated"] = float(tokens >= BUDGET)
    f["n_boxed"] = float(raw.count("\\boxed"))
    lower = raw.lower()
    f["n_selfcorrect"] = float(sum(lower.count(w) for w in ("wait", "however", "actually")))
    span = boxed_char_span(raw)
    answer = raw[span[0] + len(BOXED):span[1] - 1] if span else None
    exprs = {re.sub(r"\s+", "", raw[s["start"]:s["end"]]) for s in evidence["math_spans"]}
    return f, {"answer": answer, "answer_text": base.answer_text(out), "exprs": exprs}


def build(artifacts: Path, manifest: Path, excluded: set[str]) -> tuple[list[str], dict, np.ndarray, np.ndarray]:
    order = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = {r["problem_id"]: r for r in order}
    runs = {}
    for side, run in (("g", G_RUN), ("v", V_RUN)):
        rec = artifacts / run / "records"
        runs[side] = {name: {r["problem_id"]: r for r in read_jsonl(rec / f"{name}.jsonl")}
                      for name in ("pass1", "native_logit_trace", "scalar_evidence", "grades")}
    ids = [r["problem_id"] for r in order if r["problem_id"] not in excluded]
    rows, y_g, y_v = {}, [], []
    for pid in ids:
        row: dict[str, float | None] = {}
        extra = {}
        for side in ("g", "v"):
            r = runs[side]
            feats, extra[side] = model_features(r["scalar_evidence"][pid], r["pass1"][pid],
                                                r["native_logit_trace"][pid])
            row.update({f"{side}_{k}": v for k, v in feats.items()})
        row["level"] = float(meta[pid]["level"])
        for i, subject in enumerate(SUBJECTS):
            row[f"subject_{i}"] = float(meta[pid]["subject"] == subject)
        gt, vt = extra["g"]["answer_text"], extra["v"]["answer_text"]
        try:
            eq = None if gt is None or vt is None else base.math_equivalent(gt, vt)[0]
        except (Exception, base.MathVerifyTimeout):   # math_verify's timeout is a BaseException
            eq = None
        row["gv_equivalent"] = None if eq is None else float(eq)
        for key in DIFF_KEYS:
            g, v = row[f"g_{key}"], row[f"v_{key}"]
            row[f"diff_{key}"] = None if g is None or v is None else v - g
        row["xor_parseable"] = float(row["g_answer_parseable"] != row["v_answer_parseable"])
        row["xor_truncated"] = float(row["g_truncated"] != row["v_truncated"])
        eg, ev = extra["g"]["exprs"], extra["v"]["exprs"]
        row["expr_jaccard"] = len(eg & ev) / len(eg | ev) if eg | ev else None
        ag, av = extra["g"]["answer"], extra["v"]["answer"]
        row["diff_answer_len"] = None if ag is None or av is None else float(len(av) - len(ag))
        kg, kv = answer_kind(ag), answer_kind(av)
        row["same_answer_kind"] = None if kg is None or kv is None else float(kg == kv)
        rows[pid] = row
        y_g.append(int(runs["g"]["grades"][pid]["u"]))
        y_v.append(int(runs["v"]["grades"][pid]["u"]))
    return ids, rows, np.asarray(y_g), np.asarray(y_v)


def feature_sets(rows: dict) -> dict[str, dict[str, list[str]]]:
    sample = next(iter(rows.values()))
    single = lambda side: sorted(k for k in sample if k.startswith(f"{side}_")) + PROBLEM
    cross = sorted(k for k in sample if k.startswith(("diff_", "xor_")) or k in
                   ("gv_equivalent", "expr_jaccard", "same_answer_kind"))
    both = sorted(set(single("g")) | set(single("v"))) + cross
    return {"F1": {"g": [f"g_{n}" for n in ORIG7], "v": [f"v_{n}" for n in ORIG7]},
            "F2": {"g": single("g"), "v": single("v")},
            "F3": {"g": both, "v": both}}


def matrix(rows: dict, ids: list[str], names: list[str]) -> np.ndarray:
    return np.asarray([[np.nan if rows[p][n] is None else float(rows[p][n]) for n in names] for p in ids])


def search(family: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    impute = ("impute", SimpleImputer(strategy="median", add_indicator=True))
    if family == "l2":
        pipe = Pipeline([impute, ("scale", StandardScaler()), ("m", LogisticRegression(max_iter=5000))])
        grid = {"m__C": [0.01, 0.1, 1, 10]}
    elif family == "tree":
        pipe = Pipeline([impute, ("m", DecisionTreeClassifier(random_state=seed))])
        grid = {"m__max_depth": [2, 3, 4, 6], "m__min_samples_leaf": [5, 20]}
    elif family == "forest":
        pipe = Pipeline([impute, ("m", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                              random_state=seed, n_jobs=1))])
        grid = {"m__max_depth": [3, 6, None], "m__min_samples_leaf": [5, 20]}
    elif family == "gbt":
        pipe = Pipeline([("m", HistGradientBoostingClassifier(random_state=seed))])
        grid = {"m__max_depth": [2, 3], "m__learning_rate": [0.05, 0.1], "m__max_iter": [100, 300]}
    else:
        raise ValueError(family)
    return GridSearchCV(pipe, grid, scoring="neg_log_loss", n_jobs=N_JOBS,
                        cv=StratifiedKFold(INNER, shuffle=True, random_state=seed))


def crossfit(x_g, x_v, y_g, y_v, family: str, formulation: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold (p_g, p_v) for formulation A; for B, p_v = P(V right | disagreement), p_g = 1 - p_v."""
    from sklearn.model_selection import StratifiedKFold

    strata = 2 * y_g + y_v
    p_g, p_v = np.full(len(y_g), np.nan), np.full(len(y_g), np.nan)
    for train, test in StratifiedKFold(OUTER, shuffle=True, random_state=seed).split(x_g, strata):
        if formulation == "A":
            for x, y, out in ((x_g, y_g, p_g), (x_v, y_v, p_v)):
                s = search(family, seed)
                s.fit(x[train], y[train])
                out[test] = s.predict_proba(x[test])[:, 1]
        else:
            disagree = train[y_g[train] != y_v[train]]
            s = search(family, seed)
            s.fit(x_v[disagree], y_v[disagree])
            p_v[test] = s.predict_proba(x_v[test])[:, 1]
            p_g[test] = 1 - p_v[test]
    return p_g, p_v


def deliver(p_g, p_v, y_g, y_v) -> dict:
    use_g = p_g > p_v                         # ties deliver V, the max(G, V) side
    delivered = np.where(use_g, y_g, y_v)
    return {"correct": int(delivered.sum()), "chose_g": int(use_g.sum()),
            "repair_vs_v": int((use_g & (y_g == 1) & (y_v == 0)).sum()),
            "harm_vs_v": int((use_g & (y_g == 0) & (y_v == 1)).sum())}


def evaluate(rows, ids, y_g, y_v, fsets, config) -> dict:
    fs, formulation, family = config
    names = fsets[fs]
    x_g, x_v = matrix(rows, ids, names["g"]), matrix(rows, ids, names["v"])
    per_seed, pg_all, pv_all = [], [], []
    for seed in SEEDS:
        pg, pv = crossfit(x_g, x_v, y_g, y_v, family, formulation, seed)
        per_seed.append(deliver(pg, pv, y_g, y_v)); pg_all.append(pg); pv_all.append(pv)
    mean_pg, mean_pv = np.mean(pg_all, axis=0), np.mean(pv_all, axis=0)
    mean_prob = deliver(mean_pg, mean_pv, y_g, y_v)
    counts = [s["correct"] for s in per_seed]
    return {"config": f"{fs}-{formulation}-{family}", "n": len(ids),
            "features_per_head": len(names["g"]), "per_seed": per_seed,
            "per_seed_correct": counts, "mean_correct": float(np.mean(counts)),
            "seed_mean_prob": mean_prob,
            # Row-aligned with the report's ids: 1 where the seed-mean probabilities deliver G.
            "seed_mean_use_g": (mean_pg > mean_pv).astype(int).tolist(),
            "g": int(y_g.sum()), "v": int(y_v.sum()),
            "max_gv": int(max(y_g.sum(), y_v.sum())), "oracle": int(np.maximum(y_g, y_v).sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--g-run", default=G_RUN, help="G archive (default: c022, prompt v1)")
    parser.add_argument("--v-run", default=V_RUN, help="V archive (default: c023, prompt v1)")
    parser.add_argument("--skip-curve", action="store_true",
                        help="only the 16 main configurations; no learning curve")
    args = parser.parse_args()
    globals()["G_RUN"], globals()["V_RUN"] = args.g_run, args.v_run
    out = Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data/manifests" / DATASET / "dev1000.jsonl"
    import yaml
    listed = {r["problem_id"] for r in yaml.safe_load(
        (ROOT / "datasets" / DATASET / "ungradeable_gold.yaml").read_text(encoding="utf-8"))["ungradeable"]}
    ids, rows, y_g, y_v = build(Path(args.artifacts), manifest, listed)
    fsets = feature_sets(rows)
    missing = {n: sum(rows[p][n] is None for p in ids) for n in fsets["F3"]["g"]}
    print(json.dumps({"rows": len(ids), "G": int(y_g.sum()), "V": int(y_v.sum()),
                      "features_F3": len(fsets["F3"]["g"])}), flush=True)

    families = ["l2", "tree", "forest", "gbt"]
    configs = [(fs, "A", fam) for fs in ("F1", "F2", "F3") for fam in families] + \
              [("F3", "B", fam) for fam in families]
    main_results = []
    for config in configs:
        t0 = time.time()
        res = evaluate(rows, ids, y_g, y_v, fsets, config)
        main_results.append(res)
        with (out / "reports" / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(res) + "\n")
        print(f"{res['config']:<14} seeds {res['per_seed_correct']} mean {res['mean_correct']:.1f} "
              f"seed-mean-prob {res['seed_mean_prob']['correct']}  ({time.time() - t0:.0f}s)", flush=True)

    curve = []
    for n in ([] if args.skip_curve else LEARNING_CURVE_N):
        sub = ids[:n]
        idx = [ids.index(p) for p in sub]
        for formulation in ("A", "B"):
            for fam in families:
                res = evaluate(rows, sub, y_g[idx], y_v[idx], fsets, ("F3", formulation, fam))
                res["gain_vs_max_pct"] = 100 * (res["mean_correct"] - res["max_gv"]) / n
                curve.append(res)
                with (out / "reports" / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(res) + "\n")
                print(f"curve n={n:<4} {res['config']:<14} gain_vs_max {res['gain_vs_max_pct']:+.2f} pt",
                      flush=True)

    tree_fams = {"tree", "forest", "gbt"}
    def gain(n, cfg):
        return next(r["gain_vs_max_pct"] for r in curve if r["n"] == n and r["config"] == cfg)
    rises = {r["config"]: gain(999, r["config"]) - gain(750, r["config"])
             for r in curve if r["n"] == 999 and r["config"].split("-")[2] in tree_fams}
    expand = any(v >= 0.5 for v in rises.values()) if curve else None

    best = max(main_results, key=lambda r: r["mean_correct"])
    report = {"g_run": G_RUN, "v_run": V_RUN, "ids": ids,
              "rows": len(ids), "g": int(y_g.sum()), "v": int(y_v.sum()),
              "max_gv": int(max(y_g.sum(), y_v.sum())), "oracle": int(np.maximum(y_g, y_v).sum()),
              "feature_missing_F3": missing, "main": main_results, "learning_curve": curve,
              "tree_gain_rise_750_to_999_pct": rises, "expand_to_dev2000": expand,
              "best_by_mean_correct": best["config"]}
    (out / "reports" / "selector_dev1000.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Selector development on mathfull dev1000", "",
             "Pre-registered in docs/plan/selector_dev1000.md. Unregistered development analysis; "
             "the sealed test is untouched.", "",
             f"- G archive {G_RUN}; V archive {V_RUN}",
             f"- Rows {len(ids)}; G {int(y_g.sum())}; V {int(y_v.sum())}; "
             f"max(G, V) {report['max_gv']}; oracle {report['oracle']}", "",
             "|config|features/head|per-seed correct|mean|vs max(G,V)|seed-mean-prob|repair/harm vs V|",
             "|---|---:|---|---:|---:|---:|---|"]
    for r in main_results:
        sm = r["seed_mean_prob"]
        lines.append(f"|{r['config']}|{r['features_per_head']}|{r['per_seed_correct']}|{r['mean_correct']:.1f}|"
                     f"{r['mean_correct'] - r['max_gv']:+.1f}|{sm['correct']}|{sm['repair_vs_v']}/{sm['harm_vs_v']}|")
    if curve:
        lines += ["", "## Learning curve (F3, gain of mean correct over max(G, V), points)", "",
                  "|config|" + "|".join(f"n={n}" for n in LEARNING_CURVE_N) + "|",
                  "|---|" + "---:|" * len(LEARNING_CURVE_N)]
        for cfg in sorted({r["config"] for r in curve}):
            lines.append(f"|{cfg}|" + "|".join(f"{gain(n, cfg):+.2f}" for n in LEARNING_CURVE_N) + "|")
        lines += ["", f"- Tree-family rise from n=750 to n=999 (points): "
                  f"{ {k: round(v, 2) for k, v in rises.items()} }",
                  f"- Pre-registered expansion rule (any rise >= 0.5): "
                  f"**{'expand to dev2000' if expand else 'stay at dev1000'}**"]
    else:
        lines += ["", "- Learning curve skipped (--skip-curve)."]
    lines += [f"- Best configuration by mean per-seed correct: {best['config']} "
              f"({best['mean_correct']:.1f} vs max(G, V) {best['max_gv']})", ""]
    (out / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
