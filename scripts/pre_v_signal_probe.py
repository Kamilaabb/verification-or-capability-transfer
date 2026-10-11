#!/usr/bin/env python3
"""Can the generator tell, before the verifier runs, that it got this one wrong?

Gate 2 left a sharp limit: of the 21 registered scalars only three exist before the
verifier is called, and on those three a cross-fit logistic reaches AUC 0.682 for
predicting z0. That is not a statement about pre-verifier signals in general; it is a
statement about a mean logprob, a program length and a problem length.

Every generator record carries the full per-token trace — 832 of 832 with status ok,
a few hundred tokens each with byte spans — and exactly one scalar was drawn from it.
This builds the rest of what that trace supports and re-measures. No new inference:
the archive already holds everything.

Why this and not a new training target. Relabelling with harm and cost weights changes
which action the label prefers, but the label is a deterministic function of (z0,z1,z2)
and the costs, so it adds no information about the row. The ceiling is set elsewhere:
harm can only occur where z0=1 and repair only where z0=0, so the whole repair/harm
tradeoff IS the z0 prediction problem, and a perfect z0 predictor would take 53 repairs
with zero harm. AUC on z0 is therefore the quantity that moves the ceiling.

Feature sets are nested and declared here, before running, so that reporting the best of
several is not an option. Engineering, not a research run: nothing is registered.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

BASELINE = ["g_output_mean_logprob", "g_code_lexemes_log", "problem_chars_log"]

TRACE = [
    "lp_mean", "lp_std", "lp_min", "lp_p10", "lp_p25",
    "frac_below_1", "frac_below_2", "frac_below_3",
    "worst_run_below_1", "min_position_rel", "n_tokens_log",
    "lp_mean_last_64", "lp_mean_first_32",
]

CODE_REGION = ["lp_mean_code", "lp_min_code", "code_frac_of_output"]

FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def trace_features(raw: str, trace: dict) -> dict[str, float]:
    toks = trace.get("tokens") or []
    lp = np.array([t["logprob"] for t in toks], float)
    out: dict[str, float] = {}
    if lp.size == 0:
        return {k: float("nan") for k in TRACE + CODE_REGION}

    out["lp_mean"] = float(lp.mean())
    out["lp_std"] = float(lp.std())
    out["lp_min"] = float(lp.min())
    out["lp_p10"] = float(np.quantile(lp, 0.10))
    out["lp_p25"] = float(np.quantile(lp, 0.25))
    for thr in (1, 2, 3):
        out[f"frac_below_{thr}"] = float((lp < -thr).mean())
    below = lp < -1
    run = best = 0
    for b in below:
        run = run + 1 if b else 0
        best = max(best, run)
    out["worst_run_below_1"] = float(best)
    out["min_position_rel"] = float(int(lp.argmin()) / len(lp))
    out["n_tokens_log"] = float(np.log1p(len(lp)))
    out["lp_mean_last_64"] = float(lp[-64:].mean())
    out["lp_mean_first_32"] = float(lp[:32].mean())

    # Restrict to the fenced program. Token byte spans share the coordinate system of
    # content_start_byte, and raw_output is that content, so the offsets found in
    # raw_output shift by content_start_byte. If the mapping selects nothing the
    # features are marked missing rather than guessed.
    match = FENCE.search(raw)
    base = trace.get("content_start_byte") or 0
    if match is None:
        for k in CODE_REGION:
            out[k] = float("nan")
        return out
    lo = base + len(raw[: match.start(1)].encode("utf-8"))
    hi = base + len(raw[: match.end(1)].encode("utf-8"))
    sel = np.array([
        t["logprob"] for t in toks
        if t.get("end_byte", 0) > lo and t.get("start_byte", 0) < hi
    ], float)
    if sel.size == 0:
        for k in CODE_REGION:
            out[k] = float("nan")
        return out
    out["lp_mean_code"] = float(sel.mean())
    out["lp_min_code"] = float(sel.min())
    out["code_frac_of_output"] = float(sel.size / len(lp))
    return out


def crossfit_auc(X: np.ndarray, y: np.ndarray, seeds: list[int]) -> tuple[float, float]:
    """Nested-free but grouped 5-fold cross-fit logistic, averaged over seeds."""

    aucs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        folds = np.array_split(idx, 5)
        pred = np.zeros(len(y))
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            mu = np.nanmean(X[tr], axis=0)
            Xtr = np.where(np.isnan(X[tr]), mu, X[tr])
            Xte = np.where(np.isnan(X[te]), mu, X[te])
            sd = Xtr.std(0)
            sd[sd == 0] = 1.0
            Ztr = (Xtr - Xtr.mean(0)) / sd
            Zte = (Xte - Xtr.mean(0)) / sd
            A = np.c_[np.ones(len(tr)), Ztr]
            w = np.zeros(A.shape[1])
            for _ in range(400):
                p = 1 / (1 + np.exp(-A @ w))
                grad = A.T @ (p - y[tr]) / len(tr) + 0.05 * w
                Hm = A.T @ (A * (p * (1 - p))[:, None]) / len(tr) + 0.05 * np.eye(len(w))
                w -= np.linalg.solve(Hm, grad)
            pred[te] = 1 / (1 + np.exp(-np.c_[np.ones(len(te)), Zte] @ w))
        aucs.append(auc(pred, y))
    return float(np.mean(aucs)), float(np.std(aucs))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    lab = label[order]
    n1 = lab.sum()
    n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(lab) + 1)
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate1-artifact", required=True)
    parser.add_argument("--gate2-artifact", required=True)
    args = parser.parse_args()

    a1 = Path(args.gate1_artifact)
    a2 = Path(args.gate2_artifact)
    ev = {r["problem_id"]: r for r in
          (json.loads(l) for l in (a2 / "records/controller_evidence.jsonl").open(encoding="utf-8"))}
    grec = {r["problem_id"]: r for r in
            (json.loads(l) for l in (a1 / "records/g.jsonl").open(encoding="utf-8"))}
    cf = [r for r in (json.loads(l) for l in
          (a1 / "records/counterfactuals.jsonl").open(encoding="utf-8"))
          if r["problem_id"] in ev and r["problem_id"] in grec]

    rows = []
    for r in cf:
        pid = r["problem_id"]
        out = grec[pid]["outputs"][0]
        feats = dict(ev[pid]["features"])
        feats.update(trace_features(out["raw_output"], out["token_logprobs"]))
        rows.append((pid, feats, r))

    y = np.array([1 - r["z0"] for _, _, r in rows])
    z0 = np.array([r["z0"] for _, _, r in rows])
    z2 = np.array([r["z2"] for _, _, r in rows])
    seeds = [20260920, 20260921, 20260922, 20260923, 20260924]

    print("predicting  the generator is wrong  (z0 = 0)")
    print("  n = %d, positives = %d" % (len(y), int(y.sum())))
    print()
    print("  %-46s %9s %8s" % ("nested feature set", "AUC", "sd"))
    sets = [
        ("baseline: the 3 registered pre-V scalars", BASELINE),
        ("+ token-trace statistics", BASELINE + TRACE),
        ("+ fenced-program region", BASELINE + TRACE + CODE_REGION),
    ]
    best = None
    for label, names in sets:
        X = np.array([[f.get(n, float("nan")) for n in names] for _, f, _ in rows], float)
        m, s = crossfit_auc(X, y, seeds)
        print("  %-46s %9.4f %8.4f   (%d features)" % (label, m, s, len(names)))
        best = (names, X)

    names, X = best
    rng = np.random.default_rng(seeds[0])
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 5)
    pred = np.zeros(len(y))
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        mu = np.nanmean(X[tr], axis=0)
        Xtr = np.where(np.isnan(X[tr]), mu, X[tr])
        Xte = np.where(np.isnan(X[te]), mu, X[te])
        sd = Xtr.std(0)
        sd[sd == 0] = 1.0
        A = np.c_[np.ones(len(tr)), (Xtr - Xtr.mean(0)) / sd]
        w = np.zeros(A.shape[1])
        for _ in range(400):
            p = 1 / (1 + np.exp(-A @ w))
            grad = A.T @ (p - y[tr]) / len(tr) + 0.05 * w
            Hm = A.T @ (A * (p * (1 - p))[:, None]) / len(tr) + 0.05 * np.eye(len(w))
            w -= np.linalg.solve(Hm, grad)
        pred[te] = 1 / (1 + np.exp(-np.c_[np.ones(len(te)), (Xte - Xtr.mean(0)) / sd] @ w))

    order = np.argsort(-pred)
    print()
    print("  gating the verifier call on that score, acting with M2 when gated in")
    print("  %-12s %6s %7s %6s %13s" % ("gate", "n", "repair", "harm", "repair/harm"))
    rep_all = int(((z0 == 0) & (z2 == 1)).sum())
    har_all = int(((z0 == 1) & (z2 == 0)).sum())
    print("  %-12s %6d %7d %6d %13.2f" % ("all rows", len(rows), rep_all, har_all, rep_all / har_all))
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        k = int(len(rows) * frac)
        gate = np.zeros(len(rows), bool)
        gate[order[:k]] = True
        rep = int((gate & (z0 == 0) & (z2 == 1)).sum())
        har = int((gate & (z0 == 1) & (z2 == 0)).sum())
        ratio = rep / har if har else float("inf")
        print("  %-12s %6d %7d %6d %13s" % ("top %d%%" % (frac * 100), k, rep, har,
                                            "inf" if har == 0 else "%.2f" % ratio))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
