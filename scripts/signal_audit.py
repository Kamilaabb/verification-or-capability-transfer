#!/usr/bin/env python3
"""One explicit post-hoc pipeline for exploratory predictability comparisons.

Written because the numbers stopped agreeing with each other. They had been produced by
a series of ad-hoc scripts whose missing-value handling, cross-validation structure and
seed sets all differed slightly, and several of them were simply wrong: an implementation
without imputation let NaNs propagate into the weights whenever a feature set contained a
column with missing values, which silently destroyed the fit. Those results are void.

This script deliberately standardizes the exploratory comparisons below.  It borrows the
model family, preprocessing ingredients, C grid, outer-fold count and five split seeds from
c008, but it does *not* reproduce c008's registered estimator.  In particular:

* c008 selects one C jointly for three heads by mean binary log loss; this script selects C
  separately for each binary target by ROC AUC;
* c008 uses seeded GroupKFold; this script uses KFold because every retained row has one
  unique problem_id and each post-hoc target is fitted independently;
* stage 1 and stage 2 are not c008 targets and have no registered condition.

Consequently, the output is an internally comparable exploratory audit, not a correction or
reproduction of c008's registered 0.5799 edge AUC or 0.4026 policy accuracy.  Those formal
numbers remain authoritative for condition c008.  The shared ingredients are:

    conditions/c008/condition.yaml
      preprocessing  median imputation fit on the training fold only
                     explicit missing indicators for every registered feature
                     StandardScaler fit on the training fold only
      estimator      LogisticRegression, lbfgs, class_weight null, max_iter 2000
      selection      inner 4-fold per binary target, ROC-AUC criterion,
                     C in {0.01, 0.1, 1.0, 10.0}
      outer          5 folds, seeds 20260920..20260924

Three questions are asked of every feature set:

    stage 1   P(z0 = 0 | x)                  will the generator get this wrong
    stage 2   P(z2 = 1 | z0 = 0, x)          if it does, can the verifier recover it
    edge      rank the rows where z1 != z0   given M0 and M1 differ, which way

Stage 2 forms its folds over the z0 = 0 rows only. Edge trains two models on all rows and
scores p(z1) - p(z0), evaluated only where the two caps actually differ.

Feature availability is not cosmetic. Anything named v_* , gv_* or critique_* exists only
after the verifier has run, so a set containing them cannot gate the verifier call; it
describes the post-verifier setting, where the verifier's cost is already sunk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.01, 0.1, 1.0, 10.0]
OUTER_SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924]
OUTER_FOLDS = 5
INNER_FOLDS = 4

POST_V_PREFIXES = ("v_", "gv_", "critique_")

_BT = chr(96) * 3
FENCE = re.compile(_BT + r"(?:python)?\s*" + chr(92) + "n(.*?)" + _BT, re.DOTALL)


def trace_features(raw: str, trace: dict) -> list[float]:
    """Everything the archived per-token trace supports. All pre-verifier."""

    toks = (trace or {}).get("tokens") or []
    lp = np.array([t["logprob"] for t in toks], float)
    if lp.size == 0:
        return [float("nan")] * 16
    below = lp < -1
    run = best = 0
    for b in below:
        run = run + 1 if b else 0
        best = max(best, run)
    feats = [
        float(lp.mean()), float(lp.std()), float(lp.min()),
        float(np.quantile(lp, 0.10)), float(np.quantile(lp, 0.25)),
        float((lp < -1).mean()), float((lp < -2).mean()), float((lp < -3).mean()),
        float(best), float(int(lp.argmin()) / len(lp)), float(np.log1p(len(lp))),
        float(lp[-64:].mean()), float(lp[:32].mean()),
    ]
    match = FENCE.search(raw or "")
    base = (trace or {}).get("content_start_byte") or 0
    sel = np.array([], float)
    if match is not None:
        lo = base + len((raw[: match.start(1)]).encode("utf-8"))
        hi = base + len((raw[: match.end(1)]).encode("utf-8"))
        sel = np.array([t["logprob"] for t in toks
                        if t.get("end_byte", 0) > lo and t.get("start_byte", 0) < hi], float)
    if sel.size:
        feats += [float(sel.mean()), float(sel.min()), float(sel.size / len(lp))]
    else:
        feats += [float("nan")] * 3
    return feats


def gating_report(Xpre: np.ndarray, z0: np.ndarray, z2: np.ndarray) -> None:
    """Single-stage against two-stage, same pipeline, same folds, no leakage.

    Stage two is fitted on the training fold's z0 = 0 rows only and applied to the whole
    test fold, which is what a deployed gate would have to do.
    """

    n = len(z0)
    print()
    print("   pre-V gating: intervene with M2 on the top-k rows")
    print("   %-9s | %-22s | %-22s" % ("gate", "single  P(z0=0)", "two-stage  x P(z2|z0=0)"))
    print("   %-9s | %6s %6s %8s | %6s %6s %8s" % ("", "repair", "harm", "ratio",
                                                   "repair", "harm", "ratio"))
    scores = {"single": [], "two": []}
    for seed in OUTER_SEEDS:
        pw = np.zeros(n)
        pr = np.zeros(n)
        kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
        for tr, te in kf.split(Xpre):
            m1 = make_pipeline(); m1.fit(Xpre[tr], (1 - z0[tr]).astype(int))
            pw[te] = m1.predict_proba(Xpre[te])[:, 1]
            trs = tr[z0[tr] == 0]
            m2 = make_pipeline(); m2.fit(Xpre[trs], z2[trs].astype(int))
            pr[te] = m2.predict_proba(Xpre[te])[:, 1]
        scores["single"].append(pw)
        scores["two"].append(pw * pr)
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = int(n * frac)
        cells = []
        for mode in ("single", "two"):
            R, H = [], []
            for sc in scores[mode]:
                gate = np.zeros(n, bool); gate[np.argsort(-sc)[:k]] = True
                R.append(int((gate & (z0 == 0) & (z2 == 1)).sum()))
                H.append(int((gate & (z0 == 1) & (z2 == 0)).sum()))
            cells.append((np.mean(R), np.mean(H)))
        (r1, h1), (r2, h2) = cells
        print("   top %3d%%  | %6.1f %6.1f %8.2f | %6.1f %6.1f %8.2f" % (
            frac * 100, r1, h1, r1 / max(h1, 1e-9), r2, h2, r2 / max(h2, 1e-9)))
    rep = int(((z0 == 0) & (z2 == 1)).sum()); har = int(((z0 == 1) & (z2 == 0)).sum())
    print("   %-9s | %6d %6d %8.2f | %6s %6s %8s" % ("all rows", rep, har,
                                                     rep / max(har, 1), "-", "-", "-"))


def make_pipeline() -> GridSearchCV:
    pre = ColumnTransformer([
        ("impute", SimpleImputer(strategy="median"), slice(0, None)),
        ("missing", MissingIndicator(features="all"), slice(0, None)),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", class_weight=None, max_iter=2000)),
    ])
    return GridSearchCV(
        pipe, {"clf__C": C_GRID},
        cv=KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=0),
        scoring="roc_auc", n_jobs=1, refit=True,
    )


def auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    lab = label[order]
    n1 = lab.sum()
    n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(lab) + 1)
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def crossfit(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    pred = np.zeros(len(y))
    kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        if len(np.unique(y[tr])) < 2:
            pred[te] = float(y[tr].mean())
            continue
        model = make_pipeline()
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    return pred


def report(name: str, X: np.ndarray, z0: np.ndarray, z1: np.ndarray, z2: np.ndarray,
           post_v: bool) -> None:
    sub = z0 == 0
    edge = z1 != z0
    edge_label = (z1 > z0)[edge].astype(int)

    s1, s2, se = [], [], []
    for seed in OUTER_SEEDS:
        s1.append(auc(crossfit(X, (1 - z0).astype(int), seed), (1 - z0).astype(int)))
        p_sub = crossfit(X[sub], z2[sub].astype(int), seed)
        s2.append(auc(p_sub, z2[sub].astype(int)))
        score = crossfit(X, z1.astype(int), seed) - crossfit(X, z0.astype(int), seed)
        se.append(auc(score[edge], edge_label))
    flag = "post-V" if post_v else "pre-V "
    print("   %-52s %s  %.4f+-%.4f  %.4f+-%.4f  %.4f+-%.4f" % (
        name, flag,
        np.nanmean(s1), np.nanstd(s1),
        np.nanmean(s2), np.nanstd(s2),
        np.nanmean(se), np.nanstd(se)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--counterfactuals", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--generator", required=True)
    args = ap.parse_args()

    read = lambda p: [json.loads(l) for l in Path(p).open(encoding="utf-8")]
    ev = {r["problem_id"]: r for r in read(args.evidence)}
    gr = {r["problem_id"]: r for r in read(args.generator)}
    rows = [r for r in read(args.counterfactuals)
            if r["problem_id"] in ev and r["problem_id"] in gr
            and r["z0"] is not None and r["z1"] is not None and r["z2"] is not None]

    names = sorted(ev[rows[0]["problem_id"]]["features"])
    pre_v_names = [n for n in names if not n.startswith(POST_V_PREFIXES)]

    def registered(sel: list[str]) -> np.ndarray:
        return np.array([[ev[r["problem_id"]]["features"].get(n, np.nan) for n in sel]
                         for r in rows], float)

    def rebuilt() -> np.ndarray:
        out = []
        for r in rows:
            g = gr[r["problem_id"]]
            out.append([
                ev[r["problem_id"]]["features"].get("g_output_mean_logprob", np.nan),
                np.log1p(len(g["problem_text"])),
                np.log1p(len(g["outputs"][0]["raw_output"])),
            ])
        return np.array(out, float)

    z0 = np.array([r["z0"] for r in rows])
    z1 = np.array([r["z1"] for r in rows])
    z2 = np.array([r["z2"] for r in rows])

    print("=" * 110)
    print("%s   n=%d   acc_G=%.4f   z0=0 rows=%d   M0/M1 edge rows=%d (repair %d, harm %d)" % (
        args.label, len(rows), z0.mean(), int((z0 == 0).sum()), int((z1 != z0).sum()),
        int((z1 > z0).sum()), int((z1 < z0).sum())))
    print("registered features (%d): %s" % (len(names), ", ".join(names)))
    print("of which pre-V (%d): %s" % (len(pre_v_names), ", ".join(pre_v_names)))
    print()
    print("   %-52s %6s  %-16s %-16s %-16s" % (
        "feature set", "avail", "stage1 P(z0=0)", "stage2 P(z2|z0=0)", "edge M0/M1"))
    report("registered pre-V only (%d)" % len(pre_v_names),
           registered(pre_v_names), z0, z1, z2, post_v=False)
    report("rebuilt pre-V (3): g_out_lp, log|problem|, log|output|",
           rebuilt(), z0, z1, z2, post_v=False)
    traces = np.array([trace_features(gr[r["problem_id"]]["outputs"][0]["raw_output"],
                                      gr[r["problem_id"]]["outputs"][0].get("token_logprobs"))
                       for r in rows], float)
    report("registered pre-V + token trace (%d)" % (len(pre_v_names) + 16),
           np.c_[registered(pre_v_names), traces], z0, z1, z2, post_v=False)
    report("registered ALL (%d)" % len(names),
           registered(names), z0, z1, z2, post_v=True)

    if (z0 == 1).sum() and (z0 == 0).sum():
        gating_report(registered(pre_v_names), z0, z2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
