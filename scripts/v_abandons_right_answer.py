"""Where does the reviewer abandon its own right answer? On questions only V answers correctly alone (G wrong, V right),
in the paper's 96 review directions, relate whether the final answer stays right to features read from the two solo
answers: V's confidence alone (mean token log-probability and mean raw vocabulary-max logit of its solo output),
V's and G's output lengths, the same two confidence measures for G, whether G's output hit the token limit; and per
direction, the domain, who reviews whom and the gap. Features are compared within each direction: raw logits are on
different scales for different models.

Associations only: the questions are selected because V got them right alone, and nothing here is manipulated.
Runs on the server (results_db/20261007 and the solo archives); writes aggregates only."""
import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("math", "knowledge", "code", "nl_reasoning")
FEATURES = ("v_conf", "v_len", "v_maxlogit", "g_conf", "g_len", "g_maxlogit", "g_truncated")
CONTINUOUS = FEATURES[:-1]


def archive(cond):
    hits = sorted(glob.glob(str(ROOT / "artifacts" / f"*__{cond}__*")))
    assert len(hits) == 1, (cond, hits)
    return Path(hits[0])


def solo_features(cond, wanted):
    """problem_id -> (mean token logprob, output tokens, hit the limit) for the solo answers in `wanted`."""
    out = {}
    with (archive(cond) / "records" / "pass1.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            pid = str(r["problem_id"])
            if pid not in wanted:
                continue
            o = r["outputs"][0]
            toks = o["token_logprobs"]["tokens"]
            lp = [t["logprob"] for t in toks if t.get("logprob") is not None]
            out[pid] = [sum(lp) / len(lp) if lp else float("nan"), o["usage"]["output_tokens"],
                        int(o["usage"].get("finish_reason") == "length"), float("nan")]
    # mean raw pre-softmax vocabulary-max logit over the output positions (same-forward native trace)
    with (archive(cond) / "records" / "native_logit_trace.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            pid = str(r["problem_id"])
            if pid in out and r.get("status") == "ok" and r["positions"]:
                out[pid][3] = sum(p["max_raw_logit"] for p in r["positions"]) / len(r["positions"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("/path/to/scratch/results_db/20261007/db"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairs = json.loads((args.db / "pairs.json").read_text())
    direction = {}
    for dom in DOMAINS:
        chosen = [p for p in pairs[dom] if p["in_design_sample"]]
        assert len(chosen) == 12
        for p in chosen:
            for gkey, conds in p["directions"].items():
                g = gkey.split("=", 1)[1]
                v = next(m for m in p["models"] if m != g)
                kind = "stronger_reviews" if v == p["stronger"] else "weaker_reviews"
                for split, cond in conds.items():
                    direction[cond] = (dom, kind, f"{dom}|{g}|{v}", g, v, split)

    solo_cond = defaultdict(set)
    for line in (args.db / "solo.jsonl").open():
        r = json.loads(line)
        solo_cond[r["domain"], r["model"], r["split"]].add(r["condition"])
    assert all(len(c) == 1 for c in solo_cond.values())
    solo_cond = {k: c.pop() for k, c in solo_cond.items()}

    rows, acc = [], defaultdict(lambda: [0, 0, 0])  # direction -> [n, g right, v right]
    for line in (args.db / "directions.jsonl").open():
        r = json.loads(line)
        d = direction.get(r["condition"])
        if d is None:
            continue
        a = acc[d[2]]
        a[0] += 1; a[1] += r["g"]; a[2] += r["v"]
        if r["g"] == 0 and r["v"] == 1:
            rows.append({"dom": d[0], "kind": d[1], "dir": d[2], "g": d[3], "v": d[4], "split": d[5],
                         "v_cond": r["v_condition"], "pid": str(r["problem_id"]), "kept": r["gv"]})
    assert len(acc) == 96, len(acc)
    gap = {k: 100 * abs(a[1] - a[2]) / a[0] for k, a in acc.items()}

    need = defaultdict(set)
    for x in rows:
        need[x["v_cond"]].add(x["pid"])
        need[solo_cond[x["dom"], x["g"], x["split"]]].add(x["pid"])
    print(f"{len(rows)} questions; reading {len(need)} solo archives", file=sys.stderr, flush=True)
    feats = {}
    for c, ids in need.items():
        feats[c] = solo_features(c, ids)
        print("read", c, file=sys.stderr, flush=True)
    for x in rows:
        vf = feats[x["v_cond"]][x["pid"]]
        gf = feats[solo_cond[x["dom"], x["g"], x["split"]]][x["pid"]]
        x.update(v_conf=vf[0], v_len=math.log(vf[1] + 1), v_maxlogit=vf[3], g_conf=gf[0], g_len=math.log(gf[1] + 1),
                 g_maxlogit=gf[3], g_truncated=gf[2], gap=gap[x["dir"]])
    rows = [x for x in rows if all(math.isfinite(x[f]) for f in FEATURES)]

    result = {"source": str(args.db), "questions": len(rows), "unit": "questions only V got right alone"}
    # kept rate by domain and kind (ratio of sums)
    tab = defaultdict(lambda: [0, 0])
    for x in rows:
        for key in ((x["dom"], x["kind"]), ("all", x["kind"])):
            tab[key][0] += 1; tab[key][1] += x["kept"]
    result["kept_rate"] = {f"{k[0]} {k[1]}": {"n": n, "kept": 100 * s / n} for k, (n, s) in sorted(tab.items())}

    # quartiles of each feature, defined within each direction (so models with different logit scales are not
    # mixed), pooled by kind
    by_dir = defaultdict(list)
    for x in rows:
        by_dir[x["dir"]].append(x)
    quart = {}
    for f in CONTINUOUS:
        bins = defaultdict(lambda: [0, 0])
        for sub in by_dir.values():
            cuts = np.quantile([x[f] for x in sub], [0.25, 0.5, 0.75])
            for x in sub:
                q = int(np.searchsorted(cuts, x[f], side="right"))
                bins[x["kind"], q][0] += 1; bins[x["kind"], q][1] += x["kept"]
        quart[f] = {f"{k} Q{q + 1}": round(100 * s / n, 1) for (k, q), (n, s) in sorted(bins.items())}
    trunc = defaultdict(lambda: [0, 0])
    for x in rows:
        trunc[x["kind"], x["g_truncated"]][0] += 1; trunc[x["kind"], x["g_truncated"]][1] += x["kept"]
    quart["g_truncated"] = {f"{k} {'hit limit' if t else 'finished'}": [n, round(100 * s / n, 1)]
                            for (k, t), (n, s) in sorted(trunc.items())}
    result["kept_by_quartile_within_direction"] = quart
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    # logistic regression with direction fixed effects; features standardised within each direction, so a
    # coefficient compares questions of the same direction; 95% intervals from cluster-robust (sandwich)
    # standard errors, clusters = (domain, problem)
    def design(sub):
        stats = {}
        for d in {x["dir"] for x in sub}:
            ds = [x for x in sub if x["dir"] == d]
            stats[d] = {f: (np.mean([x[f] for x in ds]), np.std([x[f] for x in ds]) or 1.0) for f in FEATURES}
        dirs = sorted({x["dir"] for x in sub})
        idx = {d: i for i, d in enumerate(dirs)}
        X = np.zeros((len(sub), len(FEATURES) + len(dirs)))
        for i, x in enumerate(sub):
            for j, f in enumerate(FEATURES):
                m, s = stats[x["dir"]][f]
                X[i, j] = (x[f] - m) / s
            X[i, len(FEATURES) + idx[x["dir"]]] = 1.0
        return X, np.array([x["kept"] for x in sub])

    regs = {}
    for kind in ("stronger_reviews", "weaker_reviews", "all"):
        sub = [x for x in rows if kind == "all" or x["kind"] == kind]
        X, y = design(sub)
        b = LogisticRegression(C=1e6, fit_intercept=False, max_iter=5000).fit(X, y).coef_[0]
        p = 1 / (1 + np.exp(-X @ b))
        bread = np.linalg.pinv((X * (p * (1 - p))[:, None]).T @ X)
        score = X * (y - p)[:, None]
        cid = {}
        groups = np.array([cid.setdefault((x["dom"], x["pid"]), len(cid)) for x in sub])
        S = np.zeros((len(cid), X.shape[1]))
        np.add.at(S, groups, score)
        V = bread @ (S.T @ S) @ bread
        se = np.sqrt(np.diag(V))
        regs[kind] = {f: {"odds_ratio_per_sd": float(np.exp(b[j])),
                          "ci95": [float(np.exp(b[j] - 1.96 * se[j])), float(np.exp(b[j] + 1.96 * se[j]))]}
                      for j, f in enumerate(FEATURES)} | {"n": len(sub), "clusters": len(cid)}
        result["logistic_direction_fixed_effects"] = regs
        args.out.write_text(json.dumps(result, indent=1) + "\n")

    # per direction: kept rate against the gap
    per = defaultdict(lambda: [0, 0])
    for x in rows:
        per[x["dir"], x["kind"]][0] += 1; per[x["dir"], x["kind"]][1] += x["kept"]
    pts = defaultdict(list)
    for (d, k), (n, s) in per.items():
        pts[k].append((gap[d], 100 * s / n))
    result["kept_vs_gap_per_direction"] = {
        k: {"directions": len(v), "slope_per_10pt_gap": float(10 * np.polyfit([a for a, _ in v], [b for _, b in v], 1)[0]),
            "corr": float(np.corrcoef([a for a, _ in v], [b for _, b in v])[0, 1])} for k, v in pts.items()}

    text = json.dumps(result, indent=1)
    args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
