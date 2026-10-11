#!/usr/bin/env python3
"""RQ2 cell summaries with the audited intervals (statistics audit, 2026-10-05). Server only.

`scripts/rq2_cells.py` derives every direction's cell counts from the committed claim JSONs, but
its intervals resample directions only, so they ignore which questions were sampled. This script
recomputes the same per-direction quantities from the results database, checks that each point
equals rq2_cells.py's, and gives every equal-domain mean an interval with the directions fixed
and one shared question resample per domain, within the fixed strata (claim_stats).

    python scripts/rq2_cells_bootstrap.py --db <results_db> --claim claims/H08_when_review_helps.yaml \
        --cells <new claims dir>/rq2_cells.json --out <new claims dir>/rq2_cells_audited.json

Writes only counts and statistics, so the output can be committed. Never overwrites.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from authority_allocation.claim_stats import Measured, bootstrap_statistic  # noqa: E402
from evaluate_claim import _directions  # noqa: E402

KINDS = ("stronger_reviews", "weaker_reviews")
KEYS = ("follow_g", "keep_rate", "adopt_rate", "gain_from_v", "loss_from_g", "net_both",
        "oracle_before", "oracle_after", "dC", "dis_before", "dis_after")


def columns(d: dict, stronger_is_g: bool) -> tuple[tuple, dict]:
    """Per-question 0/1 indicators from which every RQ2 quantity is a ratio of means."""
    keys = tuple(sorted(d))
    g = [d[k][0] for k in keys]
    v = [d[k][1] for k in keys]
    gv = [d[k][2] for k in keys]
    s = g if stronger_is_g else v
    z = list(zip(g, v, gv))
    # Only sparse columns (mostly zero), so the bootstrap stays fast: the best of the two
    # independent answers needs only the two one-sided counts, min(only_g, only_v).
    cols = {
        "gv_minus_s": [a - b for a, b in zip(gv, s)],
        "dis": [int(a != b) for a, b, _ in z],
        "follow": [int(a != b and c == a) for a, b, c in z],
        "only_g": [int(a == 1 and b == 0) for a, b, _ in z],
        "kept": [int(a == 1 and b == 0 and c == 1) for a, b, c in z],
        "only_v": [int(a == 0 and b == 1) for a, b, _ in z],
        "adopted": [int(a == 0 and b == 1 and c == 1) for a, b, c in z],
        "fixed": [int(a == b == 0 and c == 1) for a, b, c in z],
        "broken": [int(a == b == 1 and c == 0) for a, b, c in z],
        "dis_after": [int(a != c) for a, _, c in z],
    }
    return keys, cols


def quantities(r: dict) -> dict:
    """The rq2_cells.py quantities of one direction, from its column means (in pp)."""
    ratio = lambda a, b: None if r[b] == 0 else 100 * r[a] / r[b]
    # oracle_before = P(at least one right) - max(acc G, acc V) = min(only_g, only_v)
    oracle_before = min(r["only_g"], r["only_v"])
    return {"follow_g": ratio("follow", "dis"), "keep_rate": ratio("kept", "only_g"),
            "adopt_rate": ratio("adopted", "only_v"), "gain_from_v": r["adopted"],
            "loss_from_g": r["only_g"] - r["kept"], "net_both": r["fixed"] - r["broken"],
            "oracle_before": oracle_before,
            "oracle_after": oracle_before + r["fixed"] - (r["only_v"] - r["adopted"]),
            "dC": r["gv_minus_s"], "dis_before": r["dis"], "dis_after": r["dis_after"]}


def equal_domain_mean(rows: list[dict], key: str) -> float | None:
    per = {}
    for r in rows:
        if r[key] is not None:
            per.setdefault(r["domain"], []).append(r[key])
    return None if not per else sum(sum(v) / len(v) for v in per.values()) / len(per)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--claim", type=Path, default=ROOT / "claims/H08_when_review_helps.yaml")
    ap.add_argument("--cells", type=Path, required=True, help="rq2_cells.json of the same claims directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reports are never overwritten")
    card = yaml.safe_load(args.claim.read_text())
    boot = card.get("bootstrap", {})
    rows = [json.loads(line) for line in (args.db / "db/directions.jsonl").read_text().splitlines()]
    pairs = json.loads((args.db / "db/pairs.json").read_text())
    found, _ = _directions(rows, pairs, card["splits"], True, "all")
    items, kind_of = [], {}
    for domain, generator, reviewer, kind, p, d in found:
        keys, cols = columns(d, p["stronger"] == generator)
        uid = f"{domain}:{reviewer} 审 {generator}"
        items.append(Measured(uid, domain, keys, cols, models=(generator, reviewer)))
        kind_of[uid] = kind

    # Every point must equal rq2_cells.py's, which derives the same counts from the claim JSONs.
    old = {r["id"]: r for r in json.loads(args.cells.read_text())["directions"]}
    for it in items:
        mine = quantities(it.record())
        for key in KEYS:
            if key in old[it.id] and old[it.id][key] is not None and abs(mine[key] - old[it.id][key]) > 1e-6:
                raise ValueError(f"{it.id} {key}: {mine[key]} here but {old[it.id][key]} in rq2_cells.json")

    def statistic(records):
        out = {}
        per = [dict(id=r["id"], domain=r["domain"], **quantities(r)) for r in records]
        for kind in KINDS:
            sub = [r for r in per if kind_of[r["id"]] == kind]
            for key in KEYS:
                out[f"{kind}|{key}"] = equal_domain_mean(sub, key)
        return out

    reps, seed = boot.get("reps", 2000), boot.get("seed", 20261003)
    fixed = bootstrap_statistic(items, statistic, reps, seed, resample="fixed")
    by_pairs = bootstrap_statistic(items, statistic, reps, seed, resample="pairs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source_db": str(args.db), "claim": str(args.claim),
                                    "resample_primary": "fixed", "statistics": fixed["statistics"],
                                    "sensitivity_pairs": by_pairs["statistics"]},
                                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
