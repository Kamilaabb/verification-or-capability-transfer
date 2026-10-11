"""Evaluate a cost claim card (H10, H11) on the cost rows of build_cost_rows.py (server only).

    python scripts/evaluate_cost_claims.py --cost /path/to/scratch/cost/<date> \
        --claim claims/H10_second_model_cost.yaml --out-dir /path/to/scratch/claims/<date>

For each comparison a − b of the card, over the design pairs: the accuracy difference with the same
bootstrap as H01–H08 (pairs fixed, questions resampled within strata, domains weighted equally;
pair resampling as appendix sensitivity), each pair's paired 95% interval, how many pairs
are above, level with, or below, and how many times as much a costs as b. Estimates only: the card
decides nothing (2026-10-04). Writes <id>_result.json and <id>_summary.md; never overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from authority_allocation.claim_stats import Unit, paired_unit, two_level_bootstrap  # noqa: E402
from authority_allocation.cost_model import OUTPUT_WEIGHTS, system_cost  # noqa: E402

LABEL = {"weaker_alone": "较弱者单独", "stronger_alone": "较强者单独", "review_by_stronger": "较强者审较弱者",
         "review_by_weaker": "较弱者审较强者", "selector_primary": "选择器", "selector_secondary": "选择器（次方案）",
         "JEV": "JEV（参照）", "oracle": "上限（至少一个对）"}


def domain_mean(values: dict[str, list[float]]) -> float:
    return sum(sum(v) / len(v) for v in values.values()) / len(values)


def evaluate(rows: list[dict], card: dict) -> dict:
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r["domain"], r["pair_id"])].append(r)
    boot = card.get("bootstrap") or {}
    out = []
    for comp in card["comparisons"]:
        a, b = comp["a"], comp["b"]
        units, pairs = [], []
        ratios = {w: defaultdict(list) for w in OUTPUT_WEIGHTS}
        for (domain, pair_id), rs in sorted(by_pair.items()):
            unit = Unit(id=f"{domain}:{pair_id}", domain=domain, a=[r["correct"][a] for r in rs],
                        b=[r["correct"][b] for r in rs], models=tuple(pair_id.split("|")),
                        keys=tuple(r["problem_id"] for r in rs))
            units.append(unit)
            pu = paired_unit(unit, reps=boot.get("reps", 2000), seed=boot.get("seed", 20261003))
            lo, hi = pu["ci95_pp"]
            cost = {}
            for w, wv in OUTPUT_WEIGHTS.items():
                ca = sum(system_cost(r["tokens"], a, r["params_b"], wv) for r in rs)
                cb = sum(system_cost(r["tokens"], b, r["params_b"], wv) for r in rs)
                cost[w] = ca / cb
                ratios[w][domain].append(ca / cb)
            pairs.append({"id": unit.id, "domain": domain, "weaker": rs[0]["weaker"], "stronger": rs[0]["stronger"],
                          "n": unit.n, "diff_pp": pu["diff_pp"], "ci95_pp": [lo, hi],
                          "side": "above" if lo > 0 else "below" if hi < 0 else "level", "cost_ratio": cost})
        reps, seed = boot.get("reps", 2000), boot.get("seed", 20261003)
        bs = two_level_bootstrap(units, reps=reps, seed=seed, resample="fixed")
        bs_pairs = two_level_bootstrap(units, reps=reps, seed=seed, resample="pairs")  # appendix sensitivity
        for part in [bs["overall"], *bs["per_domain"].values(), bs_pairs["overall"], *bs_pairs["per_domain"].values()]:
            part.pop("p_two_sided", None)  # estimates only
        out.append({"a": a, "b": b, "label": comp.get("label"), "bootstrap": bs, "bootstrap_pairs": bs_pairs,
                    "counts": {s: sum(p["side"] == s for p in pairs) for s in ("above", "level", "below")},
                    "cost_ratio": {w: {"overall": domain_mean(v), "per_domain": {d: sum(x) / len(x) for d, x in v.items()}}
                                   for w, v in ratios.items()},
                    "pairs": pairs})
    return {"id": card["id"], "analysis": "cost", "units": "design pairs", "comparisons": out}


def summary(result: dict, card: dict) -> str:
    lines = [f"# {card['id']}：{card['statement_zh']}", "",
             "只估计、不检验：差是四个域等权的平均（百分点；配对固定、按题重抽的 bootstrap 95% 区间）；"
             "高于 / 持平 / 低于按每对自己的配对 95% 区间数；成本倍数 = a 每题成本 / b 每题成本，四个域等权平均。", "",
             "|a − b|差（pp）|95% 区间|高于 / 持平 / 低于|成本倍数|成本倍数（输出×4）|", "|---|---:|---|---|---:|---:|"]
    for c in result["comparisons"]:
        o = c["bootstrap"]["overall"]
        n = c["counts"]
        lines.append(f"|{LABEL.get(c['a'], c['a'])} − {LABEL.get(c['b'], c['b'])}|{o['mean_diff_pp']:+.2f}|"
                     f"[{o['ci95_pp'][0]:+.2f}, {o['ci95_pp'][1]:+.2f}]|{n['above']} / {n['level']} / {n['below']}|"
                     f"{c['cost_ratio']['main']['overall']:.2f}|{c['cost_ratio']['output_x4']['overall']:.2f}|")
    lines += ["", "## 逐域", "", "|a − b|域|差（pp）|95% 区间|成本倍数|", "|---|---|---:|---|---:|"]
    for c in result["comparisons"]:
        for d, v in c["bootstrap"]["per_domain"].items():
            lines.append(f"|{LABEL.get(c['a'], c['a'])} − {LABEL.get(c['b'], c['b'])}|{d}|{v['mean_diff_pp']:+.2f}|"
                         f"[{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}]|{c['cost_ratio']['main']['per_domain'][d]:.2f}|")
    lines += ["", f"写卡片时看过什么：{json.dumps(card.get('seen_when_written'), ensure_ascii=False)}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost", type=Path, required=True, help="output directory of build_cost_rows.py")
    ap.add_argument("--claim", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    card = yaml.safe_load(args.claim.read_text(encoding="utf-8"))
    if card.get("analysis") != "cost":
        sys.exit(f"{args.claim}: not a cost card")
    target = args.out_dir / f"{card['id']}_result.json"
    if target.exists():
        sys.exit(f"{target} exists; completed reports are never overwritten")
    path = args.cost / "db/cost_rows.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate(rows, card)
    result["inputs"] = {"cost_rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "card_sha256": hashlib.sha256(args.claim.read_bytes()).hexdigest(),
                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / f"{card['id']}_summary.md").write_text(summary(result, card), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
