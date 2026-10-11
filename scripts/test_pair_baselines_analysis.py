#!/usr/bin/env python3
"""Selector against the development-chosen baselines on one mathfull test1500 pair.

The analysis of docs/plan/test_pair_baselines.md, frozen before the test GV runs of any pair it
is first applied to. A practitioner who cannot know the hindsight best chooses on development
data and deploys the choice unchanged:

- baseline 1: the most accurate of {V, G, GV-A, GV-B} on the pair's development rows;
- baseline 2: the more accurate of {V, G} on the development rows, which is V by construction.

Development ties go to the cheaper system, in the order V, G, GV-A, GV-B. GV-A is V reviewing G
(the stronger model reviews the weaker); GV-B is G reviewing V. The frozen selectors (members,
seeds and fit_predict of scripts/test_mathfull_v2_analysis.py, imported unchanged) are fitted
once on the pair's development solo runs and applied to its test solo runs. Test grades are read
only to score.

Primary family: the primary selector against baseline 1 and against baseline 2, each an exact
two-sided sign test on discordant rows, Holm-corrected at 0.05 over the distinct baselines (one
test when both baselines are V). Everything else is reported, not tested: the secondary selector,
the hindsight best of the four, each GV against max(G, V), anchoring, the 6-point dev-to-test
solo check, and cost as input plus output tokens.

--dry-run scores the development archives against themselves only to exercise the code; its
numbers mean nothing.
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
from test_mathfull_v2_analysis import ALPHA, PRIMARY, SECONDARY, fit_predict, gv_labels, joint, paired  # noqa: E402

# G is the model with the lower development accuracy. Each run is (development, test1500).
PAIRS = {
    "ministral-8b__qwen25-3b": {
        "g": ("Ministral-8B", "20260926T061450Z__c032__mathfull-dev1000__r012_solo4096",
              "20260928T130000Z__c096__mathfull-test1500__r012_solo4096"),
        "v": ("Qwen2.5-3B", "20260926T053834Z__c030__mathfull-dev1000__r012_solo4096",
              "20260927T053222Z__c057__mathfull-test1500__r012_solo4096"),
        "gv_a": ("20260928T190000Z__c150__mathfull-dev1000__r012_review",
                 "20260928T190000Z__c152__mathfull-test1500__r012_review"),
        "gv_b": ("20260928T190000Z__c151__mathfull-dev1000__r012_review",
                 "20260928T190000Z__c153__mathfull-test1500__r012_review"),
    },
    "qwen25-7b__qwen3-8b": {
        "g": ("Qwen2.5-7B", "20260926T060236Z__c031__mathfull-dev1000__r012_solo4096",
              "20260927T055111Z__c063__mathfull-test1500__r012_solo4096"),
        "v": ("Qwen3-8B", "20260925T172823Z__c026__mathfull-dev1000__r012_solo4096",
              "20260928T090000Z__c092__mathfull-test1500__r012_solo4096"),
        "gv_a": ("20260928T185500Z__c154__mathfull-dev1000__r012_review",
                 "20260928T185500Z__c156__mathfull-test1500__r012_review"),
        "gv_b": ("20260928T185500Z__c155__mathfull-dev1000__r012_review",
                 "20260928T185500Z__c157__mathfull-test1500__r012_review"),
    },
    # Already tested under docs/plan/test_qwen3_4b_qwen25_7b.md; listed so that this analysis can
    # be exercised end to end on complete archives. Its numbers here are post hoc.
    "qwen25-7b__qwen3-4b": {
        "g": ("Qwen2.5-7B", "20260926T060236Z__c031__mathfull-dev1000__r012_solo4096",
              "20260927T055111Z__c063__mathfull-test1500__r012_solo4096"),
        "v": ("Qwen3-4B", "20260925T172823Z__c025__mathfull-dev1000__r012_solo4096",
              "20260927T055111Z__c062__mathfull-test1500__r012_solo4096"),
        "gv_a": ("20260927T055111Z__c061__mathfull-dev1000__r012_review",
                 "20260927T055111Z__c065__mathfull-test1500__r012_review"),
        "gv_b": ("20260927T055111Z__c060__mathfull-dev1000__r012_review",
                 "20260927T055111Z__c064__mathfull-test1500__r012_review"),
    },
}
ORDER = ("V", "G", "GV-A", "GV-B")  # development tie-break: cheaper first
SHIFT_FLAG = 0.06  # |test - dev| solo accuracy above this flags the pair (docs/plan/test_qwen3_8b_mistral24b.md)
MARGIN = 0.01  # non-inferiority margin


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    rejected, still = {}, True
    for rank, (key, p) in enumerate(sorted(pvalues.items(), key=lambda kv: kv[1])):
        still = still and p < alpha / (len(pvalues) - rank)
        rejected[key] = still
    return rejected


def solo_usage(artifacts: Path, run: str, ids: list[str]) -> dict[str, int]:
    want, got = set(ids), {"input": 0, "output": 0, "rows": 0}
    for r in sel.read_jsonl(artifacts / run / "records" / "pass1.jsonl"):
        if r["problem_id"] in want:
            usage = r["outputs"][0]["usage"]
            got["input"] += int(usage.get("input_tokens") or 0)
            got["output"] += int(usage.get("output_tokens") or 0)
            got["rows"] += 1
    return got


def review_usage(artifacts: Path, run: str, ids: list[str]) -> dict[str, int]:
    want, got = set(ids), {"input": 0, "output": 0, "rows": 0}
    for r in sel.read_jsonl(artifacts / run / "records" / "review.jsonl"):
        if r["problem_id"] in want:
            usage = r["outputs"][0]["usage"]
            got["input"] += int(usage.get("input_tokens") or 0)
            got["output"] += int(usage.get("output_tokens") or 0)
            got["rows"] += 1
    return got


def add(*parts: dict[str, int]) -> dict[str, int]:
    return {"input": sum(p["input"] for p in parts), "output": sum(p["output"] for p in parts)}


def develop_choice(dev_correct: dict[str, int]) -> str:
    best = max(dev_correct.values())
    return next(name for name in ORDER if dev_correct[name] == best)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", required=True, choices=sorted(PAIRS))
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pair, artifacts, out = PAIRS[args.pair], Path(args.artifacts), Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    listed = {r["problem_id"] for r in yaml.safe_load(
        (ROOT / "datasets" / sel.DATASET / "ungradeable_gold.yaml").read_text(encoding="utf-8"))["ungradeable"]}
    manifests = ROOT / "data/manifests" / sel.DATASET
    g_name, v_name = pair["g"][0], pair["v"][0]
    t = 0 if args.dry_run else 1
    target_manifest = "dev1000" if args.dry_run else "test1500"
    runs = {"G": pair["g"][1:], "V": pair["v"][1:], "GV-A": pair["gv_a"], "GV-B": pair["gv_b"]}

    # Development: the selectors' training rows and the accuracies both baselines choose from.
    sel.G_RUN, sel.V_RUN = runs["G"][0], runs["V"][0]
    ids_d, rows_d, yg_d, yv_d = sel.build(artifacts, manifests / "dev1000.jsonl", listed)
    dev_y = {"G": yg_d, "V": yv_d,
             "GV-A": gv_labels(artifacts, runs["GV-A"][0], ids_d)[0],
             "GV-B": gv_labels(artifacts, runs["GV-B"][0], ids_d)[0]}
    dev_correct = {k: int(v.sum()) for k, v in dev_y.items()}
    if dev_correct["V"] < dev_correct["G"]:
        raise SystemExit("G must be the model with the lower development accuracy")
    baseline1, baseline2 = develop_choice(dev_correct), "V"

    # Target rows: test1500, or the development rows themselves in a dry run.
    sel.G_RUN, sel.V_RUN = runs["G"][t], runs["V"][t]
    ids_t, rows_t, yg, yv = sel.build(artifacts, manifests / f"{target_manifest}.jsonl", listed)
    y = {"G": yg, "V": yv}
    fallbacks = {}
    for key in ("GV-A", "GV-B"):
        y[key], fallbacks[key], _ = gv_labels(artifacts, runs[key][t], ids_t)
    dev = (rows_d, ids_d, yg_d, yv_d, sel.feature_sets(rows_d))
    test = (rows_t, ids_t, sel.feature_sets(rows_t))
    for name, members in (("selector", PRIMARY), ("selector (secondary)", SECONDARY)):
        pg, pv = fit_predict(dev, test, members)
        y[name] = np.where(pg > pv, yg, yv)

    usage_g, usage_v = solo_usage(artifacts, runs["G"][t], ids_t), solo_usage(artifacts, runs["V"][t], ids_t)
    cost = {"G": add(usage_g), "V": add(usage_v),
            "GV-A": add(usage_g, review_usage(artifacts, runs["GV-A"][t], ids_t)),
            "GV-B": add(usage_v, review_usage(artifacts, runs["GV-B"][t], ids_t)),
            "selector": add(usage_g, usage_v)}
    cost["selector (secondary)"] = cost["selector"]

    comparisons = {}
    for label, system in (("baseline 1", baseline1), ("baseline 2", baseline2)):
        for name in ("selector", "selector (secondary)"):
            comparisons[f"{name} vs {label} ({system})"] = paired(y[name], y[system])
    distinct = {"baseline 1": baseline1} if baseline1 == baseline2 else {"baseline 1": baseline1, "baseline 2": baseline2}
    ps = {}
    for label, system in distinct.items():
        c = comparisons[f"selector vs {label} ({system})"]
        ps[label] = c["p_sign"] if c["diff_rows"] > 0 else 1.0
    rejected = holm(ps)

    four = {k: int(y[k].sum()) for k in ORDER}
    hindsight = max(four.values())
    hindsight_systems = [k for k in ORDER if four[k] == hindsight]
    strong = "V" if four["V"] >= four["G"] else "G"
    shift = {name: {"dev": dev_correct[side] / len(ids_d), "test": float(y[side].mean())}
             for side, name in (("G", g_name), ("V", v_name))}
    for s in shift.values():
        s["flag"] = abs(s["test"] - s["dev"]) > SHIFT_FLAG

    report = {"pair": args.pair, "dry_run": args.dry_run, "g": g_name, "v": v_name, "runs": runs,
              "rows": {"dev": len(ids_d), "target": len(ids_t)}, "dev_correct": dev_correct,
              "baseline_1": baseline1, "baseline_2": baseline2,
              "target_correct": {k: int(v.sum()) for k, v in y.items()},
              "oracle_g_or_v": int(np.maximum(yg, yv).sum()),
              "hindsight_best": {"correct": hindsight, "systems": hindsight_systems},
              "comparisons": comparisons, "holm": {"p": ps, "rejected": rejected},
              "selector_vs_hindsight": {k: paired(y["selector"], y[k]) for k in hindsight_systems},
              "gv_vs_max_single": {k: paired(y[k], y[strong]) for k in ("GV-A", "GV-B")},
              "anchoring": {"GV-A": {"independent": joint(yg, yv), "exposed": joint(yg, y["GV-A"])},
                            "GV-B": {"independent": joint(yv, yg), "exposed": joint(yv, y["GV-B"])}},
              "solo_shift": shift, "tokens": cost, "gv_fallback_rows": fallbacks}
    (out / "reports" / "pair_baselines.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    n = len(ids_t)
    title = f"{g_name} (G) / {v_name} (V), {'DRY RUN on dev1000 - numbers meaningless' if args.dry_run else 'test1500'}"
    lines = [f"# Selector against development-chosen baselines: {title}", "",
             "Plan: docs/plan/test_pair_baselines.md. GV-A = V reviews G; GV-B = G reviews V.", "",
             f"- Rows: development {len(ids_d)}, scored {n}; oracle (G or V right) {report['oracle_g_or_v']}",
             f"- Development correct: " + ", ".join(f"{k} {v}" for k, v in dev_correct.items()),
             f"- Baseline 1 (best of four on development): **{baseline1}**; baseline 2 (better single model): **V**",
             "", "|system|correct|accuracy|input tokens|output tokens|total tokens|tokens per correct|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for k in ("G", "V", "GV-A", "GV-B", "selector", "selector (secondary)"):
        c, tok = int(y[k].sum()), cost[k]
        total = tok["input"] + tok["output"]
        lines.append(f"|{k}|{c}|{c / n:.4f}|{tok['input']:,}|{tok['output']:,}|{total:,}|{total / max(c, 1):,.0f}|")
    lines += ["", "## Primary: selector against the development-chosen baselines", "",
              "|comparison|selector only : other only|diff|95% CI|sign p|Holm|within 1pp non-inferiority|",
              "|---|---|---:|---|---:|---|---|"]
    for key, c in comparisons.items():
        label = next((b for b in distinct if f"vs {b} (" in key), None)
        holm_text = ("significant" if rejected[label] else "not significant") if key.startswith("selector vs") and label else "-"
        lines.append(f"|{key}|{c['x_only']}:{c['y_only']}|{100 * c['diff']:+.2f}pp|"
                     f"[{100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}]|{c['p_sign']:.4f}|{holm_text}|"
                     f"{'yes' if c['ci95'][0] >= -MARGIN else 'no'}|")
    lines += ["", "## Reported, not tested", "",
              f"- Hindsight best of the four on these rows: {hindsight} ({', '.join(hindsight_systems)})"]
    for k, c in report["selector_vs_hindsight"].items():
        lines.append(f"  - selector vs {k}: {c['x_only']}:{c['y_only']}, {100 * c['diff']:+.2f}pp, p = {c['p_sign']:.4f}")
    for k, c in report["gv_vs_max_single"].items():
        lines.append(f"- {k} vs max(G, V) = {strong}: {c['x_only']}:{c['y_only']}, {100 * c['diff']:+.2f}pp "
                     f"[{100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}], p = {c['p_sign']:.4f}")
    for name, s in shift.items():
        lines.append(f"- Solo shift {name}: dev {100 * s['dev']:.1f}% -> {100 * s['test']:.1f}% "
                     f"({100 * (s['test'] - s['dev']):+.1f}pp){' FLAGGED (> 6pp)' if s['flag'] else ''}")
    lines += ["", "|anchoring|G vs V independent|G vs GV-A|V vs G independent|V vs GV-B|", "|---|---:|---:|---:|---:|"]
    a = report["anchoring"]
    for field in ("disagree", "both_wrong", "union", "phi"):
        vals = [a["GV-A"]["independent"][field], a["GV-A"]["exposed"][field],
                a["GV-B"]["independent"][field], a["GV-B"]["exposed"][field]]
        lines.append(f"|{field}|" + "|".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals) + "|")
    lines += ["", f"- GV rows delivered from the reviewed model because the reviewer gave no answer: {fallbacks}", ""]
    (out / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
