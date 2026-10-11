#!/usr/bin/env python3
"""Pair selectors on MMLU-Pro test1500 for the stratified pairs (docs/plan/mmlu_pro_study.md sections 14-17).

Baseline 2 on test ("selector against max(G, V)"), run on 2026-09-29. Nothing on test is
used to fit, tune or choose anything: per pair, the screen's two pre-set selectors (section 14:
primary = mean of F2-A-l2 and F3-A-gbt, secondary = F2-A-l2) are refitted once on all of that pair's dev1000
solo archives per seed with the inner grid search, the five seeds' test probabilities are averaged, and the
averaged probabilities pick one model's answer per test row (a tie delivers V). V is the model with the higher
dev1000 solo accuracy, so the dev-chosen fixed baseline is V; the hindsight max(G, V) on test is reported too.
The same table carries, for reference, the test GV runs (registered primary delivery) and the JEV external
judge (section 17). No decision rule was set: paired comparisons give discordant rows, the difference with
a 95% interval and the exact two-sided sign-test p.

Extended 2026-10-01 from the first seven pairs to every pair with test GV runs: Mistral-24B x
Qwen2.5-7B, Qwen3-32B x Qwen2.5-3B, Mistral-24B x Llama-3.2-3B (section 16), the strong/strong cell Qwen3-32B x
Mistral-24B and Qwen2.5-72B x Mistral-24B, and Qwen3-4B x Qwen3-8B (sections 10-11). The method is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mc_pair_selector_dev1000 as mps  # noqa: E402
import mc_pair_selector_test1500 as t1  # noqa: E402
from test_mathfull_v2_analysis import paired  # noqa: E402

TEST_TAG = "mmlupro-test1500"
TEST_MANIFEST = ROOT / "data/manifests/mmlupro-rb189ec7-s20260927/test1500.jsonl"
DEV_TEST = {"c070": "c081", "c071": "c082", "c074": "c120", "c075": "c121", "c076": "c122", "c077": "c123",
            "c078": "c089", "c079": "c124", "c080": "c125"}
# id, (G name, G dev solo), (V name, V dev solo), test GV runs, JEV pair name
PAIRS = [
    ("qwen25-7b__qwen3-4b", ("Qwen2.5-7B", "c076"), ("Qwen3-4B", "c070"), ["c136", "c137"]),
    ("qwen25-7b__qwen3-8b", ("Qwen2.5-7B", "c076"), ("Qwen3-8B", "c071"), ["c138", "c139"]),
    ("ministral-8b__qwen25-3b", ("Ministral-8B", "c077"), ("Qwen2.5-3B", "c075"), ["c140", "c141"]),
    ("llama-32-3b__qwen25-3b", ("Llama-3.2-3B", "c074"), ("Qwen2.5-3B", "c075"), ["c142", "c143"]),
    ("qwen25-3b__qwen25-7b", ("Qwen2.5-3B", "c075"), ("Qwen2.5-7B", "c076"), ["c144", "c145"]),
    ("llama-32-3b__qwen3-8b", ("Llama-3.2-3B", "c074"), ("Qwen3-8B", "c071"), ["c146", "c147"]),
    ("qwen3-8b__mistral-24b", ("Qwen3-8B", "c071"), ("Mistral-Small-24B", "c078"), ["c176", "c177"]),
    ("qwen25-7b__mistral-24b", ("Qwen2.5-7B", "c076"), ("Mistral-Small-24B", "c078"), ["c178", "c179"]),
    ("qwen25-3b__qwen3-32b", ("Qwen2.5-3B", "c075"), ("Qwen3-32B", "c079"), ["c180", "c181"]),
    ("llama-32-3b__mistral-24b", ("Llama-3.2-3B", "c074"), ("Mistral-Small-24B", "c078"), ["c182", "c183"]),
    ("mistral-24b__qwen3-32b", ("Mistral-Small-24B", "c078"), ("Qwen3-32B", "c079"), ["c188", "c189"]),
    ("mistral-24b__qwen25-72b", ("Mistral-Small-24B", "c078"), ("Qwen2.5-72B", "c080"), ["c190", "c191"]),
    ("qwen3-4b__qwen3-8b", ("Qwen3-4B", "c070"), ("Qwen3-8B", "c071"), ["c083", "c084"]),
]


def jev_rows(artifacts: Path) -> dict[str, dict[str, bool]]:
    """pair -> problem_id -> JEV delivered V (only disagreement rows are present)."""
    hits = sorted(artifacts.glob("*__mc-jev-judge7__mmlupro-test1500__posthoc"))
    if not hits:
        return {}
    probs: dict[tuple[str, str], list[float]] = {}
    for r in mps.sel.read_jsonl(hits[-1] / "records" / "jev_responses.jsonl"):
        if "error" in r:
            continue
        p = r["response"]["answers"]["correct"]["probabilities"]
        probs.setdefault((r["pair"], r["problem_id"]), []).append(float(p["A"] if r["order"] == "VG" else p["B"]))
    out: dict[str, dict[str, bool]] = {}
    for (pair, pid), ps in probs.items():
        out.setdefault(pair, {})[pid] = len(ps) < 2 or sum(ps) / 2 >= 0.5
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    fsets = mps.feature_sets()
    jev = jev_rows(artifacts)
    report = []
    lines = ["# MMLU-Pro test1500: pair selectors (baseline 2), with GV and JEV for reference", "",
             "Per pair, the screen's pre-set selectors (primary = F2-A-l2 + F3-A-gbt, secondary = F2-A-l2) refitted "
             "on all of that pair's dev1000 rows, five seeds' test probabilities averaged, tie -> V. V = the model "
             "with the higher dev1000 solo accuracy (the dev-chosen fixed baseline). Nothing on test was used to fit, "
             "tune or choose. Comparisons: selector:baseline discordant rows, difference in rows, exact sign-test p.",
             "", "|pair|G|V|max(G,V) test|oracle|primary|primary vs V|secondary|secondary vs V|single-seed primary|"
             "GV runs|JEV|", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pid_, (gname, gdev), (vname, vdev), gvs in PAIRS:
        gtest, vtest = DEV_TEST[gdev], DEV_TEST[vdev]
        ids_d, rows_d, yg_d, yv_d = mps.build(artifacts, gdev, vdev)
        if yv_d.sum() < yg_d.sum():
            raise SystemExit(f"{pid_}: V must have the higher dev accuracy")
        ids_t, rows_t, yg_t, yv_t = mps.build(artifacts, gtest, vtest, TEST_MANIFEST, TEST_TAG)
        dev_data, test_data = (rows_d, ids_d, yg_d, yv_d), (rows_t, ids_t, yg_t, yv_t)
        entry = {"pair": pid_, "g": [gname, gdev, gtest, int(yg_t.sum())], "v": [vname, vdev, vtest, int(yv_t.sum())],
                 "max_gv_test": int(max(yg_t.sum(), yv_t.sum())), "oracle": int(np.maximum(yg_t, yv_t).sum()),
                 "selectors": {}, "gv": {}, "jev": None}
        cells = {}
        for name, members in mps.MEMBERS.items():
            pg, pv, per_seed = t1.fit_predict(dev_data, test_data, fsets, members)
            rows = np.where(pg > pv, yg_t, yv_t)
            cmp_v = paired(rows, yv_t)
            entry["selectors"][name] = {"correct": int(rows.sum()), "chose_g": int((pg > pv).sum()),
                                        "per_seed_fits": per_seed, "vs_v": cmp_v}
            cells[name] = (int(rows.sum()), cmp_v, per_seed)
            print(pid_, name, int(rows.sum()), per_seed, flush=True)
        gv_txt = []
        for c in gvs:
            run = mps.archive(artifacts, c, TEST_TAG)
            if run is None:
                gv_txt.append(f"{c} pending")
                continue
            u = t1.gv_u(run, ids_t)
            entry["gv"][c] = {"run": run.name, "correct": int(u.sum())}
            gv_txt.append(f"{c} {int(u.sum())}")
        jev_txt = "-"
        if pid_ in jev:
            picks = jev[pid_]
            uj = np.asarray([(yv_t[i] if picks.get(p, True) else yg_t[i]) for i, p in enumerate(ids_t)])
            entry["jev"] = {"correct": int(uj.sum()), "vs_v": paired(uj, yv_t)}
            jev_txt = str(int(uj.sum()))
        (pc, pcmp, pseeds), (sc, scmp, _) = cells["primary"], cells["secondary"]
        lines.append(f"|{pid_}|{gname} {int(yg_t.sum())}|{vname} {int(yv_t.sum())}|{entry['max_gv_test']}|"
                     f"{entry['oracle']}|{pc}|{pcmp['x_only']}:{pcmp['y_only']}, {pcmp['diff_rows']:+d}, "
                     f"p={pcmp['p_sign']:.3f}|{sc}|{scmp['x_only']}:{scmp['y_only']}, {scmp['diff_rows']:+d}, "
                     f"p={scmp['p_sign']:.3f}|{pseeds}|{'; '.join(gv_txt)}|{jev_txt}|")
        report.append(entry)
    lines.append("")
    (out / "reports" / "mc_pair_selector_test_pairs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
