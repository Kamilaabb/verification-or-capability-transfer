"""Reproduce the frozen code screen, retaining OOF decisions; verify before accepting.

Server only: no new generation or grading. Each task uses an unused output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
import code_pair_selector_dev832 as cps
from gv_domain_tables import Archives, paired
from test_mathfull_v2_analysis import paired as old_paired

OLD = ROOT / "artifacts/20260928T190457Z__code-pair-screen45__bcb-dev832__posthoc/reports/code_pair_screen.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_reports(ar):
    old = json.loads(ar.checked(OLD.parent.parent, "reports/code_pair_screen.json").read_text())
    registry = yaml.safe_load((ROOT / "pairs/bigcodebench_screen_dev832.yaml").read_text())
    if registry["manifest"] != "dev832" or len(old) != 45 or len(registry["pairs"]) != 45:
        raise ValueError("Expected frozen development-832 screen of 45 pairs")
    if len({r["pair"] for r in old}) != 45:
        raise ValueError("Duplicate archived pair")
    lookup = {r["pair"]: r for r in old}
    for p in registry["pairs"]:
        r = lookup[p["id"]]
        if (r["g"], r["v"]) != (p["g"]["condition"], p["v"]["condition"]):
            raise ValueError("Frozen pair roles changed")
    return [lookup[p["id"]] for p in registry["pairs"]]


def assignments(yg, yv):
    folds = []
    for seed in cps.sel.SEEDS:
        assignment = np.full(len(yg), -1, dtype=int)
        for fold, (_, heldout) in enumerate(StratifiedKFold(cps.sel.OUTER, shuffle=True,
                random_state=seed).split(np.zeros((len(yg), 1)), 2 * yg + yv)):
            assignment[heldout] = fold
        if np.any(assignment < 0):
            raise ValueError("Missing heldout fold")
        folds.append(assignment)
    return np.asarray(folds)


def compare(old, new):
    fields = {"domain": "code", "split": "dev", "n": 832, "pair": old["pair"],
              "g_condition": old["g"], "v_condition": old["v"],
              "runs": [old["g_run"], old["v_run"]], "G": old["g_correct"],
              "V": old["v_correct"], "oracle": old["oracle"],
              "seeds": cps.sel.SEEDS, "outer_folds": cps.sel.OUTER, "inner_folds": cps.sel.INNER}
    for field, expected in fields.items():
        if new[field] != expected:
            raise ValueError(f"{old['pair']}: {field} differs from frozen summary")
    for kind in cps.MEMBERS:
        a, b = new["selectors"][kind], old["selector"][kind]
        for key, old_key in (("correct", "seed_mean_prob"), ("per_seed", "per_seed"), ("mean", "mean"),
                             ("vs_max_gv", "vs_max_gv")):
            if a[key] != b[old_key]:
                raise ValueError(f"{old['pair']} {kind}: {key} differs from frozen summary")


def validate_decisions(path, ids, yg, yv, folds):
    predictions, seeded_g, seeded_v = [], [], []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            r = json.loads(line)
            if i >= len(ids) or set(r) != {"problem_id", "p_g", "p_v", "pick_v", "per_seed"} or r["problem_id"] != ids[i]:
                raise ValueError("Unexpected schema, labels, duplicate or reordered membership")
            if len(r["per_seed"]) != len(cps.sel.SEEDS):
                raise ValueError("Missing seed predictions")
            for j, seed in enumerate(cps.sel.SEEDS):
                p = r["per_seed"][j]
                if set(p) != {"seed", "fold", "p_g", "p_v"} or p["seed"] != seed or p["fold"] != int(folds[j, i]):
                    raise ValueError("Seed or heldout assignment changed")
            for p in [r] + r["per_seed"]:
                for name in ("p_g", "p_v"):
                    if type(p[name]) not in (int, float) or not math.isfinite(p[name]) or not 0 <= p[name] <= 1:
                        raise ValueError("Invalid OOF probability")
            for name in ("p_g", "p_v"):
                if r[name] != float(np.mean([p[name] for p in r["per_seed"]])):
                    raise ValueError("Aggregate probability differs from seed mean")
            if type(r["pick_v"]) is not bool or r["pick_v"] != (r["p_v"] >= r["p_g"]):
                raise ValueError("V tie-break changed")
            predictions.append(yv[i] if r["pick_v"] else yg[i])
            seeded_g.append([p["p_g"] for p in r["per_seed"]])
            seeded_v.append([p["p_v"] for p in r["per_seed"]])
    if len(predictions) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Incomplete or duplicate membership")
    pg, pv = np.asarray(seeded_g).T, np.asarray(seeded_v).T
    return np.asarray(predictions), [int(np.where(g > v, yg, yv).sum()) for g, v in zip(pg, pv)]


def fit(index, root):
    ar = Archives(ROOT / "artifacts")
    old = old_reports(ar)[index]
    out = root / f"code__{old['g']}__{old['v']}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "records").mkdir()
    (out / "reports").mkdir()
    try:
        runs = [cps.archive(ar.root, old[s]) for s in ("g", "v")]
        if [p.name for p in runs] != [old["g_run"], old["v_run"]]:
            raise ValueError("Upstream archive changed; do not fit")
        for run in runs:
            for name in ("condition.resolved.yaml", "records/grades.jsonl", "records/pass1.jsonl",
                         "records/native_logit_trace.jsonl", "records/scalar_evidence.jsonl"):
                ar.checked(run, name)
        cps.G_COND, cps.V_COND = old["g"], old["v"]
        ids, rows, yg, yv = cps.build(ar.root)
        if len(ids) != 832 or len(set(ids)) != 832 or yv.sum() < yg.sum():
            raise ValueError("Membership or stronger-side assignment changed")
        for run, y in zip(runs, (yg, yv)):
            grades = ar.grades(run)
            if set(grades) != set(ids) or [grades[p] for p in ids] != y.tolist():
                raise ValueError("Builder and checked grades disagree")
        folds = assignments(yg, yv)
        report = dict(domain="code", split="dev", evaluation="nested cross-fit; out-of-fold",
            pair=old["pair"], g_condition=old["g"], v_condition=old["v"],
            g=ar.config(runs[0])["model"]["model_id"], v=ar.config(runs[1])["model"]["model_id"],
            runs=[p.name for p in runs], n=len(ids), G=int(yg.sum()), V=int(yv.sum()),
            oracle=int(np.maximum(yg, yv).sum()), seeds=cps.sel.SEEDS,
            outer_folds=cps.sel.OUTER, inner_folds=cps.sel.INNER,
            manifest_sha256=digest(cps.MANIFEST), selectors={})
        for kind, members in cps.MEMBERS.items():
            d = cps.ensemble(rows, ids, yg, yv, cps.feature_sets(), members, retain_predictions=True)
            path = out / f"records/{kind}_decisions.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for i, pid in enumerate(ids):
                    r = dict(problem_id=pid, p_g=float(d["p_g"][i]), p_v=float(d["p_v"][i]),
                        pick_v=bool(d["p_v"][i] >= d["p_g"][i]), per_seed=[dict(seed=seed,
                        fold=int(folds[j, i]), p_g=float(d["seed_p_g"][j, i]), p_v=float(d["seed_p_v"][j, i]))
                        for j, seed in enumerate(cps.sel.SEEDS)])
                    handle.write(json.dumps(r) + "\n")
            yy, per_seed = validate_decisions(path, ids, yg, yv, folds)
            if not np.array_equal(yy, d["rows"]) or per_seed != d["per_seed"]:
                raise ValueError("Saved decisions differ from original ensemble")
            report["selectors"][kind] = dict(correct=int(yy.sum()), per_seed=per_seed,
                mean=d["mean"], vs_baseline2=paired(yy.tolist(), yv.tolist()), vs_max_gv=old_paired(yy, yv))
            print(f"{old['pair']} {kind}: {int(yy.sum())}, seeds={per_seed}", flush=True)
        compare(old, report)
        report["source_sha256"] = ar.sources
        report["script_sha256"] = {p.name: digest(p) for p in (Path(__file__), Path(cps.__file__), Path(cps.sel.__file__))}
        (out / "reports/selector_dev.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (out / "reports/summary.md").write_text(f"# Code development OOF backfill: PASSED\n\n{old['pair']}; 832 rows.\n\n"
            "Original features, ensembles, five seeds, 5 outer / 5 inner folds and V tie-break retained. "
            "Both delivered counts, all per-seed counts and paired contrasts exactly reproduce the immutable old summary.\n",
            encoding="utf-8")
    except Exception as exc:
        (out / "reports/summary.md").write_text(f"# Code backfill FAILED\n\n{exc}\n\nStop; do not tune to align.\n", encoding="utf-8")
        raise


def verify(root, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "reports").mkdir()
    ar = Archives(ROOT / "artifacts")
    olds = old_reports(ar)
    results = []
    try:
        if len(list(root.glob("code__*/reports/selector_dev.json"))) != 45:
            raise ValueError("Incomplete batch: expected 45 finished reports")
        for old in olds:
            pair_root = root / f"code__{old['g']}__{old['v']}"
            report = json.loads((pair_root / "reports/selector_dev.json").read_text())
            compare(old, report)
            grades = [ar.grades(ar.root / r) for r in report["runs"]]
            # Archived solo order equals the manifest/build order; recover from decisions,
            # then verify complete membership and derive the original stratified fold map.
            path = pair_root / "records/primary_decisions.jsonl"
            ids = [json.loads(line)["problem_id"] for line in path.read_text().splitlines()]
            if len(ids) != 832 or any(set(ids) != set(g) for g in grades):
                raise ValueError("Decision membership differs from upstream grades")
            yg, yv = [np.asarray([g[p] for p in ids]) for g in grades]
            folds, files = assignments(yg, yv), {}
            for kind in cps.MEMBERS:
                path = pair_root / f"records/{kind}_decisions.jsonl"
                yy, seeds = validate_decisions(path, ids, yg, yv, folds)
                expected = old["selector"][kind]
                if int(yy.sum()) != expected["seed_mean_prob"] or seeds != expected["per_seed"] or old_paired(yy, yv) != expected["vs_max_gv"]:
                    raise ValueError("Recorded decisions disagree with frozen summary")
                files[kind] = dict(path=str(path), sha256=digest(path), correct=int(yy.sum()),
                    old_correct=expected["seed_mean_prob"], per_seed=seeds, old_per_seed=expected["per_seed"])
            if not (pair_root / "reports/summary.md").is_file():
                raise ValueError("Missing human report")
            results.append(dict(pair=old["pair"], n=832, files=files, runs=report["runs"], source_sha256=report["source_sha256"]))
    except Exception as exc:
        (out / "reports/summary.md").write_text(f"# Code backfill FAILED\n\n{exc}\n\nStop; do not tune to align.\n", encoding="utf-8")
        raise
    result = dict(status="passed", pairs=results, old_summary=str(OLD), old_summary_sha256=digest(OLD), script_sha256=digest(Path(__file__)))
    (out / "reports/validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Code selector backfill: PASSED", "", "45 development-832 pairs; 90 files and 74,880 OOF decisions.",
        "Both delivered counts, all 450 per-seed counts and paired contrasts exactly match the immutable old summary.",
        "Schema, membership, finite probabilities, seed means, heldout folds and V tie-break all pass.",
        "No new generation or grading. Development cross-fit results; not test-set evidence.", "",
        "|pair|primary old/new|secondary old/new|", "|---|---:|---:|"]
    for r in results:
        a, b = r["files"]["primary"], r["files"]["secondary"]
        lines.append(f"|{r['pair']}|{a['old_correct']}/{a['correct']}|{b['old_correct']}/{b['correct']}|")
    (out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("PASSED: 45 pairs, 90 files; exact delivered and seed counts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verify-root", type=Path)
    args = parser.parse_args()
    if args.verify_root is not None:
        verify(args.verify_root, args.out_dir)
    elif args.index is not None and 0 <= args.index < 45:
        fit(args.index, args.out_dir)
    else:
        parser.error("Specify --index 0..44 or --verify-root")


if __name__ == "__main__":
    main()
