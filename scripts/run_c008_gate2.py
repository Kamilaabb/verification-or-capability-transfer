#!/usr/bin/env python3
"""Gate 2 development cross-fit for the registered c008 scalar controller.

This runner never reads the held-out test manifest.  It reduces the saved c007
action-preceding G/V records to registered scalars, joins labels only in a separate
target file, performs nested grouped cross-fitting, and fits the frozen development
checkpoint for a later confirmatory condition.
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata as importlib_metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.cap_outcomes import build_cap_outcome_target  # noqa: E402
from authority_allocation.code_scalar_features import FEATURE_NAMES, build_code_scalar_payload  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.controller_training import (  # noqa: E402
    CAPS,
    choose_action,
    explicit_missing_matrix,
    make_pipeline,
    select_c,
)
from authority_allocation.records import read_jsonl  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json,
    sha256_bytes,
    sha256_file,
    stable_id,
    utc_now,
)

CONDITION_PATH = ROOT / "conditions/c008/condition.yaml"
FEATURE_CONTRACT = ROOT / "protocols/authority_v6/features_code_scalar_v1.yaml"
CONDITION_ID = "c008"
DATASET_ID = "bigcodebench-b74c0d0-instruct-s20260917"
REPLICATE_PLAN_ID = "r006"
EXPECTED_ROWS = 832


def load_condition() -> dict[str, Any]:
    return resolve_condition(CONDITION_PATH, repository_root=ROOT)


def _referenced_files(value: Any) -> set[Path]:
    found: set[Path] = set()
    if isinstance(value, dict):
        for child in value.values():
            found.update(_referenced_files(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_referenced_files(child))
    elif isinstance(value, str) and value.endswith((".yaml", ".json", ".txt")):
        path = (ROOT / value).resolve()
        if path.is_file() and ROOT.resolve() in path.parents:
            found.add(path)
    return found


def condition_hash() -> str:
    condition = load_condition()
    excluded = set(condition["hash_policy"]["exclude"])
    behavioural = {key: value for key, value in condition.items() if key not in excluded}
    files = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(_referenced_files(condition))
    }
    return sha256_bytes(
        canonical_json({"condition": behavioural, "referenced_file_sha256": files}).encode()
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable stage output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = str(row["problem_id"])
        if problem_id in result:
            raise ValueError(f"duplicate problem_id: {problem_id}")
        result[problem_id] = row
    return result


def _require_same_832(*indices: dict[str, dict[str, Any]]) -> list[str]:
    sets = [set(item) for item in indices]
    if not sets or any(item != sets[0] for item in sets[1:]):
        raise ValueError("input problem sets differ")
    if len(sets[0]) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} development rows, found {len(sets[0])}")
    return sorted(sets[0])


def stage_evidence(args: argparse.Namespace) -> None:
    g_rows = index(read_jsonl(Path(args.g_records)))
    v_rows = index(read_jsonl(Path(args.v_records)))
    problem_ids = _require_same_832(g_rows, v_rows)
    current_hash = condition_hash()
    contract_hash = sha256_file(FEATURE_CONTRACT)

    def records():
        for problem_id in problem_ids:
            g_row, v_row = g_rows[problem_id], v_rows[problem_id]
            y0 = g_row["outputs"][0]
            payload = build_code_scalar_payload(
                problem_text=g_row["problem_text"],
                g_text=y0["raw_output"],
                g_trace=y0["token_logprobs"],
                v_text=v_row["raw_output"],
                v_trace=v_row["token_logprobs"],
                v_report=v_row["v_report"],
                v_report_parse_status=v_row["report_parse_status"],
                registered_request=v_row["registered_request"],
            )
            evidence_sha = sha256_bytes(canonical_json(payload).encode())
            yield {
                "schema_version": "controller-evidence-code-scalar-v1",
                "record_id": stable_id(args.run_id, problem_id, "evidence"),
                "run_id": args.run_id,
                "source_g_run_id": g_row["run_id"],
                "source_v_run_id": v_row["run_id"],
                "condition_id": CONDITION_ID,
                "condition_hash": current_hash,
                "replicate_id": "e01",
                "stage": "gate2",
                "cell_id": v_row["cell_id"],
                "problem_id": problem_id,
                "feature_set_id": "code_scalar_v1",
                "feature_contract_sha256": contract_hash,
                **payload,
                "evidence_sha256": evidence_sha,
                "created_at_utc": utc_now(),
            }

    print(canonical_json({"evidence_rows": write_jsonl(Path(args.output), records())}))


def stage_targets(args: argparse.Namespace) -> None:
    evidence = index(read_jsonl(Path(args.evidence_records)))
    counterfactuals = index(read_jsonl(Path(args.counterfactual_records)))
    problem_ids = _require_same_832(evidence, counterfactuals)

    def records():
        for problem_id in problem_ids:
            source, outcome = evidence[problem_id], counterfactuals[problem_id]
            target = build_cap_outcome_target(outcome["z0"], outcome["z1"], outcome["z2"])
            yield {
                "schema_version": "controller-training-target-v1",
                "problem_id": problem_id,
                "cell_id": source["cell_id"],
                "source_evidence_record_id": source["record_id"],
                "evidence_sha256": source["evidence_sha256"],
                "cap_outcomes": target["cap_outcomes"],
                "oracle_actions": target["oracle_actions"],
                "minimal_oracle_action": target["minimal_oracle_action"],
            }

    print(canonical_json({"target_rows": write_jsonl(Path(args.output), records())}))


def _training_arrays(evidence_path: Path, target_path: Path):
    evidence = index(read_jsonl(evidence_path))
    targets = index(read_jsonl(target_path))
    problem_ids = _require_same_832(evidence, targets)
    feature_rows = [evidence[problem_id]["features"] for problem_id in problem_ids]
    y = np.asarray(
        [[targets[p]["cap_outcomes"][cap] for cap in CAPS] for p in problem_ids],
        dtype=int,
    )
    return problem_ids, evidence, feature_rows, y


def stage_crossfit(args: argparse.Namespace) -> None:
    from sklearn.model_selection import GroupKFold

    condition = load_condition()["controller"]
    groups_config = condition["cumulative_feature_ablations"]
    training = condition["training"]
    problem_ids, evidence, feature_rows, y = _training_arrays(
        Path(args.evidence_records), Path(args.target_records)
    )
    group_ids = np.asarray(problem_ids)
    rows: list[dict[str, Any]] = []
    for ablation, names in groups_config.items():
        x = explicit_missing_matrix(feature_rows, names)
        # Split seeds are replicate identity, so they live in r006 rather than in the
        # behavioral estimator block.
        seeds = yaml.safe_load((ROOT / "replicates/r006.yaml").read_text())["controller_training"]["outer_split_seeds"]
        for seed in seeds:
            outer = GroupKFold(n_splits=training["outer_folds"], shuffle=True, random_state=seed)
            for outer_fold, (train, valid) in enumerate(outer.split(x, groups=group_ids)):
                selected_c = select_c(
                    x[train], y[train], group_ids[train],
                    c_grid=training["regularization_selection"]["C_grid"],
                    folds=training["inner_folds"], seed=seed + 1000 + outer_fold,
                )
                probability_matrix = np.empty((len(valid), 3), dtype=float)
                for head in range(3):
                    model = make_pipeline(selected_c)
                    model.fit(x[train], y[train, head])
                    probability_matrix[:, head] = model.predict_proba(x[valid])[:, 1]
                for local_index, row_index in enumerate(valid):
                    problem_id = problem_ids[row_index]
                    probabilities = {
                        cap: float(probability_matrix[local_index, head])
                        for head, cap in enumerate(CAPS)
                    }
                    selections = {
                        menu: choose_action(probabilities, caps)
                        for menu, caps in condition["menus"].items()
                    }
                    sensitivity = {
                        f"eps_{epsilon:.2f}".replace(".", "_"): choose_action(
                            probabilities, condition["menus"]["full"],
                            lower_authority_tolerance=float(epsilon),
                        )
                        for epsilon in condition["decision_rule"]["sensitivity_tolerances"]
                    }
                    source = evidence[problem_id]
                    rows.append({
                        "schema_version": "controller-crossfit-prediction-v1",
                        "record_id": stable_id(args.run_id, problem_id, ablation, str(seed)),
                        "run_id": args.run_id,
                        "condition_id": CONDITION_ID,
                        "replicate_id": f"s{seed}",
                        "stage": "gate2",
                        "problem_id": problem_id,
                        "cell_id": source["cell_id"],
                        "feature_ablation": ablation,
                        "outer_seed": seed,
                        "outer_fold": outer_fold,
                        "selected_C": selected_c,
                        "source_evidence_record_id": source["record_id"],
                        "evidence_sha256": source["evidence_sha256"],
                        "probabilities": probabilities,
                        "selections": selections,
                        "sensitivity_full_menu": sensitivity,
                        "created_at_utc": utc_now(),
                    })
    expected = EXPECTED_ROWS * len(groups_config) * len(seeds)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} predictions, found {len(rows)}")
    print(canonical_json({"prediction_rows": write_jsonl(Path(args.output), rows)}))


def stage_final_fit(args: argparse.Namespace) -> None:
    import joblib

    condition = load_condition()["controller"]
    primary = condition["primary_feature_ablation"]
    names = condition["cumulative_feature_ablations"][primary]
    training = condition["training"]
    problem_ids, _, feature_rows, y = _training_arrays(
        Path(args.evidence_records), Path(args.target_records)
    )
    x = explicit_missing_matrix(feature_rows, names)
    selected_c = select_c(
        x, y, np.asarray(problem_ids),
        c_grid=training["regularization_selection"]["C_grid"],
        folds=training["outer_folds"], seed=20260920,
    )
    models = []
    for head in range(3):
        model = make_pipeline(selected_c)
        model.fit(x, y[:, head])
        models.append(model)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() or Path(args.manifest).exists():
        raise FileExistsError("refusing to overwrite final controller checkpoint")
    joblib.dump(
        {"feature_names": names, "target_order": CAPS, "selected_C": selected_c, "models": models},
        checkpoint,
    )
    versions = {
        "python": platform.python_version(),
        "numpy": importlib_metadata.version("numpy"),
        "scipy": importlib_metadata.version("scipy"),
        "scikit_learn": importlib_metadata.version("scikit-learn"),
        "joblib": importlib_metadata.version("joblib"),
    }
    manifest = {
        "schema_version": "controller-checkpoint-manifest-v1",
        "run_id": args.run_id,
        "condition_id": CONDITION_ID,
        "controller_id": condition["controller_id"],
        "feature_ablation": primary,
        "feature_names": names,
        "training_rows": EXPECTED_ROWS,
        "target_order": list(CAPS),
        "selected_C": selected_c,
        "implementation_versions": versions,
        "checkpoint_file": "controller/final_s2.joblib",
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_evidence_sha256": sha256_file(Path(args.evidence_records)),
        "source_targets_sha256": sha256_file(Path(args.target_records)),
        "created_at_utc": utc_now(),
    }
    Path(args.manifest).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(canonical_json({"selected_C": selected_c, "checkpoint": str(checkpoint)}))


def _policy_stats(rows, targets, menu: str) -> dict[str, Any]:
    selected = [row["selections"][menu] for row in rows]
    values = [targets[row["problem_id"]]["cap_outcomes"][cap] for row, cap in zip(rows, selected)]
    relative = {}
    for fixed_cap in CAPS:
        fixed_values = [
            targets[row["problem_id"]]["cap_outcomes"][fixed_cap] for row in rows
        ]
        repair = sum(a == 1 and b == 0 for a, b in zip(values, fixed_values))
        harm = sum(a == 0 and b == 1 for a, b in zip(values, fixed_values))
        relative[fixed_cap] = {"repair": repair, "harm": harm, "net": repair - harm}
    return {
        "accuracy": float(np.mean(values)),
        "correct": int(sum(values)),
        "selection_counts": dict(sorted(collections.Counter(selected).items())),
        "relative_to_fixed": relative,
    }


def stage_analyze(args: argparse.Namespace) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    predictions = read_jsonl(Path(args.prediction_records))
    targets = index(read_jsonl(Path(args.target_records)))
    y = {cap: np.asarray([targets[p]["cap_outcomes"][cap] for p in sorted(targets)]) for cap in CAPS}
    problem_order = sorted(targets)
    fixed = {cap: float(values.mean()) for cap, values in y.items()}
    oracle = float(np.mean([max(targets[p]["cap_outcomes"].values()) for p in problem_order]))
    by_group: dict[str, Any] = {}
    for ablation in load_condition()["controller"]["cumulative_feature_ablations"]:
        seed_rows = []
        for seed in sorted({row["outer_seed"] for row in predictions}):
            rows = sorted(
                (row for row in predictions if row["feature_ablation"] == ablation and row["outer_seed"] == seed),
                key=lambda row: row["problem_id"],
            )
            if len(rows) != EXPECTED_ROWS:
                raise ValueError(f"incomplete predictions for {ablation}/{seed}")
            probabilities = np.asarray([[row["probabilities"][cap] for cap in CAPS] for row in rows])
            labels = np.asarray([[targets[row["problem_id"]]["cap_outcomes"][cap] for cap in CAPS] for row in rows])
            discordant = labels[:, 0] != labels[:, 1]
            edge_auc = float(roc_auc_score(labels[discordant, 1], (probabilities[:, 1] - probabilities[:, 0])[discordant]))
            calibration = {}
            for head, cap in enumerate(CAPS):
                observed, predicted = calibration_curve(
                    labels[:, head], probabilities[:, head], n_bins=10, strategy="quantile"
                )
                calibration[cap] = [
                    {"mean_predicted": float(p), "observed_fraction": float(o)}
                    for o, p in zip(observed, predicted)
                ]
            sensitivity = {}
            for key in ("eps_0_01", "eps_0_02", "eps_0_05"):
                chosen = [row["sensitivity_full_menu"][key] for row in rows]
                values = [
                    targets[row["problem_id"]]["cap_outcomes"][cap]
                    for row, cap in zip(rows, chosen)
                ]
                sensitivity[key] = {
                    "accuracy": float(np.mean(values)),
                    "selection_counts": dict(sorted(collections.Counter(chosen).items())),
                }
            seed_rows.append({
                "outer_seed": seed,
                "selected_C_by_outer_fold": [
                    next(row["selected_C"] for row in rows if row["outer_fold"] == fold)
                    for fold in range(5)
                ],
                "mean_head_log_loss": float(np.mean([
                    log_loss(labels[:, head], probabilities[:, head], labels=[0, 1]) for head in range(3)
                ])),
                "mean_head_brier": float(np.mean([
                    brier_score_loss(labels[:, head], probabilities[:, head]) for head in range(3)
                ])),
                "m0_m1_edge_auc": edge_auc,
                "policies": {menu: _policy_stats(rows, targets, menu) for menu in ("live_edge", "legacy_extremes", "full")},
                "calibration_curve_quantile_10": calibration,
                "sensitivity_full_menu": sensitivity,
            })
        by_group[ablation] = {
            "seeds": seed_rows,
            "mean_m0_m1_edge_auc": float(np.mean([row["m0_m1_edge_auc"] for row in seed_rows])),
            "mean_policy_accuracy": {
                menu: float(np.mean([row["policies"][menu]["accuracy"] for row in seed_rows]))
                for menu in ("live_edge", "legacy_extremes", "full")
            },
        }
    primary = by_group["S2_program_and_critique_content"]
    summary = {
        "run_id": args.run_id,
        "condition_id": CONDITION_ID,
        "n": EXPECTED_ROWS,
        "fixed_cap_accuracy": fixed,
        "oracle_accuracy": oracle,
        "ablations": by_group,
        "primary_interpretation": (
            "Development cross-fit is design evidence only. Confirmatory claims require the later untouched test run."
        ),
        "created_at_utc": utc_now(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    lines = [
        "# Gate 2 development cross-fit (c008)", "",
        "This is development-only design evidence; the 300-row test manifest remained unread.", "",
        "## Fixed references", "",
        "|policy|accuracy|", "|---|---:|",
        *(f"|{cap}|{value:.4f}|" for cap, value in fixed.items()),
        f"|oracle|{oracle:.4f}|", "", "## Cumulative scalar ablations", "",
        "|features|M0-M1 edge AUC|M0/M1 policy|M0/M2 policy|full policy|", "|---|---:|---:|---:|---:|",
    ]
    for name, result in by_group.items():
        policy = result["mean_policy_accuracy"]
        lines.append(f"|{name}|{result['mean_m0_m1_edge_auc']:.4f}|{policy['live_edge']:.4f}|{policy['legacy_extremes']:.4f}|{policy['full']:.4f}|")
    lines.extend([
        "", "## Primary reading", "",
        f"The registered S2 scalars reach M0-M1 edge AUC {primary['mean_m0_m1_edge_auc']:.4f}; "
        f"their full-menu cross-fitted accuracy is {primary['mean_policy_accuracy']['full']:.4f}. "
        "This must be compared with the fixed caps above, not interpreted as a held-out result.", "",
        "All five split seeds are retained in `summary.json`; no best seed was selected.", "",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(canonical_json(summary))


def stage_finalize(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle_dir)
    files = sorted(
        path for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"RUN_MANIFEST.json", "checksums.sha256"}
    )
    checksums = "".join(f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n" for path in files)
    (bundle / "checksums.sha256").write_text(checksums, encoding="utf-8")
    git_commit = subprocess.run(
        ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], cwd=ROOT,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    manifest = {
        "schema_version": "run-manifest-gate2-v1",
        "run_id": args.run_id,
        "protocol_id": "authority_v6",
        "dataset_id": DATASET_ID,
        "condition_id": CONDITION_ID,
        "condition_hash": condition_hash(),
        "replicate_plan_id": REPLICATE_PLAN_ID,
        "stage": "gate2",
        "run_kind": "controller_train",
        "source_run_ids": [args.source_run_id],
        "git_commit": git_commit,
        "status": "completed",
        "environment": {"python": platform.python_version(), "scikit_learn": importlib_metadata.version("scikit-learn")},
        "started_at_utc": args.started_at_utc,
        "ended_at_utc": utc_now(),
        "output_files": len(files),
    }
    (bundle / "RUN_MANIFEST.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(canonical_json({"run_id": args.run_id, "git_commit": git_commit, "files": len(files)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="c008 Gate 2 scalar controller")
    sub = parser.add_subparsers(dest="command", required=True)
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--g-records", required=True); evidence.add_argument("--v-records", required=True)
    evidence.add_argument("--output", required=True); evidence.add_argument("--run-id", required=True)
    evidence.set_defaults(function=stage_evidence)
    targets = sub.add_parser("targets")
    targets.add_argument("--evidence-records", required=True); targets.add_argument("--counterfactual-records", required=True)
    targets.add_argument("--output", required=True); targets.set_defaults(function=stage_targets)
    crossfit = sub.add_parser("crossfit")
    crossfit.add_argument("--evidence-records", required=True); crossfit.add_argument("--target-records", required=True)
    crossfit.add_argument("--output", required=True); crossfit.add_argument("--run-id", required=True)
    crossfit.set_defaults(function=stage_crossfit)
    final_fit = sub.add_parser("final-fit")
    final_fit.add_argument("--evidence-records", required=True); final_fit.add_argument("--target-records", required=True)
    final_fit.add_argument("--checkpoint", required=True); final_fit.add_argument("--manifest", required=True); final_fit.add_argument("--run-id", required=True)
    final_fit.set_defaults(function=stage_final_fit)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--prediction-records", required=True); analyze.add_argument("--target-records", required=True)
    analyze.add_argument("--output-dir", required=True); analyze.add_argument("--run-id", required=True)
    analyze.set_defaults(function=stage_analyze)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--bundle-dir", required=True); finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--source-run-id", required=True); finalize.add_argument("--started-at-utc", required=True)
    finalize.set_defaults(function=stage_finalize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
