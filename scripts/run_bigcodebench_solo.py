#!/usr/bin/env python3
"""The BigCodeBench code study (docs/plan/code_bigcodebench_study.md): each model alone on the
832 development rows (c100-c109). Reviews are registered later with their own runner stage.

Usage: run_bigcodebench_solo.py --condition-id {c100,...,c109} <command> [args]

Commands:
    check-runnable  the frozen prompt, the model identity, dataset hash, decoding and seed cell
                    are the registered ones, and c101-c109 differ from c100 only in the model
    smoke           one call on a toy task: exact token alignment and a paired capture
    generate        one solve call per row with same-forward capture; resumable
    reconcile       keep only rows present and valid in both the record and trace files
    evidence        code_solve_scalar_v1, before grading
    grade-requests  the (record_id, task_id, text) file scripts/bigcodebench_grade.py reads
    grade           join the container's results to the records; any grader_error stops here
    analyze         reports/analysis.json and reports/summary.md

GPU stages run in slurm/bigcodebench_solo.sbatch; grading and analysis run on the cpu partition
in slurm/bigcodebench_solo_grade.sbatch, inside the pinned evaluate container.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c014_router as shared  # noqa: E402
import run_mathfull_solo as mathfull  # noqa: E402

from authority_allocation import bigcodebench_study, derive_call_seed  # noqa: E402
from authority_allocation.api import api_call, chat_template_kwargs_for  # noqa: E402
from authority_allocation.code_solve_features import FEATURE_CONTRACT, derive_features  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.prompts import split_prompt  # noqa: E402
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json, sha256_file, stable_id, utc_now,
)

DATASET_ID = "bigcodebench-b74c0d0-instruct-s20260917"
DATASET_PATH = ROOT / "datasets" / DATASET_ID / "dataset.yaml"
SOURCE_REPO = "bigcode/bigcodebench"
SOURCE_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
SOURCE_SPLIT = "v0.1.4"
PROBLEM_FIELD = "instruct_prompt"
MANIFEST = "development"
REPLICATE_PLAN_ID = "r015"
REPLICATE_ID = "p01"
BASE_SEED = 20260928
EXPECTED_ROWS = 832
BUDGET = 4096
IDENTITY_FIELDS = ("model_id", "revision", "tokenizer_revision", "server_path", "chat_template_hash",
                   "chat_template_kwargs")
# Measured by the ground-truth check; mirrors scripts/bigcodebench_grade.py, which cannot be
# imported on the host (it needs the bigcodebench package from the evaluate container).
EXCLUDED_TASKS = frozenset({
    "BigCodeBench/205", "BigCodeBench/363", "BigCodeBench/590", "BigCodeBench/1005",
    "BigCodeBench/101", "BigCodeBench/176", "BigCodeBench/314", "BigCodeBench/1012",
})

SOLO_SECTIONS = {"condition_id", "status", "purpose", "gate", "runnable_gate", "model", "replicates"}
SOLO_MODEL_KEYS = {"model_id", "revision", "tokenizer_revision", "provenance", "identity_reference",
                   "server_path", "chat_template_hash", "chat_template_kwargs"}
# condition -> (model, identity reference). A reference that is a condition id other than the
# condition itself names a registered model block; the coder models, whose earlier conditions
# predate model blocks, reference themselves and are checked against their provenance file.
MODELS = {"c100": ("Qwen/Qwen3-4B", "c025"),
          "c101": ("Qwen/Qwen3-8B", "c026"),
          "c102": ("meta-llama/Llama-3.2-3B-Instruct", "c029"),
          "c103": ("Qwen/Qwen2.5-3B-Instruct", "c030"),
          "c104": ("Qwen/Qwen2.5-7B-Instruct", "c031"),
          "c105": ("mistralai/Ministral-8B-Instruct-2410", "c032"),
          "c106": ("mistralai/Mistral-Small-24B-Instruct-2501", "c054"),
          "c107": ("Qwen/Qwen3-32B", "c055"),
          "c108": ("Qwen/Qwen2.5-Coder-7B-Instruct", "c108"),
          "c109": ("ByteDance-Seed/Seed-Coder-8B-Instruct", "c109")}
CONDITION_ID = "c100"
ROLE = "solve"
STAGE = "bigcodebench-solve"

SMOKE_TASK = ("Write a function that returns the sum of two integers.\n"
              "You should write self-contained code starting with:\n```\ndef task_func(a, b):\n```")


# ------------------------------------------------------------------ configuration

def configure(condition_id: str) -> None:
    global CONDITION_ID
    if condition_id not in MODELS:
        raise SystemExit(f"unknown condition {condition_id}")
    CONDITION_ID = condition_id
    shared.CONDITION_ID = condition_id
    shared.CONDITION_PATH = ROOT / "conditions" / condition_id / "condition.yaml"
    shared.DATASET_ID = DATASET_ID
    shared.REPLICATE_PLAN_ID = REPLICATE_PLAN_ID
    shared.STAGE = STAGE
    shared.SEED_CELL_ID = condition_id
    shared.EXPECTED_ROWS = EXPECTED_ROWS
    shared.configure()


def condition(condition_id: str | None = None) -> dict[str, Any]:
    cid = condition_id or CONDITION_ID
    return resolve_condition(ROOT / "conditions" / cid / "condition.yaml", repository_root=ROOT)


def registered_dataset() -> dict[str, Any]:
    return yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8"))


def reference_identity(condition_id: str, reference: str) -> dict[str, Any]:
    if reference != condition_id:
        model = condition(reference)["model"]
        return {field: model.get(field) for field in IDENTITY_FIELDS}
    provenance = json.loads((ROOT / condition(condition_id)["model"]["provenance"]).read_text(encoding="utf-8"))
    return {"model_id": provenance["repo_id"], "revision": provenance["revision"],
            "tokenizer_revision": provenance["tokenizer_revision"],
            "server_path": provenance["server_path"],
            "chat_template_hash": provenance["chat_template_hash"], "chat_template_kwargs": None}


def require_registered(condition_id: str) -> dict[str, Any]:
    model_id, reference = MODELS[condition_id]
    cond = condition(condition_id)
    model = cond["model"]
    if model["prompt_pass1"] != bigcodebench_study.CODE_SOLVE_PROMPT:
        raise SystemExit(f"{condition_id} must use the frozen prompt {bigcodebench_study.CODE_SOLVE_PROMPT}")
    bigcodebench_study.require_frozen_prompt(ROOT, model["prompt_pass1"])
    if model["model_id"] != model_id or model.get("identity_reference") != reference:
        raise SystemExit(f"{condition_id} must run {model_id} with {reference}'s identity")
    expected = reference_identity(condition_id, reference)
    for field in IDENTITY_FIELDS:
        if model.get(field) != expected[field]:
            raise SystemExit(f"{condition_id} model.{field} differs from {reference}'s")
    if model.get("provenance"):
        provenance = json.loads((ROOT / model["provenance"]).read_text(encoding="utf-8"))
        for field, key in (("model_id", "repo_id"), ("revision", "revision"),
                           ("server_path", "server_path"), ("chat_template_hash", "chat_template_hash")):
            if model[field] != provenance[key]:
                raise SystemExit(f"{condition_id} model.{field} differs from {model['provenance']}")
    decoding = model["pass1_decoding"]
    if float(decoding["temperature"]) != 0.0 or model.get("thinking") is not False:
        raise SystemExit(f"{condition_id} requires temperature 0 and thinking false")
    if int(decoding["max_new_tokens"]) != BUDGET:
        raise SystemExit(f"{condition_id} output budget must be {BUDGET}")
    if float(decoding["repetition_penalty"]) != 1.0 or float(decoding["presence_penalty"]) != 0.0:
        raise SystemExit(f"{condition_id} must send repetition_penalty 1.0 and presence_penalty 0.0")
    registered = registered_dataset()
    manifest = registered["manifests"][MANIFEST]
    if (cond["dataset"]["id"] != DATASET_ID or cond["dataset"]["manifest"] != MANIFEST
            or cond["dataset"]["manifest_rows"] != manifest["rows"] or manifest["rows"] != EXPECTED_ROWS
            or cond["dataset"]["manifest_sha256"] != manifest["sha256"]
            or cond["dataset"]["problem_field"] != PROBLEM_FIELD
            or registered["source"]["revision"] != SOURCE_REVISION):
        raise SystemExit(f"{condition_id} dataset block disagrees with {DATASET_PATH.name}")
    if cond["replicates"]["plan_id"] != REPLICATE_PLAN_ID \
            or cond["replicates"]["seed_cell_id"] != condition_id:
        raise SystemExit(f"{condition_id} must use {REPLICATE_PLAN_ID} with its own seed cell")
    if condition_id != "c100":
        parent = condition("c100")
        for key in sorted(set(cond) | set(parent)):
            if key not in SOLO_SECTIONS and cond.get(key) != parent.get(key):
                raise SystemExit(f"{condition_id} {key} differs from c100")
        for key in sorted(set(model) | set(parent["model"])):
            if key not in SOLO_MODEL_KEYS and model.get(key) != parent["model"].get(key):
                raise SystemExit(f"{condition_id} model.{key} differs from c100")
    return model


# ------------------------------------------------------------------ data

def manifest_rows(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} manifest rows, got {len(rows)}")
    excluded = sorted({str(r["problem_id"]) for r in rows} & EXCLUDED_TASKS)
    if excluded:
        raise ValueError(f"manifest holds excluded tasks {excluded}")
    return rows


def load_bigcodebench_rows(problem_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Join an id-only manifest to the pinned source."""
    from datasets import load_dataset

    wanted = set(problem_ids)
    rows: dict[str, dict[str, Any]] = {}
    for row in load_dataset(SOURCE_REPO, revision=SOURCE_REVISION, split=SOURCE_SPLIT):
        pid = str(row["task_id"])
        if pid not in wanted:
            continue
        if pid in rows:
            raise ValueError(f"duplicate source row {pid}")
        rows[pid] = {"problem_text": row[PROBLEM_FIELD], "entry_point": row["entry_point"]}
    missing = wanted.difference(rows)
    if missing:
        raise ValueError(f"manifest ids missing from the pinned source: {sorted(missing)[:5]}")
    return rows


# ------------------------------------------------------------------ model calls

def model_identity(model: Mapping[str, Any]) -> dict[str, Any]:
    return {"model_id": model["model_id"], "revision": model["revision"],
            "server_path": model["server_path"], "backend": model["backend"],
            "prompt_sha256": sha256_file(ROOT / model["prompt_pass1"]),
            "chat_template_sha256": model["chat_template_hash"], "dtype": model["dtype"],
            "quantization": model["quantization"]}


def build_messages(model: Mapping[str, Any], problem: str) -> list[dict[str, str]]:
    system, user = split_prompt(ROOT / model["prompt_pass1"])
    return [{"role": "system", "content": system},
            {"role": "user", "content": bigcodebench_study.user_message(user, problem=problem)}]


def call_model(args: argparse.Namespace, model: Mapping[str, Any], messages: list[dict[str, str]],
               seed: int, capture_id: str) -> tuple[str, dict[str, Any]]:
    content, _, usage = api_call(
        api_base=args.api_base, served_model=args.served_model, messages=messages,
        decoding={**model["pass1_decoding"],
                  "output_logprobs": {"enabled": True, "mode": "raw_logprobs",
                                      "return_token_ids": True},
                  "vllm_xargs": {"c014_capture_id": capture_id}},
        seed=seed, thinking=False, chat_template_kwargs=chat_template_kwargs_for(model))
    return content, usage


def stage_smoke(args: argparse.Namespace) -> None:
    model = require_registered(CONDITION_ID)
    capture_id = stable_id(args.run_id, "smoke", "same_forward_logit_capture")
    content, usage = call_model(args, model, build_messages(model, SMOKE_TASK), 1, capture_id)
    trace = usage.get("token_logprobs") or {}
    record = {"run_id": args.run_id, "problem_id": "smoke",
              "outputs": [{"raw_output": content, "token_logprobs": trace}]}
    capture = shared.wait_for_same_forward_capture(Path(args.capture_sidecar), capture_id,
                                                   timeout_seconds=60)
    native = shared.build_same_forward_logit_record(run_id=args.run_id, pass1_record=record,
                                                    capture=capture)
    problems = []
    if not content:
        problems.append("empty content")
    if trace.get("status") != "ok":
        problems.append(f"trace {trace.get('status')}")
    if native["status"] != "ok":
        problems.append(f"native capture {native['status']}")
    print(canonical_json({"stage": "smoke", "condition_id": CONDITION_ID,
                          "native_status": native["status"],
                          "python_blocks": bigcodebench_study.python_block_count(content),
                          "finish_reason": usage.get("finish_reason"), "problems": problems}))
    if problems:
        raise SystemExit("smoke failed: " + "; ".join(problems))


def stage_generate(args: argparse.Namespace) -> None:
    model = require_registered(CONDITION_ID)
    manifest = manifest_rows(Path(args.manifest), int(args.expected_rows))
    ids = [str(r["problem_id"]) for r in manifest]
    by_id = {str(r["problem_id"]): r for r in manifest}
    sources = load_bigcodebench_rows(ids)
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for problem_id in ids:
        if (problem_id,) in done:
            continue
        row = sources[problem_id]
        messages = build_messages(model, row["problem_text"])
        seed = derive_call_seed(REPLICATE_PLAN_ID, "pass1", args.replicate_id, int(args.base_seed),
                                CONDITION_ID, problem_id, 0)
        capture_id = stable_id(args.run_id, problem_id, "same_forward_logit_capture")
        content, usage = call_model(args, model, messages, seed, capture_id)
        trace = usage.pop("token_logprobs", None)
        record = {
            "schema_version": "code-record-v1",
            "record_id": stable_id(args.run_id, problem_id, "pass1"),
            "run_id": args.run_id, "condition_id": CONDITION_ID,
            "condition_hash": shared.condition_hash(), "dataset_id": DATASET_ID,
            "replicate_id": args.replicate_id, "stage": STAGE,
            "problem_id": problem_id, "problem_text": row["problem_text"],
            "problem_metadata": {"entry_point": row["entry_point"],
                                 "libs": str(by_id[problem_id].get("libs", ""))},
            "generator": model_identity(model),
            "outputs": [{"sample_index": 0, "call_seed": seed, "raw_output": content,
                         "python_block_count": bigcodebench_study.python_block_count(content),
                         "token_logprobs": trace, "usage": shared.base.compact_usage(usage)}],
            "y0_sample_index": 0, "created_at_utc": utc_now(),
        }
        append_jsonl(output, record)
        capture = shared.wait_for_same_forward_capture(Path(args.capture_sidecar), capture_id,
                                                       timeout_seconds=60)
        native = shared.build_same_forward_logit_record(run_id=args.logit_run_id,
                                                        pass1_record=record, capture=capture)
        append_jsonl(Path(args.logit_output), native)
        if native["status"] != "ok":
            raise RuntimeError(f"same-forward logit capture failed for {problem_id}: {native['status']}")
        written += 1
    print(canonical_json({"stage": "pass1", "written": written, "total": len(ids)}))


# ------------------------------------------------------------------ evidence, grading, analysis

def stage_evidence(args: argparse.Namespace) -> None:
    natives = {row["problem_id"]: row for row in read_jsonl(Path(args.logit_traces))}
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for record in read_jsonl(Path(args.records)):
        pid = record["problem_id"]
        if (pid,) in done:
            continue
        native = natives.get(pid)
        if native is None or native.get("status") != "ok":
            raise RuntimeError(f"missing valid native trace for {pid}")
        features, audit = derive_features(output=record["outputs"][0], native=native,
                                          problem_text=record["problem_text"])
        append_jsonl(output, {
            "schema_version": "code-evidence-record-v1",
            "record_id": stable_id(args.run_id, pid, "code-evidence"), "run_id": args.run_id,
            "source_record_run_id": record["run_id"], "source_logit_run_id": native["run_id"],
            "condition_id": CONDITION_ID, "condition_hash": shared.condition_hash(),
            "problem_id": pid, "role": ROLE, "feature_contract": FEATURE_CONTRACT,
            "features": features, "missingness": {k: v is None for k, v in features.items()},
            "audit": audit, "created_at_utc": utc_now()})
        written += 1
    print(canonical_json({"stage": "evidence", "written": written}))


def stage_grade_requests(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.unlink(missing_ok=True)
    records = read_jsonl(Path(args.records))
    for record in records:
        append_jsonl(output, {"record_id": record["record_id"], "task_id": record["problem_id"],
                              "text": record["outputs"][0]["raw_output"]})
    print(canonical_json({"stage": "grade-requests", "rows": len(records)}))


def stage_grade(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.records))
    results = {r["record_id"]: r for r in read_jsonl(Path(args.results))}
    if len(results) != len(records):
        raise RuntimeError(f"{len(results)} grader results for {len(records)} records")
    errors = [r for r in results.values() if r["status"] == "grader_error"]
    if errors:
        raise RuntimeError(f"{len(errors)} grader_error rows, e.g. {errors[0]}; the run is rejected")
    output = Path(args.output)
    output.unlink(missing_ok=True)
    for record in records:
        result = results.get(record["record_id"])
        if result is None or result["task_id"] != record["problem_id"]:
            raise RuntimeError(f"no grader result for {record['problem_id']}")
        append_jsonl(output, {"problem_id": record["problem_id"], "outcome": "u0", "u": int(result["u"]),
                              "status": result["status"], "detail": result["detail"],
                              "seconds": result["seconds"], "source_record_id": record["record_id"]})
    print(canonical_json({"stage": "grade", "rows": len(records)}))


def repetition_share(text: str, n: int = 8, window: int = 400) -> float:
    """Share of repeated word n-grams in the last ``window`` words; near 1 for a loop."""
    words = text.split()[-window:]
    grams = [tuple(words[i:i + n]) for i in range(max(len(words) - n + 1, 0))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def stage_analyze(args: argparse.Namespace) -> None:
    records = {r["problem_id"]: r for r in read_jsonl(Path(args.records))}
    evidence = {r["problem_id"]: r for r in read_jsonl(Path(args.evidence))}
    grades = {r["problem_id"]: r for r in read_jsonl(Path(args.grades)) if r["outcome"] == "u0"}
    expected = int(args.expected_rows)
    if not (len(records) == len(evidence) == len(grades) == expected):
        raise RuntimeError(f"incomplete inputs: {len(records)}, {len(evidence)}, {len(grades)}")
    ids = sorted(records)
    passed = sum(int(grades[p]["u"]) for p in ids)
    details = collections.Counter(grades[p]["detail"] for p in ids)
    usage = [records[p]["outputs"][0]["usage"] for p in ids]
    tokens = [int(u.get("output_tokens") or 0) for u in usage]
    truncated = [p for p, u in zip(ids, usage) if u.get("finish_reason") == "length"]
    loops = [p for p in truncated if repetition_share(records[p]["outputs"][0]["raw_output"]) >= 0.5]
    blocks = collections.Counter(min(records[p]["outputs"][0]["python_block_count"], 2) for p in ids)
    missing = {name: sum(evidence[p]["missingness"][name] for p in ids)
               for name in evidence[ids[0]]["features"]}
    report: dict[str, Any] = {
        "condition_id": CONDITION_ID, "run_id": args.run_id, "role": ROLE, "rows": expected,
        "passed": passed, "pass_rate": passed / expected, "grader_detail_counts": dict(details),
        "model_invalid_rows": details.get("no_program", 0), "timeout_rows": details.get("timeout", 0),
        "truncated_rows": len(truncated), "truncated_repetition_loops": len(loops),
        "truncated_problem_ids": truncated,
        "python_block_count": {"0": blocks.get(0, 0), "1": blocks.get(1, 0), "2_or_more": blocks.get(2, 0)},
        "output_tokens": {"sum": sum(tokens), "mean": statistics.mean(tokens),
                          "median": statistics.median(tokens), "max": max(tokens)},
        "scalar_missingness": missing, "feature_contract": FEATURE_CONTRACT,
        "development_only": True,
    }
    lines = [f"# {CONDITION_ID} BigCodeBench solve summary", "",
             f"- Rows: {expected}. Passed all tests: {passed}/{expected} = {passed / expected:.4f}.",
             f"- Grader detail: {json.dumps(dict(details), sort_keys=True)} "
             "(no_program is model_invalid; a timeout counts as a failure; grader errors: 0).",
             f"- Complete fenced Python blocks per output: none {blocks.get(0, 0)}, one {blocks.get(1, 0)}, "
             f"two or more {blocks.get(2, 0)}.",
             f"- Output tokens: sum {sum(tokens)}, mean {statistics.mean(tokens):.1f}, "
             f"max {max(tokens)}; truncated at {BUDGET}: {len(truncated)} rows, "
             f"{len(loops)} of them repetition loops (repeated 8-gram share >= 0.5).",
             f"- Scalar missingness: {json.dumps({k: v for k, v in missing.items() if v}, sort_keys=True)}",
             "", "Evidence was written before grading. Grading runs the task's own tests in the pinned "
             "BigCodeBench evaluate container with networking disabled.", ""]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output.parent / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------ entry point

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-runnable")
    smoke = sub.add_parser("smoke")
    for name in ("--run-id", "--api-base", "--served-model", "--capture-sidecar"):
        smoke.add_argument(name, required=True)
    generate = sub.add_parser("generate")
    for name in ("--run-id", "--api-base", "--served-model", "--manifest", "--output",
                 "--capture-sidecar", "--logit-output", "--logit-run-id"):
        generate.add_argument(name, required=True)
    generate.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    generate.add_argument("--replicate-id", default=REPLICATE_ID)
    generate.add_argument("--base-seed", type=int, default=BASE_SEED)
    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("records")
    reconcile.add_argument("traces")
    ev = sub.add_parser("evidence")
    for name in ("--run-id", "--records", "--logit-traces", "--output"):
        ev.add_argument(name, required=True)
    requests = sub.add_parser("grade-requests")
    for name in ("--records", "--output"):
        requests.add_argument(name, required=True)
    grade = sub.add_parser("grade")
    for name in ("--records", "--results", "--output"):
        grade.add_argument(name, required=True)
    analyze = sub.add_parser("analyze")
    for name in ("--run-id", "--records", "--evidence", "--grades", "--output"):
        analyze.add_argument(name, required=True)
    analyze.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    return parser


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] != "--condition-id":
        raise SystemExit(__doc__)
    configure(argv[1])
    args = build_parser().parse_args(argv[2:])
    if args.command == "check-runnable":
        model = require_registered(CONDITION_ID)
        print(canonical_json({"runnable": True, "condition_id": CONDITION_ID, "role": ROLE,
                              "model_id": model["model_id"], "revision": model["revision"],
                              "max_new_tokens": BUDGET, "dataset_id": DATASET_ID}))
    elif args.command == "smoke":
        stage_smoke(args)
    elif args.command == "generate":
        stage_generate(args)
    elif args.command == "reconcile":
        print(canonical_json({"stage": "reconcile",
                              **mathfull.reconcile(Path(args.records), Path(args.traces))}))
    elif args.command == "evidence":
        stage_evidence(args)
    elif args.command == "grade-requests":
        stage_grade_requests(args)
    elif args.command == "grade":
        stage_grade(args)
    else:
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
