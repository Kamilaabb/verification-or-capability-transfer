#!/usr/bin/env python3
"""Gate 0 for the code domain: generate -> verify -> grade -> analyze.

Rewrite is skipped on purpose. Grading scores candidate_answer unconditionally, so
every Gate 0 measure comes off the generate/verify/grade chain; running rewrite would
buy inference on a question this gate does not ask.

Grading is not a function call here. Correctness is defined by executing untrusted code
inside the pinned sandbox, so the runner emits a request file, the job script runs the
container, and the runner ingests the results. Keeping the container invocation in the
job script is deliberate: its flags carry hard-won constraints (network namespace off,
--mem above the address-space rlimit, never pipe its stdout) that belong next to the
Slurm directives they interact with.

Shared infrastructure comes from the package; what lives here is the code domain's own
shape. See docs/decisions/runner_architecture.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import derive_call_seed  # noqa: E402
from authority_allocation.api import api_call, format_repair_call  # noqa: E402
from authority_allocation.payloads import (  # noqa: E402
    parse_json_object,
    validate_v_report,
)
from authority_allocation.prompts import render, split_prompt  # noqa: E402
from authority_allocation.records import (  # noqa: E402
    append_jsonl,
    completed_keys,
    read_jsonl,
)
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.resampling import summarize_usage  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json,
    sha256_file,
    stable_id,
    utc_now,
)

CONDITION_ID = "c005"
CONDITION_PATH = ROOT / "conditions" / CONDITION_ID / "condition.yaml"
DATASET_ID = "bigcodebench-b74c0d0-instruct-s20260917"
PROBLEM_FIELD = "instruct_prompt"
DATASET_REPO = "bigcode/bigcodebench"
DATASET_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
DATASET_SPLIT = "v0.1.4"
REPLICATE_PLAN_ID = "r004"
GENERATOR_BASE_SEED = 20260917
VERIFIER_BASE_SEED = 20260917
STAGE = "gate0"
CELL_ID = "V-A"
EXPECTED_ROWS = 300

G_RECORD_SCHEMA_VERSION = "g-record-single-v2"
V_RECORD_SCHEMA_VERSION = "v-record-v2"
GRADE_RECORD_SCHEMA_VERSION = "code-grade-record-v1"

PROMPTS = {
    "g": ROOT / "conditions" / "c005" / "prompts" / "g_initial_v1.txt",
    "v": ROOT / "conditions" / "c005" / "prompts" / "v_report_v1.txt",
}
SCHEMAS = {
    "v": ROOT / "protocols/authority_v5/schemas/v_report_payload_v1.json",
}


def configure_c006_guidance_backend() -> None:
    """Select c006 without changing c005's default CLI.

    c006 differs from c005 in exactly one registered field: the structured-outputs
    backend. Everything else is held fixed on purpose.

    The prompts are the same files — c005's numbers were measured under a verifier that
    could not write code, and changing the wording at the same time as the backend would
    leave no way to attribute the difference.

    The replicate plan stays r004 as well. derive_call_seed hashes the plan id, so a new
    plan would redraw every call seed and add a second difference to a comparison meant
    to isolate one. c004 reused c003's plan for the same reason.
    """

    global CONDITION_ID, CONDITION_PATH

    CONDITION_ID = "c006"
    CONDITION_PATH = ROOT / "conditions/c006/condition.yaml"


# ---------------------------------------------------------------- condition access

def load_condition() -> dict[str, Any]:
    return resolve_condition(CONDITION_PATH, repository_root=ROOT)


def condition_hash() -> str:
    """Hash the resolved behavioural fields plus the exact prompt bytes.

    Excludes the fields the condition's own hash_policy excludes, so a status edit or a
    different node does not invent a new identity.
    """

    condition = load_condition()
    excluded = set(condition["hash_policy"]["exclude"])
    behavioural = {key: value for key, value in condition.items() if key not in excluded}
    prompt_bytes = {
        role: sha256_file(path) for role, path in sorted(PROMPTS.items())
    }
    payload = {
        "condition": behavioural,
        "prompt_sha256": prompt_bytes,
        "schema_sha256": {role: sha256_file(path) for role, path in sorted(SCHEMAS.items())},
    }
    from authority_allocation.serialization import sha256_bytes

    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def model_config(role: str) -> dict[str, Any]:
    condition = load_condition()
    if role == "g":
        return condition["generator"]
    if role == "v":
        return condition["verifiers"][0]
    raise ValueError(f"unknown role: {role}")


def decoding_config(role: str) -> dict[str, Any]:
    config = model_config(role)
    return config["initial_decoding"] if role == "g" else config["decoding"]


def model_identity(role: str, prompt_path: Path) -> dict[str, Any]:
    config = model_config(role)
    return {
        "model_id": config["model_id"],
        "revision": config["revision"],
        "backend": config["backend"],
        "backend_version": config["backend_version"],
        "dtype": config["dtype"],
        "quantization": config["quantization"],
        "chat_template_sha256": config["chat_template_hash"],
        "thinking": config["thinking"],
        "prompt_sha256": sha256_file(prompt_path),
    }


def compact_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "latency_ms": usage.get("latency_ms"),
        "finish_reason": usage.get("finish_reason"),
        "retries": usage.get("retries", 0),
    }


# ------------------------------------------------------------------- dataset access

def manifest_task_ids(path: Path) -> list[str]:
    rows = read_jsonl(path)
    task_ids = [str(row["problem_id"]) for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate problem_id in {path}")
    return task_ids


def load_source_rows(task_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Join an id-only frozen manifest to the pinned dataset revision."""

    from datasets import load_dataset

    wanted = set(task_ids)
    indexed: dict[str, dict[str, Any]] = {}
    dataset = load_dataset(DATASET_REPO, revision=DATASET_REVISION, split=DATASET_SPLIT)
    for row in dataset:
        task_id = str(row["task_id"])
        if task_id not in wanted:
            continue
        indexed[task_id] = {
            "problem_text": row[PROBLEM_FIELD],
            "entry_point": row["entry_point"],
            "libs": row.get("libs"),
        }
    missing = sorted(wanted - set(indexed))
    if missing:
        raise RuntimeError(f"manifest ids absent from the pinned revision: {missing[:5]}")
    return indexed


# ------------------------------------------------------------------------- stages

def stage_smoke(args: argparse.Namespace) -> None:
    """Exercise one loaded model on a non-research input before data are read."""

    role = args.role
    config = model_config(role)
    decoding = decoding_config(role)
    schema: dict[str, Any] | None = None
    if role == "g":
        messages = [{
            "role": "user",
            "content": (
                "This is a non-research interface test. Write a Python function "
                "`add(a, b)` returning the sum of two integers, in one code block."
            ),
        }]
    else:
        schema = json.loads(SCHEMAS["v"].read_text(encoding="utf-8"))
        messages = [{
            "role": "user",
            "content": (
                "This is a non-research interface test. For the task 'add two integers', "
                "inspect the proposed program 'def add(a, b): return a + b'. Return "
                "verdict, critique, a complete candidate_answer program, and requested_action."
            ),
        }]
    content, _, usage = api_call(
        api_base=args.api_base,
        served_model=args.served_model,
        messages=messages,
        decoding=decoding,
        seed=0,
        thinking=config["thinking"],
        schema=schema,
    )
    valid = bool(content.strip())
    if role == "v":
        valid = validate_v_report(parse_json_object(content))
    if not valid:
        raise RuntimeError(f"{role} compatibility smoke returned an invalid response")
    trace = usage.get("token_logprobs") or {}
    if trace.get("status") != "ok":
        raise RuntimeError(
            f"{role} compatibility smoke did not return an exact content logprob trace"
        )
    print(canonical_json({"role": role, "status": "passed", "usage": compact_usage(usage)}))


def stage_generate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    system, template = split_prompt(PROMPTS["g"])
    config = model_config("g")
    decoding = config["initial_decoding"]
    current_condition_hash = condition_hash()
    task_ids = manifest_task_ids(Path(args.manifest))
    source_rows = load_source_rows(task_ids)
    for task_id in task_ids:
        if (task_id,) in done:
            continue
        problem = source_rows[task_id]["problem_text"]
        seed = derive_call_seed(
            REPLICATE_PLAN_ID, "generator", "g01", GENERATOR_BASE_SEED, "G", task_id, 0
        )
        content, _, usage = api_call(
            api_base=args.api_base,
            served_model=args.served_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": render(template, problem=problem)},
            ],
            decoding=decoding,
            seed=seed,
            thinking=config["thinking"],
        )
        # Extraction is the grader's job here: whether a program can be recovered is
        # decided by the same sanitizer that will run it, inside the sandbox. Deciding
        # it twice, in two places, is how the two answers drift apart.
        row = {
            "schema_version": G_RECORD_SCHEMA_VERSION,
            "record_id": stable_id(args.run_id, task_id),
            "run_id": args.run_id,
            "condition_id": CONDITION_ID,
            "condition_hash": current_condition_hash,
            "dataset_id": DATASET_ID,
            "replicate_id": "g01",
            "stage": STAGE,
            "problem_id": task_id,
            "problem_text": problem,
            "problem_metadata": {
                "entry_point": source_rows[task_id]["entry_point"],
                "libs": source_rows[task_id]["libs"],
            },
            "generator": model_identity("g", PROMPTS["g"]),
            "outputs": [{
                "sample_index": 0,
                "call_seed": seed,
                "raw_output": content,
                "extracted_answer": None,
                "parse_status": "deferred_to_grader",
                "token_logprobs": usage.get("token_logprobs"),
                "usage": compact_usage(usage),
            }],
            "y0_sample_index": 0,
            "created_at_utc": utc_now(),
        }
        append_jsonl(output, row)


def stage_verify(args: argparse.Namespace) -> None:
    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "cell_id"))
    system, template = split_prompt(PROMPTS["v"])
    config = model_config("v")
    decoding = config["decoding"]
    schema = json.loads(SCHEMAS["v"].read_text(encoding="utf-8"))
    current_condition_hash = condition_hash()
    for g_row in read_jsonl(Path(args.g_records)):
        task_id = g_row["problem_id"]
        if (task_id, CELL_ID) in done:
            continue
        y0 = g_row["outputs"][0]["raw_output"]
        seed = derive_call_seed(
            REPLICATE_PLAN_ID, "verifier", "v01", VERIFIER_BASE_SEED, CELL_ID, task_id, 0
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": render(template, problem=g_row["problem_text"], y0=y0)},
        ]
        content, reasoning, usage = api_call(
            api_base=args.api_base,
            served_model=args.served_model,
            messages=messages,
            decoding=decoding,
            seed=seed,
            thinking=config["thinking"],
            schema=schema,
        )
        report = parse_json_object(content)
        parse_status = "ok"
        if not validate_v_report(report):
            repair_messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Repair only the JSON format. Return exactly one object matching "
                        "the required schema."
                    ),
                },
            ]
            retry_result = format_repair_call(
                api_base=args.api_base,
                served_model=args.served_model,
                messages=repair_messages,
                decoding=decoding,
                seed=seed,
                thinking=config["thinking"],
                schema=schema,
            )
            usage["retries"] = 1
            if retry_result is not None:
                content_retry, reasoning_retry, retry_usage = retry_result
                retry_usage["retries"] = 1
                for field in (
                    "input_tokens",
                    "reasoning_tokens",
                    "output_tokens",
                    "latency_ms",
                ):
                    retry_usage[field] = (usage.get(field) or 0) + (retry_usage.get(field) or 0)
                content, reasoning, usage = content_retry, reasoning_retry, retry_usage
                report = parse_json_object(content)
                parse_status = "retry_ok"
        if not validate_v_report(report):
            report = None
            parse_status = "fallback"
        # A report that cannot be parsed registers as KEEP: the verifier is not credited
        # with an action it never successfully requested.
        registered_request = "KEEP" if report is None else report["requested_action"]
        row = {
            "schema_version": V_RECORD_SCHEMA_VERSION,
            "record_id": stable_id(args.run_id, CELL_ID, task_id),
            "run_id": args.run_id,
            "source_g_run_id": g_row["run_id"],
            "condition_id": CONDITION_ID,
            "condition_hash": current_condition_hash,
            "replicate_id": "v01",
            "stage": STAGE,
            "cell_id": CELL_ID,
            "problem_id": task_id,
            "verifier": model_identity("v", PROMPTS["v"]),
            "call_seed": seed,
            "raw_output": content,
            "reasoning_content": reasoning,
            "token_logprobs": usage.get("token_logprobs"),
            "v_report": report,
            "report_parse_status": parse_status,
            "registered_request": registered_request,
            "usage": {
                **compact_usage(usage),
                "reasoning_tokens": usage.get("reasoning_tokens"),
            },
            "created_at_utc": utc_now(),
        }
        append_jsonl(output, row)


def stage_grade_requests(args: argparse.Namespace) -> None:
    """Emit one grading request per outcome the gate needs.

    y2 is requested whenever a report parsed, not only for OVERRIDE ones. That is what
    makes P(u2 = 1 | request = KEEP) computable — whether the verifier held back an
    answer that was in fact correct.

    A report that did not parse gets no y2 request at all. Grading its absent candidate
    as a wrong program would be a category error: the registered request is KEEP, so M2
    deploys y0 and never deploys an empty program. Scoring it 0 would depress mean(u2)
    with a parsing failure.
    """

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    v_by_problem = {row["problem_id"]: row for row in read_jsonl(Path(args.v_records))}
    rows = []
    for g_row in read_jsonl(Path(args.g_records)):
        task_id = g_row["problem_id"]
        rows.append({
            "record_id": stable_id(args.run_id, task_id, "y0"),
            "task_id": task_id,
            "outcome": "u0",
            "text": g_row["outputs"][0]["raw_output"],
        })
        v_row = v_by_problem.get(task_id)
        report = v_row["v_report"] if v_row is not None else None
        if report is None:
            continue
        rows.append({
            "record_id": stable_id(args.run_id, task_id, "y2"),
            "task_id": task_id,
            "outcome": "u2",
            "text": report["candidate_answer"],
        })
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    print(canonical_json({"requests": len(rows), "output": str(output)}))


def stage_grade(args: argparse.Namespace) -> None:
    """Ingest the sandbox's results into auditable per-outcome grading records."""

    output = Path(args.output)
    requests = {row["record_id"]: row for row in read_jsonl(Path(args.requests))}
    results = {row["record_id"]: row for row in read_jsonl(Path(args.results))}
    missing = sorted(set(requests) - set(results))
    if missing:
        raise RuntimeError(f"grader returned no result for {len(missing)} requests: {missing[:5]}")
    extra = sorted(set(results) - set(requests))
    if extra:
        raise RuntimeError(f"grader returned unrequested results: {extra[:5]}")
    current_condition_hash = condition_hash()
    for record_id, request_row in requests.items():
        result = results[record_id]
        if result["task_id"] != request_row["task_id"]:
            raise RuntimeError(f"grader result {record_id} is for the wrong task")
        append_jsonl(output, {
            "schema_version": GRADE_RECORD_SCHEMA_VERSION,
            "record_id": record_id,
            "run_id": args.run_id,
            "condition_id": CONDITION_ID,
            "condition_hash": current_condition_hash,
            "stage": STAGE,
            "problem_id": request_row["task_id"],
            "outcome": request_row["outcome"],
            "u": result["u"],
            "status": result["status"],
            "detail": result["detail"],
            "seconds": result["seconds"],
            "created_at_utc": utc_now(),
        })


def stage_analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    g_rows = read_jsonl(Path(args.g_records))
    v_rows = {row["problem_id"]: row for row in read_jsonl(Path(args.v_records))}
    grades: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(Path(args.grade_records)):
        grades[(row["problem_id"], row["outcome"])] = row

    items = []
    for g_row in g_rows:
        task_id = g_row["problem_id"]
        v_row = v_rows.get(task_id)
        if v_row is None:
            raise RuntimeError(f"no verifier record for {task_id}")
        u0_row = grades.get((task_id, "u0"))
        if u0_row is None:
            raise RuntimeError(f"missing y0 grading record for {task_id}")
        # No y2 record means no report parsed, so M2 would deploy y0. That is not_run,
        # not a wrong answer and not a grader failure.
        u2_row = grades.get((task_id, "u2"))
        if u2_row is None and v_row["v_report"] is not None:
            raise RuntimeError(f"report parsed but no y2 grading record for {task_id}")
        items.append({
            "problem_id": task_id,
            "u0": u0_row["u"],
            "u0_status": u0_row["status"],
            "u0_detail": u0_row["detail"],
            "u2": None if u2_row is None else u2_row["u"],
            "u2_status": "not_run" if u2_row is None else u2_row["status"],
            "u2_detail": None if u2_row is None else u2_row["detail"],
            "registered_request": v_row["registered_request"],
            "report_parse_status": v_row["report_parse_status"],
            "verdict": None if v_row["v_report"] is None else v_row["v_report"]["verdict"],
            "g_finish_reason": g_row["outputs"][0]["usage"].get("finish_reason"),
            "v_finish_reason": v_row["usage"].get("finish_reason"),
        })

    # A grader_error nulls an outcome, and the contract refuses to validate a run on top
    # of one. not_run is different: it is a registered consequence of a failed parse.
    grader_errors = [
        item for item in items
        if item["u0_status"] == "grader_error" or item["u2_status"] == "grader_error"
    ]
    if grader_errors:
        raise RuntimeError(
            f"{len(grader_errors)} items carry a null outcome from grader error; "
            f"first: {grader_errors[0]['problem_id']}"
        )

    n = len(items)
    acc_g = sum(item["u0"] for item in items) / n
    scored_u2 = [item for item in items if item["u2"] is not None]
    mean_u2 = (
        sum(item["u2"] for item in scored_u2) / len(scored_u2) if scored_u2 else None
    )
    cross = collections.Counter((item["u0"], item["u2"]) for item in scored_u2)
    requests = collections.Counter(item["registered_request"] for item in items)
    parse_statuses = collections.Counter(item["report_parse_status"] for item in items)
    timeouts = collections.Counter(
        detail for item in items for detail in (item["u0_detail"], item["u2_detail"])
        if detail == "timeout"
    )

    correct = [item for item in items if item["u0"] == 1]
    incorrect = [item for item in items if item["u0"] == 0]
    over_correction = (
        sum(1 for item in correct if item["registered_request"] != "KEEP") / len(correct)
        if correct else None
    )
    coverage = (
        sum(1 for item in incorrect if item["registered_request"] != "KEEP") / len(incorrect)
        if incorrect else None
    )

    summary = {
        "run_id": args.run_id,
        "condition_id": CONDITION_ID,
        "dataset_id": DATASET_ID,
        "n": n,
        "acc_G": acc_g,
        "mean_u2": mean_u2,
        "mean_u2_denominator": len(scored_u2),
        "u2_not_run": n - len(scored_u2),
        "cross_u0_u2": {f"u0={u0},u2={u2}": count for (u0, u2), count in sorted(cross.items())},
        "cross_u0_u2_denominator": len(scored_u2),
        "error_complementarity": (
            (cross[(1, 0)] + cross[(0, 1)]) / len(scored_u2) if scored_u2 else None
        ),
        "request_distribution": dict(sorted(requests.items())),
        "over_correction_rate": over_correction,
        "over_correction_denominator": len(correct),
        "error_coverage_rate": coverage,
        "error_coverage_denominator": len(incorrect),
        "v_parse_status": dict(sorted(parse_statuses.items())),
        "grader_timeouts": timeouts["timeout"],
        "v_length_finishes": sum(1 for item in items if item["v_finish_reason"] == "length"),
        "g_length_finishes": sum(1 for item in items if item["g_finish_reason"] == "length"),
        "usage": {
            "generator": summarize_usage(row["outputs"][0]["usage"] for row in g_rows),
            "verifier": summarize_usage(row["usage"] for row in v_rows.values()),
        },
        "created_at_utc": utc_now(),
    }

    (output_dir / "gate0_items.jsonl").write_text(
        "".join(canonical_json(item) + "\n" for item in items), encoding="utf-8"
    )
    (output_dir / "gate0_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Gate 0 summary",
        "",
        f"run_id: {args.run_id}",
        f"n: {n}",
        "",
        "## Capability",
        f"- acc_G = {acc_g:.3f}",
        f"- mean(u2) = {'n/a' if mean_u2 is None else format(mean_u2, '.3f')} "
        f"(denominator {len(scored_u2)}; {n - len(scored_u2)} not_run)",
        f"- error complementarity = "
        f"{'n/a' if summary['error_complementarity'] is None else format(summary['error_complementarity'], '.3f')}",
        "",
        f"## (u0, u2) cross table (n={len(scored_u2)})",
        "",
        "|  | u2=1 | u2=0 |",
        "|---|---|---|",
        f"| **u0=1** | {cross[(1, 1)]} | {cross[(1, 0)]} |",
        f"| **u0=0** | {cross[(0, 1)]} | {cross[(0, 0)]} |",
        "",
        "## Requests",
        *(f"- {name}: {count}" for name, count in sorted(requests.items())),
        "",
        "## The hypothesis under test",
        f"- P(request != KEEP | u0 = 1) = "
        f"{'n/a' if over_correction is None else format(over_correction, '.3f')} "
        f"(denominator {len(correct)}); c003 measured 0.024",
        f"- P(request != KEEP | u0 = 0) = "
        f"{'n/a' if coverage is None else format(coverage, '.3f')} "
        f"(denominator {len(incorrect)}); c003 measured 0.774",
        "",
        "## Output quality",
        *(f"- v_parse_status {name}: {count}" for name, count in sorted(parse_statuses.items())),
        f"- verifier finish_reason=length: {summary['v_length_finishes']}",
        f"- generator finish_reason=length: {summary['g_length_finishes']}",
        f"- grader timeouts: {summary['grader_timeouts']}",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(canonical_json(summary))


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def stage_finalize(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle_dir)
    manifest_path = Path(args.dataset_manifest)
    git_commit = subprocess.run(
        ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    manifest = {
        "schema_version": "run-manifest-v1",
        "run_id": args.run_id,
        "condition_id": CONDITION_ID,
        "condition_hash": condition_hash(),
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "replicate_plan_id": REPLICATE_PLAN_ID,
        "stage": STAGE,
        "git_commit": git_commit,
        "node": platform.node(),
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("datasets", "huggingface-hub", "numpy", "PyYAML")
        },
        "grader_contract": "protocols/authority_v6/grader_bigcodebench_v1.yaml",
        "started_at_utc": args.started_at_utc,
        "completed_at_utc": utc_now(),
        "source_run_ids": list(args.source_run_id or []),
    }
    (bundle / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(canonical_json({"run_id": args.run_id, "git_commit": git_commit}))


# -------------------------------------------------------------------------- parser

def add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 0 for the code domain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--role", choices=("g", "v"), required=True)
    add_api_arguments(smoke)
    smoke.set_defaults(function=stage_smoke)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--output", required=True)
    add_api_arguments(generate)
    generate.set_defaults(function=stage_generate)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--g-records", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--output", required=True)
    add_api_arguments(verify)
    verify.set_defaults(function=stage_verify)

    grade_requests = subparsers.add_parser("grade-requests")
    grade_requests.add_argument("--g-records", required=True)
    grade_requests.add_argument("--v-records", required=True)
    grade_requests.add_argument("--run-id", required=True)
    grade_requests.add_argument("--output", required=True)
    grade_requests.set_defaults(function=stage_grade_requests)

    grade = subparsers.add_parser("grade")
    grade.add_argument("--requests", required=True)
    grade.add_argument("--results", required=True)
    grade.add_argument("--run-id", required=True)
    grade.add_argument("--output", required=True)
    grade.set_defaults(function=stage_grade)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--g-records", required=True)
    analyze.add_argument("--v-records", required=True)
    analyze.add_argument("--grade-records", required=True)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.set_defaults(function=stage_analyze)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--bundle-dir", required=True)
    finalize.add_argument("--dataset-manifest", required=True)
    finalize.add_argument("--started-at-utc", required=True)
    finalize.add_argument("--source-run-id", action="append")
    finalize.set_defaults(function=stage_finalize)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
