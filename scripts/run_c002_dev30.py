#!/usr/bin/env python3
"""Minimal staged runner for c002 on the frozen dev30 manifest.

The stages are deliberately separate so evidence is written before grading and
Controller inference never receives a path to the label file.
"""

from __future__ import annotations

import argparse
import collections
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import (  # noqa: E402
    build_cap_outcome_target,
    counterfactual_outcomes,
    derive_call_seed,
    effective_action,
    paired_policy_metrics,
)
from authority_allocation.single_sample_features import build_single_sample_payload  # noqa: E402
from authority_allocation.api import (  # noqa: E402
    api_call,
    format_repair_call,
    usage_from_response,
)
from authority_allocation.payloads import (  # noqa: E402
    parse_json_object,
    validate_controller,
    validate_v_report,
)
from authority_allocation.prompts import PROMPT_VARIABLE, render, split_prompt  # noqa: E402
from authority_allocation.records import (  # noqa: E402
    append_jsonl,
    completed_keys,
    read_jsonl,
)
from authority_allocation.resampling import (  # noqa: E402
    paired_bootstrap_difference,
    percentile,
    summarize_usage,
)
from authority_allocation.serialization import (  # noqa: E402
    canonical_json,
    sha256_bytes,
    sha256_file,
    stable_id,
    utc_now,
)
from authority_allocation.generated_token_logprobs import (  # noqa: E402
    build_generated_token_trace,
    derive_scalar_logprob_features,
    derive_scalar_logprob_features_v3,
)
from authority_allocation.single_sample_features_v2 import (  # noqa: E402
    build_single_sample_v2_payload,
    mathematical_answer_type,
)
from authority_allocation.single_sample_features_v3 import (  # noqa: E402
    build_single_sample_v3_payload,
)


CONDITION_ID = "c002"
CONDITION_PATH = ROOT / "conditions" / CONDITION_ID / "condition.yaml"
DATASET_ID = "math500-r6e4ed1-l3to5-s20260910"
DATASET_REPO = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
PROTOCOL_ID = "authority_v5"
REPLICATE_PLAN_ID = "r002"
CELL_ID = "V-A"
STAGE = "dev30"
FEATURE_SET_ID = "scalar_single_v1"
FEATURE_CONTRACT_PATH = ROOT / "protocols/authority_v5/features_scalar_single_v1.yaml"
G_RECORD_SCHEMA_VERSION = "g-record-single-v1"
V_RECORD_SCHEMA_VERSION = "v-record-v1"
EVIDENCE_SCHEMA_VERSION = "controller-evidence-record-single-v1"
EXPECTED_ROWS = 30
CAPTURE_LOGPROBS = False
RUN_REPLICATE_IDS = ("g01", "v01", "w01", "c01")
PROMPTS = {
    "g": ROOT / "conditions/c002/prompts/g_initial_v1.txt",
    "rewrite": ROOT / "conditions/c002/prompts/g_rewrite_v1.txt",
    "v": ROOT / "conditions/c002/prompts/v_report_v1.txt",
    "binary": ROOT / "conditions/c002/prompts/controller_binary_scalar_single_v1.txt",
    "ternary": ROOT / "conditions/c002/prompts/controller_ternary_scalar_single_v1.txt",
}

SCHEMAS = {
    "v": ROOT / "protocols/authority_v5/schemas/v_report_payload_v1.json",
    "binary": ROOT / "protocols/authority_v5/schemas/controller_binary_payload_v1.json",
    "ternary": ROOT / "protocols/authority_v5/schemas/controller_ternary_payload_v1.json",
}


def configure_c003_development300() -> None:
    """Select the frozen c003 profile without changing c002's default CLI."""

    global CONDITION_ID, CONDITION_PATH, DATASET_ID, PROTOCOL_ID, REPLICATE_PLAN_ID
    global STAGE, FEATURE_SET_ID, FEATURE_CONTRACT_PATH
    global G_RECORD_SCHEMA_VERSION, V_RECORD_SCHEMA_VERSION, EVIDENCE_SCHEMA_VERSION
    global EXPECTED_ROWS, CAPTURE_LOGPROBS, RUN_REPLICATE_IDS, PROMPTS, SCHEMAS

    CONDITION_ID = "c003"
    CONDITION_PATH = ROOT / "conditions/c003/condition.yaml"
    DATASET_ID = "math500-r6e4ed1-l3to5-dev300-s20260910"
    PROTOCOL_ID = "authority_v6"
    REPLICATE_PLAN_ID = "r003"
    STAGE = "pilot300"
    FEATURE_SET_ID = "scalar_single_v2"
    FEATURE_CONTRACT_PATH = ROOT / "protocols/authority_v6/features_scalar_single_v2.yaml"
    G_RECORD_SCHEMA_VERSION = "g-record-single-v2"
    V_RECORD_SCHEMA_VERSION = "v-record-v2"
    EVIDENCE_SCHEMA_VERSION = "controller-evidence-record-single-v2"
    EXPECTED_ROWS = 300
    CAPTURE_LOGPROBS = True
    RUN_REPLICATE_IDS = ("g01", "v01", "w01", "c01")
    PROMPTS = {
        "g": ROOT / "conditions/c002/prompts/g_initial_v1.txt",
        "rewrite": ROOT / "conditions/c002/prompts/g_rewrite_v1.txt",
        "v": ROOT / "conditions/c002/prompts/v_report_v1.txt",
        "binary": ROOT / "conditions/c003/prompts/controller_binary_scalar_single_v2.txt",
        "ternary": ROOT / "conditions/c003/prompts/controller_ternary_scalar_single_v2.txt",
    }
    SCHEMAS = {
        "v": ROOT / "protocols/authority_v5/schemas/v_report_payload_v1.json",
        "binary": ROOT / "protocols/authority_v5/schemas/controller_binary_payload_v1.json",
        "ternary": ROOT / "protocols/authority_v5/schemas/controller_ternary_payload_v1.json",
    }


def configure_c004_candidate_field() -> None:
    """Select c004, which rebuilds evidence from frozen c003 G/V traces."""

    global CONDITION_ID, CONDITION_PATH, DATASET_ID, PROTOCOL_ID, REPLICATE_PLAN_ID
    global STAGE, FEATURE_SET_ID, FEATURE_CONTRACT_PATH
    global G_RECORD_SCHEMA_VERSION, V_RECORD_SCHEMA_VERSION, EVIDENCE_SCHEMA_VERSION
    global EXPECTED_ROWS, CAPTURE_LOGPROBS, RUN_REPLICATE_IDS, PROMPTS, SCHEMAS

    CONDITION_ID = "c004"
    CONDITION_PATH = ROOT / "conditions/c004/condition.yaml"
    DATASET_ID = "math500-r6e4ed1-l3to5-dev300-s20260910"
    PROTOCOL_ID = "authority_v6"
    REPLICATE_PLAN_ID = "r003"
    STAGE = "pilot300"
    FEATURE_SET_ID = "scalar_single_v3"
    FEATURE_CONTRACT_PATH = ROOT / "protocols/authority_v6/features_scalar_single_v3.yaml"
    G_RECORD_SCHEMA_VERSION = "g-record-single-v2"
    V_RECORD_SCHEMA_VERSION = "v-record-v2"
    EVIDENCE_SCHEMA_VERSION = "controller-evidence-record-single-v3"
    EXPECTED_ROWS = 300
    CAPTURE_LOGPROBS = True
    RUN_REPLICATE_IDS = ("v01", "c01")
    PROMPTS = {
        "g": ROOT / "conditions/c002/prompts/g_initial_v1.txt",
        "rewrite": ROOT / "conditions/c002/prompts/g_rewrite_v1.txt",
        "v": ROOT / "conditions/c002/prompts/v_report_v1.txt",
        "binary": ROOT / "conditions/c004/prompts/controller_binary_scalar_single_v3.txt",
        "ternary": ROOT / "conditions/c004/prompts/controller_ternary_scalar_single_v3.txt",
    }
    SCHEMAS = {
        "v": ROOT / "protocols/authority_v5/schemas/v_report_payload_v1.json",
        "binary": ROOT / "protocols/authority_v5/schemas/controller_binary_payload_v1.json",
        "ternary": ROOT / "protocols/authority_v5/schemas/controller_ternary_payload_v1.json",
    }


def manifest_fields(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    problem_id = row.get("problem_id") or row.get("unique_id") or row.get("id")
    if problem_id is None:
        raise ValueError("manifest row needs problem_id/unique_id/id")
    metadata = {
        key: value
        for key, value in row.items()
        if key in {"subject", "level", "split"}
        and isinstance(value, (str, int, float, bool))
    }
    return str(problem_id), metadata


def load_source_rows(problem_ids: Iterable[str]) -> dict[str, dict[str, str]]:
    """Join an ID-only frozen manifest to the pinned dataset revision."""

    from datasets import load_dataset

    wanted = set(problem_ids)
    indexed: dict[str, dict[str, str]] = {}
    dataset = load_dataset(DATASET_REPO, revision=DATASET_REVISION, split="test")
    for row in dataset:
        problem_id = str(row["unique_id"])
        if problem_id not in wanted:
            continue
        problem = row.get("problem")
        gold = row.get("answer")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"source row {problem_id!r} has no problem text")
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError(f"source row {problem_id!r} has no reference answer")
        indexed[problem_id] = {"problem": problem, "gold": gold}
    missing = wanted.difference(indexed)
    if missing:
        raise ValueError(f"manifest IDs missing from pinned dataset: {sorted(missing)}")
    return indexed


def load_condition() -> dict[str, Any]:
    import yaml

    condition = yaml.safe_load(CONDITION_PATH.read_text(encoding="utf-8"))
    if condition.get("condition_id") != CONDITION_ID:
        raise ValueError("unexpected active condition")
    return condition


def condition_hash() -> str:
    condition = load_condition()
    excluded = set(condition["hash_policy"]["exclude"])

    def filtered(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: filtered(item) for key, item in value.items() if key not in excluded}
        if isinstance(value, list):
            return [filtered(item) for item in value]
        return value

    referenced: dict[str, str] = {}

    def find_paths(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                find_paths(item)
        elif isinstance(value, list):
            for item in value:
                find_paths(item)
        elif isinstance(value, str):
            candidate = ROOT / value
            if candidate.is_file():
                referenced[value] = sha256_file(candidate)

    find_paths(condition)
    schemas_directory = ROOT / condition["protocol"]["schemas_directory"]
    for schema_path in sorted(schemas_directory.glob("*.json")):
        relative = schema_path.relative_to(ROOT).as_posix()
        referenced[relative] = sha256_file(schema_path)
    payload = {"condition": filtered(condition), "referenced_files": referenced}
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def model_config(role: str) -> dict[str, Any]:
    condition = load_condition()
    if role == "g":
        return condition["generator"]
    if role == "v":
        return condition["verifiers"][0]
    if role == "c":
        return condition["controller"]
    raise ValueError(role)


def decoding_config(role: str) -> dict[str, Any]:
    config = model_config(role)
    if role == "g":
        return config["initial_decoding"]
    return config["decoding"]


def stage_smoke(args: argparse.Namespace) -> None:
    """Exercise one loaded model on a non-research input before data are read."""

    role = args.role
    config = model_config(role)
    decoding = decoding_config(role)
    schema: dict[str, Any] | None = None
    if role == "g":
        messages = [{"role": "user", "content": "Compute 2+2. End with a boxed final answer."}]
    elif role == "v":
        schema = json.loads(SCHEMAS["v"].read_text(encoding="utf-8"))
        messages = [{
            "role": "user",
            "content": (
                "For the non-research test problem 2+2, inspect the proposed answer 4. "
                "Return verdict, critique, a complete candidate_answer, and requested_action."
            ),
        }]
    else:
        schema = json.loads(SCHEMAS["binary"].read_text(encoding="utf-8"))
        messages = [{
            "role": "user",
            "content": (
                "This is a non-research interface test. Return one allowed binary authority_cap "
                "JSON object for a fully missing verifier report."
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
    elif role == "c":
        valid = validate_controller(parse_json_object(content), "binary")
    if not valid:
        raise RuntimeError(f"{role} compatibility smoke returned an invalid response")
    if CAPTURE_LOGPROBS and role in ("g", "v"):
        trace = usage.get("token_logprobs") or {}
        if trace.get("status") != "ok":
            raise RuntimeError(
                f"{role} compatibility smoke did not return an exact content logprob trace"
            )
    print(canonical_json({"role": role, "status": "passed", "usage": compact_usage(usage)}))


try:
    # math_verify's timeout derives from BaseException, so `except Exception` never catches it.
    from math_verify.errors import TimeoutException as MathVerifyTimeout
except ImportError:  # the local test environment has no math_verify
    class MathVerifyTimeout(BaseException):
        pass


def math_parse(text: str) -> tuple[list[Any], str | None]:
    from math_verify import parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    parsed = parse(
        text,
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=10,
        raise_on_error=True,
    )
    if not parsed:
        return [], None
    return parsed, str(parsed[0])


def math_parse_gold(answer: str) -> tuple[list[Any], str | None]:
    """Parse MATH-500's bare LaTeX answer field as a math expression."""

    return math_parse(f"${answer.strip()}$")


def audit_gold_answers(source_rows: Mapping[str, Mapping[str, str]]) -> None:
    for problem_id, row in source_rows.items():
        try:
            parsed, _ = math_parse_gold(row["gold"])
        except Exception as exc:
            raise RuntimeError(f"gold audit failed for {problem_id}") from exc
        if not parsed:
            raise RuntimeError(f"gold audit found an unparseable answer for {problem_id}")


def math_equivalent(left: str, right: str) -> tuple[bool | None, bool, bool]:
    from math_verify import verify

    left_parsed, _ = math_parse(left)
    right_parsed, _ = math_parse(right)
    left_ok, right_ok = bool(left_parsed), bool(right_parsed)
    if not left_ok or not right_ok:
        return None, left_ok, right_ok
    equivalent = verify(
        left_parsed,
        right_parsed,
        float_rounding=6,
        numeric_precision=15,
        strict=True,
        allow_set_relation_comp=False,
        timeout_seconds=10,
        raise_on_error=True,
    )
    return bool(equivalent), True, True


def grade_prediction(gold_text: str, prediction_text: str | None) -> tuple[int | None, str]:
    """Grade one prediction: (1, "ok"), (0, "ok"), (0, "model_invalid"), (0, "grader_timeout")
    or (None, "grader_error").

    math_verify's TimeoutException derives from BaseException, so the retry below never saw it
    and a timeout killed the whole grading stage (c049, job 65139505). A timeout while parsing
    the prediction or comparing it with the gold means the model's answer cannot be judged
    within the budget, and it is graded wrong as grader_timeout, as unparseable answers are.
    A timeout on the gold answer is a grader_error, which stops the analysis as before.
    Graded runs before this change never hit a timeout: it would have stopped them.
    """
    from math_verify import verify
    from math_verify.errors import TimeoutException

    if prediction_text is None:
        return 0, "model_invalid"
    for attempt in range(2):
        on_gold = True
        try:
            gold, _ = math_parse_gold(gold_text)
            if not gold:
                return None, "grader_error"
            on_gold = False
            prediction, _ = math_parse(prediction_text)
            if not prediction:
                return 0, "model_invalid"
            result = verify(
                gold,
                prediction,
                float_rounding=6,
                numeric_precision=15,
                strict=True,
                allow_set_relation_comp=False,
                timeout_seconds=10,
                raise_on_error=True,
            )
            return int(bool(result)), "ok"
        except TimeoutException:
            return (None, "grader_error") if on_gold else (0, "grader_timeout")
        except Exception:
            if attempt == 1:
                return None, "grader_error"
    raise AssertionError("unreachable grader retry state")


def answer_text(output: Mapping[str, Any]) -> str | None:
    """The text a stored math answer was extracted from, for grading or comparison.

    extracted_answer is str() of the sympy object math_parse returned, such as
    'sqrt(3)/3', '8*pi' or 'Matrix([[-18], [-49], [96]])'. Parsing that string again
    fails or keeps a fragment ('8*pi' becomes 8), which marked 45 correct c017 answers
    wrong. Graders and comparisons re-parse the original text instead; see docs/analysis/c018_report_format.md.
    None when no answer was extracted.
    """

    if output.get("extracted_answer") is None:
        return None
    return output.get("raw_output")


def compact_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "latency_ms": usage.get("latency_ms"),
        "finish_reason": usage.get("finish_reason"),
        "retries": usage.get("retries", 0),
    }


def model_identity(role: str, prompt_path: Path) -> dict[str, Any]:
    config = model_config(role)
    identity = {
        "model_id": config["model_id"],
        "revision": config["revision"],
        "server_path": config["server_path"],
        "backend": config["backend"],
        "prompt_sha256": sha256_file(prompt_path),
        "chat_template_sha256": config["chat_template_hash"],
    }
    if role == "g":
        identity.update({"dtype": config["dtype"], "quantization": config["quantization"]})
    return identity


def stage_generate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    system, template = split_prompt(PROMPTS["g"])
    config = model_config("g")
    decoding = config["initial_decoding"]
    current_condition_hash = condition_hash()
    memberships = [manifest_fields(row) for row in read_jsonl(Path(args.manifest))]
    source_rows = load_source_rows(problem_id for problem_id, _ in memberships)
    audit_gold_answers(source_rows)
    for problem_id, metadata in memberships:
        if (problem_id,) in done:
            continue
        problem = source_rows[problem_id]["problem"]
        seed = derive_call_seed(REPLICATE_PLAN_ID, "generator", "g01", 20260910, "G", problem_id, 0)
        content, _, usage = api_call(
            api_base=args.api_base,
            served_model=args.served_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": render(template, problem=problem)}],
            decoding=decoding,
            seed=seed,
            thinking=config["thinking"],
        )
        _, extracted = math_parse(content)
        output_record = {
            "sample_index": 0,
            "call_seed": seed,
            "raw_output": content,
            "extracted_answer": extracted,
            "parse_status": "ok" if extracted is not None else "model_invalid",
            "usage": compact_usage(usage),
        }
        if CAPTURE_LOGPROBS:
            output_record["token_logprobs"] = usage.get("token_logprobs")
        row = {
            "schema_version": G_RECORD_SCHEMA_VERSION,
            "record_id": stable_id(args.run_id, problem_id),
            "run_id": args.run_id,
            "condition_id": CONDITION_ID,
            "condition_hash": current_condition_hash,
            "dataset_id": DATASET_ID,
            "replicate_id": "g01",
            "stage": STAGE,
            "problem_id": problem_id,
            "problem_text": problem,
            "problem_metadata": metadata,
            "generator": model_identity("g", PROMPTS["g"]),
            "outputs": [output_record],
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
        problem_id = g_row["problem_id"]
        if (problem_id, CELL_ID) in done:
            continue
        y0 = g_row["outputs"][0]["raw_output"]
        seed = derive_call_seed(REPLICATE_PLAN_ID, "verifier", "v01", 20260910, CELL_ID, problem_id, 0)
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
                {"role": "user", "content": "Repair only the JSON format. Return exactly one object matching the required schema."},
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
                retry_usage["input_tokens"] = (usage.get("input_tokens") or 0) + (retry_usage.get("input_tokens") or 0)
                retry_usage["reasoning_tokens"] = (usage.get("reasoning_tokens") or 0) + (retry_usage.get("reasoning_tokens") or 0)
                retry_usage["output_tokens"] = (usage.get("output_tokens") or 0) + (retry_usage.get("output_tokens") or 0)
                retry_usage["latency_ms"] = (usage.get("latency_ms") or 0) + (retry_usage.get("latency_ms") or 0)
                content, reasoning, usage = content_retry, reasoning_retry, retry_usage
                report = parse_json_object(content)
                parse_status = "retry_ok"
        if not validate_v_report(report):
            report = None
            parse_status = "fallback"
        registered_request = "KEEP" if report is None else report["requested_action"]
        row = {
            "schema_version": V_RECORD_SCHEMA_VERSION,
            "record_id": stable_id(args.run_id, CELL_ID, problem_id),
            "run_id": args.run_id,
            "source_g_run_id": g_row["run_id"],
            "condition_id": CONDITION_ID,
            "condition_hash": current_condition_hash,
            "replicate_id": "v01",
            "stage": STAGE,
            "cell_id": CELL_ID,
            "problem_id": problem_id,
            "verifier": model_identity("v", PROMPTS["v"]),
            "call_seed": seed,
            "raw_output": content,
            "reasoning_content": reasoning,
            "v_report": report,
            "report_parse_status": parse_status,
            "registered_request": registered_request,
            "usage": {
                **compact_usage(usage),
                "reasoning_tokens": usage.get("reasoning_tokens"),
            },
            "created_at_utc": utc_now(),
        }
        if CAPTURE_LOGPROBS:
            row["token_logprobs"] = usage.get("token_logprobs")
        append_jsonl(output, row)


def index_unique(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise ValueError(f"duplicate {key}: {value}")
        indexed[value] = row
    return indexed


def stage_evidence(args: argparse.Namespace) -> None:
    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "cell_id"))
    g_by_problem = index_unique(read_jsonl(Path(args.g_records)), "problem_id")
    contract_path = FEATURE_CONTRACT_PATH
    contract_hash = sha256_file(contract_path)
    current_condition_hash = condition_hash()
    for v_row in read_jsonl(Path(args.v_records)):
        problem_id = v_row["problem_id"]
        if (problem_id, v_row["cell_id"]) in done:
            continue
        g_row = g_by_problem[problem_id]
        y0 = g_row["outputs"][0]
        report = v_row["v_report"]
        y0_extractable = y0["parse_status"] == "ok"
        candidate_extractable: bool | None = None
        equivalent: bool | None = None
        parsed_y0_values: list[Any] = []
        parsed_y2_values: list[Any] = []
        if report is not None:
            equivalent, parsed_y0, parsed_y2 = math_equivalent(
                y0["raw_output"], report["candidate_answer"]
            )
            if parsed_y0 != y0_extractable:
                raise ValueError(f"G parse status drift for {problem_id}")
            candidate_extractable = parsed_y2
            if parsed_y0 and parsed_y2:
                parsed_y0_values, _ = math_parse(y0["raw_output"])
                parsed_y2_values, _ = math_parse(report["candidate_answer"])
        if FEATURE_SET_ID in {"scalar_single_v2", "scalar_single_v3"}:
            derive_features = (
                derive_scalar_logprob_features_v3
                if FEATURE_SET_ID == "scalar_single_v3"
                else derive_scalar_logprob_features
            )
            derived = derive_features(
                g_text=y0["raw_output"],
                g_trace=y0.get("token_logprobs") or {"status": "missing"},
                v_text=v_row["raw_output"],
                v_trace=v_row.get("token_logprobs") or {"status": "missing"},
                v_report_parse_status=v_row["report_parse_status"],
            )
            g_type = mathematical_answer_type(parsed_y0_values[0]) if parsed_y0_values else None
            v_type = mathematical_answer_type(parsed_y2_values[0]) if parsed_y2_values else None
            build_payload = (
                build_single_sample_v3_payload
                if FEATURE_SET_ID == "scalar_single_v3"
                else build_single_sample_v2_payload
            )
            payload = build_payload(
                y0_extractable=y0_extractable,
                v_report=report,
                candidate_extractable=candidate_extractable,
                answers_equivalent=equivalent,
                gv_answer_type_match=(
                    None if g_type is None or v_type is None else g_type == v_type
                ),
                **derived,
            )
        else:
            payload = build_single_sample_payload(
                y0_extractable=y0_extractable,
                v_report=report,
                candidate_extractable=candidate_extractable,
                answers_equivalent=equivalent,
            )
        evidence_hash = sha256_bytes(canonical_json(payload).encode("utf-8"))
        row = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "record_id": stable_id(args.run_id, v_row["cell_id"], problem_id),
            "run_id": args.run_id,
            "source_g_run_id": g_row["run_id"],
            "source_v_run_id": v_row["run_id"],
            "condition_id": CONDITION_ID,
            "condition_hash": current_condition_hash,
            "replicate_id": "v01",
            "stage": STAGE,
            "cell_id": v_row["cell_id"],
            "problem_id": problem_id,
            "feature_set_id": FEATURE_SET_ID,
            "feature_contract_sha256": contract_hash,
            "features": payload["features"],
            "missingness": payload["missingness"],
            "evidence_sha256": evidence_hash,
            "created_at_utc": utc_now(),
        }
        append_jsonl(output, row)


def stage_rewrite(args: argparse.Namespace) -> None:
    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "cell_id"))
    g_by_problem = index_unique(read_jsonl(Path(args.g_records)), "problem_id")
    system, template = split_prompt(PROMPTS["rewrite"])
    config = model_config("g")
    decoding = config["rewrite_decoding"]
    current_condition_hash = condition_hash()
    for v_row in read_jsonl(Path(args.v_records)):
        problem_id = v_row["problem_id"]
        cell_id = v_row["cell_id"]
        if (problem_id, cell_id) in done:
            continue
        g_row = g_by_problem[problem_id]
        registered_request = v_row["registered_request"]
        if registered_request == "KEEP":
            call_seed = None
            raw_output = None
            extracted = None
            status = "not_requested"
            fallback = None
            usage = None
        else:
            report = v_row["v_report"]
            if report is None:
                raise ValueError("non-KEEP registered request without V report")
            call_seed = derive_call_seed(
                REPLICATE_PLAN_ID, "rewrite", "w01", 20260910, cell_id, problem_id, 0
            )
            try:
                raw_output, _, call_usage = api_call(
                    api_base=args.api_base,
                    served_model=args.served_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": render(
                            template,
                            problem=g_row["problem_text"],
                            y0=g_row["outputs"][0]["raw_output"],
                            critique=report["critique"],
                        )},
                    ],
                    decoding=decoding,
                    seed=call_seed,
                    thinking=config["thinking"],
                )
                _, extracted = math_parse(raw_output)
                status = "ok" if extracted is not None else "model_invalid"
                fallback = None if status == "ok" else "KEEP"
                usage = compact_usage(call_usage)
            except Exception:
                raw_output = None
                extracted = None
                status = "call_error"
                fallback = "KEEP"
                usage = None
        row = {
            "schema_version": "rewrite-record-v1",
            "record_id": stable_id(args.run_id, cell_id, problem_id),
            "run_id": args.run_id,
            "source_v_run_id": v_row["run_id"],
            "source_v_record_id": v_row["record_id"],
            "condition_id": CONDITION_ID,
            "condition_hash": current_condition_hash,
            "replicate_id": "w01",
            "stage": STAGE,
            "cell_id": cell_id,
            "problem_id": problem_id,
            "registered_request": registered_request,
            "rewrite_ran": registered_request != "KEEP",
            "call_seed": call_seed,
            "raw_output": raw_output,
            "extracted_answer": extracted,
            "status": status,
            "failure_fallback_action": fallback,
            "usage": usage,
            "created_at_utc": utc_now(),
        }
        append_jsonl(output, row)


def stage_controller(args: argparse.Namespace) -> None:
    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "cell_id", "menu", "replicate_id"))
    v_by_problem = index_unique(read_jsonl(Path(args.v_records)), "problem_id")
    config = model_config("c")
    decoding = config["decoding"]
    current_condition_hash = condition_hash()
    for evidence in read_jsonl(Path(args.evidence_records)):
        problem_id = evidence["problem_id"]
        v_row = v_by_problem[problem_id]
        payload = {"features": evidence["features"], "missingness": evidence["missingness"]}
        if sha256_bytes(canonical_json(payload).encode("utf-8")) != evidence["evidence_sha256"]:
            raise ValueError(f"evidence hash mismatch for {problem_id}")
        for menu in ("binary", "ternary"):
            if (problem_id, evidence["cell_id"], menu, "c01") in done:
                continue
            system, template = split_prompt(PROMPTS[menu])
            schema = json.loads(SCHEMAS[menu].read_text(encoding="utf-8"))
            seed = derive_call_seed(
                REPLICATE_PLAN_ID,
                "controller",
                "c01",
                2026091401,
                evidence["cell_id"],
                problem_id,
                0,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": render(template, controller_input_json=canonical_json(payload))},
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
            decision = parse_json_object(content)
            parse_status = "ok"
            if not validate_controller(decision, menu):
                repair_messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Repair only the JSON format. Return exactly one allowed authority_cap object."},
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
                    retry_usage["input_tokens"] = (usage.get("input_tokens") or 0) + (retry_usage.get("input_tokens") or 0)
                    retry_usage["reasoning_tokens"] = (usage.get("reasoning_tokens") or 0) + (retry_usage.get("reasoning_tokens") or 0)
                    retry_usage["output_tokens"] = (usage.get("output_tokens") or 0) + (retry_usage.get("output_tokens") or 0)
                    retry_usage["latency_ms"] = (usage.get("latency_ms") or 0) + (retry_usage.get("latency_ms") or 0)
                    content, reasoning, usage = content_retry, reasoning_retry, retry_usage
                    decision = parse_json_object(content)
                    parse_status = "retry_ok"
            if not validate_controller(decision, menu):
                decision = {"authority_cap": "M0"}
                parse_status = "fallback_m0"
            cap = decision["authority_cap"]
            actual_action = effective_action(cap, v_row["registered_request"]).name
            row = {
                "schema_version": "controller-decision-record-v1",
                "record_id": stable_id(args.run_id, evidence["cell_id"], problem_id, menu, "c01"),
                "run_id": args.run_id,
                "condition_id": CONDITION_ID,
                "condition_hash": current_condition_hash,
                "replicate_id": "c01",
                "stage": STAGE,
                "cell_id": evidence["cell_id"],
                "problem_id": problem_id,
                "controller_id": config["controller_id"],
                "controller_kind": config["kind"],
                "controller_revision_or_checkpoint": config["revision"],
                "menu": menu,
                "source_evidence_record_id": evidence["record_id"],
                "evidence_sha256": evidence["evidence_sha256"],
                "registered_request": v_row["registered_request"],
                "input_payload_sha256": sha256_bytes(canonical_json(payload).encode("utf-8")),
                "call_seed": seed,
                "raw_output": content,
                "reasoning_content": reasoning,
                "final_content": content or None,
                "authority_cap": cap,
                "effective_action": actual_action,
                "parse_status": parse_status,
                "usage": {
                    **compact_usage(usage),
                    "reasoning_tokens": usage.get("reasoning_tokens"),
                },
                "created_at_utc": utc_now(),
            }
            append_jsonl(output, row)


def stage_grade(args: argparse.Namespace) -> None:
    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "cell_id"))
    memberships = [manifest_fields(row) for row in read_jsonl(Path(args.manifest))]
    source_rows = load_source_rows(problem_id for problem_id, _ in memberships)
    audit_gold_answers(source_rows)
    manifest = {problem_id: source_rows[problem_id]["gold"] for problem_id, _ in memberships}
    g_by_problem = index_unique(read_jsonl(Path(args.g_records)), "problem_id")
    v_by_problem = index_unique(read_jsonl(Path(args.v_records)), "problem_id")
    rewrite_by_problem = index_unique(read_jsonl(Path(args.rewrite_records)), "problem_id")
    evidence_by_problem = index_unique(read_jsonl(Path(args.evidence_records)), "problem_id")
    current_condition_hash = condition_hash()
    for problem_id, v_row in v_by_problem.items():
        cell_id = v_row["cell_id"]
        if (problem_id, cell_id) in done:
            continue
        g_row = g_by_problem[problem_id]
        rewrite = rewrite_by_problem[problem_id]
        evidence = evidence_by_problem[problem_id]
        gold = manifest[problem_id]
        y0 = g_row["outputs"][0]["raw_output"]
        u0, status0 = grade_prediction(gold, y0)
        if rewrite["status"] == "ok":
            u1, status1 = grade_prediction(gold, rewrite["raw_output"])
        else:
            u1, status1 = None, "not_run"
        report = v_row["v_report"]
        if report is not None:
            u2, status2 = grade_prediction(gold, report["candidate_answer"])
        else:
            u2, status2 = None, "not_run"
        outcomes = counterfactual_outcomes(
            v_row["registered_request"],
            u0=u0,
            u1=u1,
            u2=u2,
            rewrite_status=rewrite["status"],
        )
        row = {
            "schema_version": "counterfactual-record-v1",
            "record_id": stable_id(args.run_id, cell_id, problem_id),
            "run_id": args.run_id,
            "source_g_run_id": g_row["run_id"],
            "source_v_run_id": v_row["run_id"],
            "source_rewrite_run_id": rewrite["run_id"] if rewrite["rewrite_ran"] else None,
            "source_evidence_record_id": evidence["record_id"],
            "evidence_sha256": evidence["evidence_sha256"],
            "condition_id": CONDITION_ID,
            "condition_hash": current_condition_hash,
            "stage": STAGE,
            "cell_id": cell_id,
            "problem_id": problem_id,
            "registered_request": v_row["registered_request"],
            "rewrite_status": rewrite["status"],
            "u0": u0,
            "u1": u1,
            "u2": u2,
            "z0": outcomes["z0"],
            "z1": outcomes["z1"],
            "z2": outcomes["z2"],
            "grader_status": {"y0": status0, "y1": status1, "y2": status2},
            "created_at_utc": utc_now(),
        }
        append_jsonl(output, row)


def metric_row(rows: list[dict[str, Any]], outcomes: list[int]) -> dict[str, float | int]:
    metrics = paired_policy_metrics([row["u0"] for row in rows], outcomes)
    return {
        "n": metrics["n"],
        "accuracy": metrics["accuracy_policy"],
        "repair": metrics["repair"],
        "harm": metrics["harm"],
        "delta_vs_m0": metrics["delta_accuracy"],
    }


def stage_analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_jsonl(Path(args.counterfactuals)), key=lambda row: row["problem_id"])
    if any(row[key] not in (0, 1) for row in rows for key in ("u0", "z0", "z1", "z2")):
        raise ValueError("analysis blocked: missing grader outcome")
    decisions = read_jsonl(Path(args.controller_records))
    g_rows = read_jsonl(Path(args.g_records)) if getattr(args, "g_records", None) else []
    v_rows = read_jsonl(Path(args.v_records)) if getattr(args, "v_records", None) else []
    rewrite_rows = (
        read_jsonl(Path(args.rewrite_records)) if getattr(args, "rewrite_records", None) else []
    )
    evidence_rows = (
        read_jsonl(Path(args.evidence_records)) if getattr(args, "evidence_records", None) else []
    )
    decision_index = {
        (row["problem_id"], row["menu"], row["replicate_id"]): row for row in decisions
    }
    if len(decision_index) != len(decisions):
        raise ValueError("duplicate controller decision")

    outcomes: dict[str, list[int]] = {
        "M0": [row["z0"] for row in rows],
        "M1": [row["z1"] for row in rows],
        "M2": [row["z2"] for row in rows],
    }
    for policy, menu in (("C2", "binary"), ("C3", "ternary")):
        selected: list[int] = []
        for row in rows:
            decision = decision_index[(row["problem_id"], menu, "c01")]
            selected.append(row[f"z{decision['authority_cap'][1]}"])
        outcomes[policy] = selected

    policy_results = {name: metric_row(rows, values) for name, values in outcomes.items()}
    oracle_rows = []
    for row in rows:
        target = build_cap_outcome_target(row["z0"], row["z1"], row["z2"])
        oracle_rows.append({
            "schema_version": "sample-oracle-record-v1",
            "problem_id": row["problem_id"],
            # Historical field retained for compatibility with the validated
            # c002 report. New training contracts call this cap_outcomes.
            "outcomes": target["cap_outcomes"],
            "oracle_correct": target["oracle_correct"],
            "oracle_actions": target["oracle_actions"],
            "minimal_oracle_action": target["minimal_oracle_action"],
            "oracle_class": target["oracle_class"],
        })
    oracle_outcomes = [row["oracle_correct"] for row in oracle_rows]
    oracle_result = metric_row(rows, oracle_outcomes)
    repair_overlap = {
        "M1_only": sum(
            row["z0"] == 0 and row["z1"] == 1 and row["z2"] == 0 for row in rows
        ),
        "M2_only": sum(
            row["z0"] == 0 and row["z1"] == 0 and row["z2"] == 1 for row in rows
        ),
        "M1_and_M2": sum(
            row["z0"] == 0 and row["z1"] == 1 and row["z2"] == 1 for row in rows
        ),
    }
    outcome_pattern_counts = [
        {"z0": pattern[0], "z1": pattern[1], "z2": pattern[2], "count": count}
        for pattern, count in sorted(
            collections.Counter((row["z0"], row["z1"], row["z2"]) for row in rows).items()
        )
    ]
    oracle_class_counts = {
        name: sum(row["oracle_class"] == name for row in oracle_rows)
        for name in (
            "unfixable",
            "M0",
            "M1",
            "M2",
            "M0_and_M1",
            "M0_and_M2",
            "M1_and_M2",
            "M0_and_M1_and_M2",
        )
    }
    minimal_oracle_action_counts = {
        name: sum(row["minimal_oracle_action"] == name for row in oracle_rows)
        for name in ("M0", "M1", "M2")
    }
    minimal_oracle_action_counts["unfixable"] = sum(
        row["minimal_oracle_action"] is None for row in oracle_rows
    )
    boundary_transitions = {
        "M0_to_M1": {
            "gain": sum(row["z0"] == 0 and row["z1"] == 1 for row in rows),
            "harm": sum(row["z0"] == 1 and row["z1"] == 0 for row in rows),
            "net": sum(row["z1"] - row["z0"] for row in rows),
        },
        "M1_to_M2": {
            "gain": sum(row["z1"] == 0 and row["z2"] == 1 for row in rows),
            "harm": sum(row["z1"] == 1 and row["z2"] == 0 for row in rows),
            "net": sum(row["z2"] - row["z1"] for row in rows),
        },
    }
    diagnostics = {
        "sample_oracle": oracle_result,
        "sample_oracle_correct": sum(oracle_outcomes),
        "unfixable": sum(outcome == 0 for outcome in oracle_outcomes),
        "oracle_gap": {
            name: oracle_result["accuracy"] - result["accuracy"]
            for name, result in policy_results.items()
        },
        "repair_overlap": repair_overlap,
        "oracle_class_counts": oracle_class_counts,
        "minimal_oracle_action_counts": minimal_oracle_action_counts,
        "boundary_transitions": boundary_transitions,
        "harm_counts": {
            "M1": sum(row["z0"] == 1 and row["z1"] == 0 for row in rows),
            "M2": sum(row["z0"] == 1 and row["z2"] == 0 for row in rows),
        },
        "raw_counterfactual_checks": {
            "v_candidate_wrong_when_g_correct": sum(
                row["u0"] == 1 and row.get("u2") == 0 for row in rows
            ),
            "override_wrong_when_g_correct": sum(
                row["u0"] == 1
                and row["registered_request"] == "OVERRIDE"
                and row.get("u2") == 0
                for row in rows
            ),
            "rewrite_wrong_when_g_correct": sum(
                row["u0"] == 1 and row.get("u1") == 0 for row in rows
            ),
            "g_and_v_correct_rewrite_wrong": sum(
                row["u0"] == 1 and row.get("u2") == 1 and row.get("u1") == 0
                for row in rows
            ),
            "m1_only_override_rewrite": sum(
                row["registered_request"] == "OVERRIDE"
                and row["u0"] == 0
                and row.get("u1") == 1
                and row.get("u2") == 0
                for row in rows
            ),
        },
        "outcome_pattern_counts": outcome_pattern_counts,
    }
    comparisons = {
        "C3_minus_M0": paired_bootstrap_difference(outcomes["C3"], outcomes["M0"], seed=20260913, repetitions=10000),
        "C3_minus_M1": paired_bootstrap_difference(outcomes["C3"], outcomes["M1"], seed=20260914, repetitions=10000),
        "C3_minus_M2": paired_bootstrap_difference(outcomes["C3"], outcomes["M2"], seed=20260915, repetitions=10000),
        "C3_minus_C2": paired_bootstrap_difference(outcomes["C3"], outcomes["C2"], seed=20260916, repetitions=10000),
    }
    request_counts = {name: 0 for name in ("KEEP", "CRITIQUE_REVISE", "OVERRIDE")}
    for row in rows:
        request_counts[row["registered_request"]] += 1
    controller_fallbacks = sum(row["parse_status"] == "fallback_m0" for row in decisions)
    grader_errors = sum(
        status == "grader_error"
        for row in rows
        for status in row["grader_status"].values()
    )
    identity_passed = all(
        abs(result["delta_vs_m0"] - (result["repair"] - result["harm"])) < 1e-12
        for result in policy_results.values()
    )
    trace_failures = {
        "G": sum(
            row["outputs"][0].get("token_logprobs", {}).get("status") != "ok"
            for row in g_rows
        ) if CAPTURE_LOGPROBS else 0,
        "V": sum(
            row.get("token_logprobs", {}).get("status") != "ok" for row in v_rows
        ) if CAPTURE_LOGPROBS else 0,
    }
    allowed_trace_statuses = {"ok", "missing", "text_mismatch"}
    trace_complete = (
        not CAPTURE_LOGPROBS
        or (
            len(g_rows) == EXPECTED_ROWS
            and len(v_rows) == EXPECTED_ROWS
            and all(
                isinstance(row["outputs"][0].get("token_logprobs"), dict)
                and row["outputs"][0]["token_logprobs"].get("status")
                in allowed_trace_statuses
                for row in g_rows
            )
            and all(
                isinstance(row.get("token_logprobs"), dict)
                and row["token_logprobs"].get("status") in allowed_trace_statuses
                for row in v_rows
            )
        )
    )
    evidence_complete = (
        not CAPTURE_LOGPROBS
        or (
            len(evidence_rows) == EXPECTED_ROWS
            and all(row.get("feature_set_id") == FEATURE_SET_ID for row in evidence_rows)
        )
    )
    qa_passed = (
        len(rows) == EXPECTED_ROWS
        and len(decisions) == EXPECTED_ROWS * 2
        and grader_errors == 0
        and identity_passed
        and trace_complete
        and evidence_complete
    )
    qa = {
        "status": "passed" if qa_passed else "failed",
        "expected_rows": EXPECTED_ROWS,
        "expected_controller_rows": EXPECTED_ROWS * 2,
        "counterfactual_rows": len(rows),
        "controller_rows": len(decisions),
        "grader_errors": grader_errors,
        "controller_fallbacks": controller_fallbacks,
        "repair_harm_identity_passed": identity_passed,
        "token_logprob_trace_failures": trace_failures,
        "token_logprob_traces_complete": trace_complete,
        "evidence_complete": evidence_complete,
    }
    cap_counts = {
        menu: {
            cap: sum(row["menu"] == menu and row["authority_cap"] == cap for row in decisions)
            for cap in (("M0", "M2") if menu == "binary" else ("M0", "M1", "M2"))
        }
        for menu in ("binary", "ternary")
    }
    effective_action_counts = {
        action: sum(row.get("effective_action") == action for row in decisions)
        for action in ("KEEP", "CRITIQUE_REVISE", "OVERRIDE")
    }
    usage_summary = {
        "G_initial": summarize_usage(
            row["outputs"][0].get("usage") for row in g_rows
        ),
        "V": summarize_usage(row.get("usage") for row in v_rows),
        "G_rewrite": summarize_usage(row.get("usage") for row in rewrite_rows),
        "Controller": summarize_usage(row.get("usage") for row in decisions),
    }
    feature_missingness = {
        name: sum(row["missingness"].get(name, True) for row in evidence_rows)
        for name in sorted(evidence_rows[0]["features"])
    } if evidence_rows else {}
    metrics = {
        "stage": STAGE,
        "condition_id": CONDITION_ID,
        "feature_set_id": FEATURE_SET_ID,
        "cell_id": CELL_ID,
        "request_counts": request_counts,
        "policies": policy_results,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "controller_cap_counts": cap_counts,
        "effective_action_counts": effective_action_counts,
        "usage": usage_summary,
        "feature_missingness_counts": feature_missingness,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "oracle_assignments.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in oracle_rows), encoding="utf-8"
    )
    if FEATURE_SET_ID in {"scalar_single_v2", "scalar_single_v3"}:
        target_rows = []
        for row in rows:
            target = build_cap_outcome_target(row["z0"], row["z1"], row["z2"])
            target_rows.append({
                "schema_version": "controller-training-target-v1",
                "problem_id": row["problem_id"],
                "cell_id": row["cell_id"],
                "source_evidence_record_id": row["source_evidence_record_id"],
                "evidence_sha256": row["evidence_sha256"],
                "cap_outcomes": target["cap_outcomes"],
                "oracle_actions": target["oracle_actions"],
                "minimal_oracle_action": target["minimal_oracle_action"],
            })
        (output_dir / "controller_training_targets.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in target_rows),
            encoding="utf-8",
        )
    (output_dir / "qa_report.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    best_fixed_name = max(("M0", "M1", "M2"), key=lambda name: policy_results[name]["accuracy"])
    best_fixed = policy_results[best_fixed_name]["accuracy"]
    c3_difference = policy_results["C3"]["accuracy"] - best_fixed
    if c3_difference > 0:
        direction = "正向"
    elif c3_difference < 0:
        direction = "负向"
    else:
        direction = "持平"

    if CONDITION_ID in {"c003", "c004"}:
        scope_note = (
            f"> 这是{EXPECTED_ROWS}题开发实验，可用于调参与训练，不是独立测试证据。"
        )
        limitation_note = (
            "本结果只有一个Controller seed，无法评估seed稳定性；该数据允许开发和调参，"
            "因此不得作为独立或确认性测试证据。"
        )
    else:
        scope_note = (
            f"> 这是{EXPECTED_ROWS}题开发实验，只用于判断管线和方向，不是正式研究结论。"
        )
        limitation_note = (
            "本结果只有一个Controller seed，无法评估seed稳定性；且dev30参与工程调试，"
            "不得作为确认性证据。"
        )

    lines = [
        f"# {CONDITION_ID} {STAGE} diagnostic result",
        "",
        f"结论：C3 相对本批次最佳固定策略 {best_fixed_name} 的准确率差为 {c3_difference:+.3f}，方向为{direction}。",
        "",
        scope_note,
        "",
        "## 实验设置",
        "",
        f"数据：固定 {STAGE}（注册用途 development）；G=Qwen2.5-7B-Instruct；"
        f"V=Qwen3-8B thinking；C=Qwen3-14B thinking；feature={FEATURE_SET_ID}；"
        "Controller seed=c01。",
        "",
        "## 请求分布",
        "",
        f"KEEP={request_counts['KEEP']}，CRITIQUE_REVISE={request_counts['CRITIQUE_REVISE']}，OVERRIDE={request_counts['OVERRIDE']}。",
        "",
        "## 主结果",
        "",
        "|策略|Accuracy|Repair|Harm|Δ vs M0|",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("M0", "M1", "M2", "C2", "C3"):
        result = policy_results[name]
        lines.append(
            f"|{name}|{result['accuracy']:.3f}|{result['repair']:.3f}|"
            f"{result['harm']:.3f}|{result['delta_vs_m0']:+.3f}|"
        )
    lines.extend(["", "## 核心比较", "", "|比较|差值|95% bootstrap CI|", "|---|---:|---:|"])
    for name, result in comparisons.items():
        lines.append(
            f"|{name.replace('_', ' ')}|{result['difference']:+.3f}|"
            f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]|"
        )
    lines.extend([
        "",
        "## Sample oracle（仅诊断）",
        "",
        f"Oracle={diagnostics['sample_oracle_correct']}/{len(rows)} "
        f"({oracle_result['accuracy']:.3f})；不可修复={diagnostics['unfixable']}。",
        "",
        f"Oracle gap：M0={diagnostics['oracle_gap']['M0']:.3f}，"
        f"M1={diagnostics['oracle_gap']['M1']:.3f}，"
        f"M2={diagnostics['oracle_gap']['M2']:.3f}，"
        f"C2={diagnostics['oracle_gap']['C2']:.3f}，"
        f"C3={diagnostics['oracle_gap']['C3']:.3f}。",
        "",
        f"修复重叠：仅 M1={repair_overlap['M1_only']}，"
        f"仅 M2={repair_overlap['M2_only']}，"
        f"M1/M2 均成功={repair_overlap['M1_and_M2']}。",
        "",
        "逐题上帝视角类别："
        f"仅 M0={oracle_class_counts['M0']}，"
        f"仅 M1={oracle_class_counts['M1']}，"
        f"仅 M2={oracle_class_counts['M2']}，"
        f"M0/M1={oracle_class_counts['M0_and_M1']}，"
        f"M0/M2={oracle_class_counts['M0_and_M2']}，"
        f"M1/M2={oracle_class_counts['M1_and_M2']}，"
        f"三者均可={oracle_class_counts['M0_and_M1_and_M2']}，"
        f"三者均错={oracle_class_counts['unfixable']}。",
        "",
        "权限边界："
        f"M0→M1 gain={boundary_transitions['M0_to_M1']['gain']}、"
        f"harm={boundary_transitions['M0_to_M1']['harm']}、"
        f"net={boundary_transitions['M0_to_M1']['net']:+d}；"
        f"M1→M2 gain={boundary_transitions['M1_to_M2']['gain']}、"
        f"harm={boundary_transitions['M1_to_M2']['harm']}、"
        f"net={boundary_transitions['M1_to_M2']['net']:+d}。",
        "",
        "存在多个正确动作时，逐题文件中的 minimal_oracle_action 以最小权限 "
        "M0 > M1 > M2 的优先顺序打破并列。",
        "",
        f"Harm：M1={diagnostics['harm_counts']['M1']}，"
        f"M2={diagnostics['harm_counts']['M2']}；"
        f"G、V 均正确但 M1 重写错误="
        f"{diagnostics['raw_counterfactual_checks']['g_and_v_correct_rewrite_wrong']}。",
    ])
    lines.extend([
        "",
        "## Controller 选择",
        "",
        f"C2：{canonical_json(cap_counts['binary'])}；C3：{canonical_json(cap_counts['ternary'])}。",
        "",
        "## 调用与成本",
        "",
        "|阶段|调用数|输入 tokens|输出 tokens|reasoning tokens|累计延迟(s)|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for stage_name in ("G_initial", "V", "G_rewrite", "Controller"):
        usage = usage_summary[stage_name]
        lines.append(
            f"|{stage_name}|{usage['calls']}|{usage['input_tokens']}|{usage['output_tokens']}|"
            f"{usage['reasoning_tokens']}|{usage['latency_seconds']:.1f}|"
        )
    lines.extend([
        "",
        "## QA",
        "",
        f"状态：{qa['status']}；grader errors={grader_errors}；Controller fallbacks={controller_fallbacks}。",
        f"Token-logprob trace failures：G={trace_failures['G']}，V={trace_failures['V']}。",
        f"Feature missingness：{canonical_json(feature_missingness)}。",
        "",
        "## 限制",
        "",
        limitation_note,
    ])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def artifact_metadata(path: Path, display_path: str) -> dict[str, Any]:
    rows: int | None = None
    if path.suffix == ".jsonl":
        rows = len(read_jsonl(path))
    return {
        "path": display_path,
        "sha256": sha256_file(path),
        "rows": rows,
        "bytes": path.stat().st_size,
    }


def stage_finalize(args: argparse.Namespace) -> None:
    """Write the minimal immutable analysis-run manifest after QA."""

    bundle = Path(args.bundle_dir).resolve()
    qa_path = bundle / "reports" / "qa_report.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    validated = qa.get("status") == "passed"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve the fixed Git commit") from exc
    if len(git_commit) != 40:
        raise RuntimeError("git rev-parse did not return a 40-character commit")

    condition_copy = bundle / "condition.resolved.yaml"
    input_artifacts = [
        artifact_metadata(condition_copy, "condition.resolved.yaml"),
        artifact_metadata(Path(args.dataset_manifest), str(Path(args.dataset_manifest))),
    ]
    output_artifacts: list[dict[str, Any]] = []
    for directory in (bundle / "records", bundle / "reports"):
        for path in sorted(item for item in directory.iterdir() if item.is_file()):
            output_artifacts.append(artifact_metadata(path, path.relative_to(bundle).as_posix()))

    manifest = {
        "schema_version": "run-manifest-v1",
        "run_id": args.run_id,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "condition_id": CONDITION_ID,
        "condition_hash": condition_hash(),
        "replicate_plan_id": REPLICATE_PLAN_ID,
        "replicate_ids": list(RUN_REPLICATE_IDS),
        "stage": STAGE,
        "run_kind": "analysis",
        "source_run_ids": args.source_run_id,
        "git_commit": git_commit,
        "status": "validated" if validated else "failed",
        "environment": {
            "python": platform.python_version(),
            "math_verify": package_version("math-verify"),
            "pyyaml": package_version("PyYAML"),
            "vllm_backend": model_config("g")["backend_version"],
        },
        "hardware": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "0")),
        },
        "started_at_utc": args.started_at_utc,
        "ended_at_utc": utc_now(),
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "failure": None,
    }
    (bundle / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not validated:
        raise RuntimeError(f"{STAGE} QA failed; bundle remains in scratch and was not validated")


def add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--role", choices=("g", "v", "c"), required=True)
    smoke.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    smoke.add_argument("--served-model", required=True)
    smoke.set_defaults(function=stage_smoke)

    generate = subparsers.add_parser("generate")
    add_api_arguments(generate)
    generate.add_argument("--manifest", required=True)
    generate.set_defaults(function=stage_generate)

    verify_parser = subparsers.add_parser("verify")
    add_api_arguments(verify_parser)
    verify_parser.add_argument("--g-records", required=True)
    verify_parser.set_defaults(function=stage_verify)

    evidence = subparsers.add_parser("evidence")
    evidence.add_argument("--run-id", required=True)
    evidence.add_argument("--g-records", required=True)
    evidence.add_argument("--v-records", required=True)
    evidence.add_argument("--output", required=True)
    evidence.set_defaults(function=stage_evidence)

    rewrite = subparsers.add_parser("rewrite")
    add_api_arguments(rewrite)
    rewrite.add_argument("--g-records", required=True)
    rewrite.add_argument("--v-records", required=True)
    rewrite.set_defaults(function=stage_rewrite)

    controller = subparsers.add_parser("controller")
    add_api_arguments(controller)
    controller.add_argument("--evidence-records", required=True)
    controller.add_argument("--v-records", required=True)
    controller.set_defaults(function=stage_controller)

    grade = subparsers.add_parser("grade")
    grade.add_argument("--run-id", required=True)
    grade.add_argument("--manifest", required=True)
    grade.add_argument("--g-records", required=True)
    grade.add_argument("--v-records", required=True)
    grade.add_argument("--rewrite-records", required=True)
    grade.add_argument("--evidence-records", required=True)
    grade.add_argument("--output", required=True)
    grade.set_defaults(function=stage_grade)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--counterfactuals", required=True)
    analyze.add_argument("--controller-records", required=True)
    analyze.add_argument("--g-records")
    analyze.add_argument("--v-records")
    analyze.add_argument("--rewrite-records")
    analyze.add_argument("--evidence-records")
    analyze.add_argument("--output-dir", required=True)
    analyze.set_defaults(function=stage_analyze)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--bundle-dir", required=True)
    finalize.add_argument("--dataset-manifest", required=True)
    finalize.add_argument("--started-at-utc", required=True)
    finalize.add_argument("--source-run-id", action="append", required=True)
    finalize.set_defaults(function=stage_finalize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if "REQUIRED_BEFORE_RUN" in CONDITION_PATH.read_text(encoding="utf-8"):
        raise RuntimeError(f"{CONDITION_ID} still contains REQUIRED_BEFORE_RUN")
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
