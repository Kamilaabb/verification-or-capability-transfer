#!/usr/bin/env python3
"""c018: neutral Qwen3-8B verifier and localization-aware scalar capture."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c002_dev30 as base  # noqa: E402
import run_c014_router as capture_shared  # noqa: E402

from authority_allocation import derive_call_seed  # noqa: E402
from authority_allocation.api import api_call  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.prompts import render, split_prompt  # noqa: E402
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json,
    sha256_bytes,
    sha256_file,
    stable_id,
    utc_now,
)
from authority_allocation.verifier_scalar_features import (  # noqa: E402
    FEATURE_NAMES,
    derive_verifier_scalar_evidence,
    parse_verifier_report,
)


CONDITION_ID = "c018"
CONDITION_PATH = ROOT / "conditions/c018/condition.yaml"
DATASET_ID = "math500-r6e4ed1-l3to5-dev300-s20260910"
REPLICATE_PLAN_ID = "r010"
REPLICATE_ID = "v01"
SEED_CELL_ID = "c018-v"
STAGE = "pilot300"
EXPECTED_ROWS = 300
IDENTITY_TOLERANCE = 1e-4
RAW_TOP1_TOLERANCE = 1e-6


def condition() -> dict[str, Any]:
    return resolve_condition(CONDITION_PATH, repository_root=ROOT)


def condition_hash() -> str:
    resolved = condition()
    excluded = set(resolved["hash_policy"]["exclude"])

    def filtered(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: filtered(item) for key, item in value.items()
                    if key not in excluded}
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
            try:
                is_file = candidate.is_file()
            except OSError:
                is_file = False
            if is_file:
                referenced[value] = sha256_file(candidate)

    behavioral = filtered(resolved)
    find_paths(behavioral)
    return sha256_bytes(canonical_json({
        "condition": behavioral, "referenced_files": referenced,
    }).encode("utf-8"))


def verifier() -> dict[str, Any]:
    return condition()["verifier"]


def require_runnable() -> dict[str, Any]:
    config = condition()
    model = config["verifier"]
    missing = [name for name in ("revision", "server_path", "chat_template_hash")
               if not model.get(name)]
    if missing:
        raise SystemExit(f"c018 unresolved verifier identity: {missing}")
    if model.get("thinking") is not False:
        raise SystemExit("c018 requires verifier thinking=false")
    if float(model["decoding"]["temperature"]) != 0.0:
        raise SystemExit("c018 requires verifier temperature=0")
    if model["decoding"].get("constrained_schema") is not None:
        raise SystemExit("c018 forbids constrained decoding so raw greedy logits remain auditable")
    if model["decoding"].get("format_repair") != "disabled":
        raise SystemExit("c018 forbids format repair")
    if config["features"]["count"] != len(FEATURE_NAMES):
        raise SystemExit("c018 feature count does not match the registered implementation")
    return model


def verifier_identity() -> dict[str, Any]:
    model = require_runnable()
    prompt = ROOT / model["prompt"]
    return {
        "model_id": model["model_id"],
        "revision": model["revision"],
        "server_path": model["server_path"],
        "backend": model["backend"],
        "prompt_sha256": sha256_file(prompt),
        "chat_template_sha256": model["chat_template_hash"],
        "dtype": model["dtype"],
        "quantization": model["quantization"],
    }


def v_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    """The usage block of a V record, including the reasoning_tokens its schema requires."""

    return {**base.compact_usage(usage), "reasoning_tokens": usage.get("reasoning_tokens")}


def verifier_messages(problem: str, y0: str) -> list[dict[str, str]]:
    system, user = split_prompt(ROOT / verifier()["prompt"])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render(user, problem=problem, y0=y0)},
    ]


def build_native_record(
    *, run_id: str, v_record: Mapping[str, Any], capture: Mapping[str, Any]
) -> dict[str, Any]:
    trace = v_record.get("token_logprobs") or {}
    tokens = trace.get("tokens") or []
    raw_positions = capture.get("positions") or []
    capture_ids = capture.get("output_token_ids") or []
    source_ids = [token.get("token_id") for token in tokens]
    exact_tokens = source_ids == capture_ids
    status = "ok"
    positions: list[dict[str, Any]] = []
    identity_error: float | None = None
    if trace.get("status") != "ok" or not tokens:
        status = "source_trace_missing"
    elif not exact_tokens:
        status = "token_mismatch"
    elif len(raw_positions) != len(tokens):
        status = "raw_trace_missing"
    else:
        identity_error = 0.0
        for token, raw in zip(tokens, raw_positions, strict=True):
            top_ids = raw.get("top20_token_ids") or []
            top_logits = raw.get("top20_raw_logits") or []
            sampled_raw = raw.get("sampled_raw_logit")
            scalar_values = [raw.get(name) for name in (
                "max_raw_logit", "logsumexp", "entropy", "top1_top2_margin"
            )]
            if (raw.get("sampled_token_id") != token.get("token_id")
                    or not isinstance(sampled_raw, (int, float))
                    or len(top_ids) != 20 or len(top_logits) != 20
                    or len(set(top_ids)) != 20
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in [sampled_raw, *scalar_values, *top_logits])):
                status = "raw_trace_missing"
                positions = []
                identity_error = None
                break
            sampled_logprob = float(token["logprob"])
            identity_error = max(identity_error, abs(sampled_logprob - (
                float(sampled_raw) - float(raw["logsumexp"]))))
            positions.append({
                "index": int(raw["index"]),
                "token_id": int(token["token_id"]),
                "sampled_raw_logit": float(sampled_raw),
                "sampled_logprob": sampled_logprob,
                "max_raw_logit": float(raw["max_raw_logit"]),
                "logsumexp": float(raw["logsumexp"]),
                "entropy": float(raw["entropy"]),
                "top1_top2_margin": float(raw["top1_top2_margin"]),
                "top20_token_ids": [int(value) for value in top_ids],
                "top20_raw_logits": [float(value) for value in top_logits],
            })
        if status == "ok" and identity_error is not None:
            if identity_error > IDENTITY_TOLERANCE:
                status = "identity_mismatch"
            elif any(abs(row["sampled_raw_logit"] - row["max_raw_logit"])
                     > RAW_TOP1_TOLERANCE for row in positions):
                status = "non_greedy"

    return {
        "schema_version": "math-verifier-native-logit-trace-record-v1",
        "record_id": stable_id(run_id, v_record["problem_id"], "v-native-logits"),
        "run_id": run_id,
        "source_v_run_id": v_record["run_id"],
        "condition_id": CONDITION_ID,
        "condition_hash": condition_hash(),
        "problem_id": v_record["problem_id"],
        "capture_id": str(capture.get("capture_id", "")),
        "capture_backend": "vllm_same_forward_logits_processor_v1",
        "status": status,
        "same_forward": True,
        "exact_token_match": exact_tokens,
        "identity_max_abs_error": identity_error,
        "positions": positions,
        "created_at_utc": utc_now(),
    }


def _call_verifier(
    *, args: argparse.Namespace, problem_id: str, problem: str, y0: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = require_runnable()
    capture_id = stable_id(args.run_id, problem_id, "v-same-forward-logits")
    decoding = dict(model["decoding"])
    decoding["vllm_xargs"] = {"c014_capture_id": capture_id}
    content, reasoning, usage = api_call(
        api_base=args.api_base,
        served_model=args.served_model,
        messages=verifier_messages(problem, y0),
        decoding=decoding,
        seed=seed,
        thinking=False,
    )
    parsed = parse_verifier_report(content)
    report, field_spans = (None, {}) if parsed is None else parsed
    v_record = {
        "schema_version": "math-verifier-record-v1",
        "record_id": stable_id(args.run_id, problem_id, "verify"),
        "run_id": args.run_id,
        "source_g_run_id": args.source_g_run_id,
        "condition_id": CONDITION_ID,
        "condition_hash": condition_hash(),
        "replicate_id": args.replicate_id,
        "stage": STAGE,
        "cell_id": "V",
        "problem_id": problem_id,
        "verifier": verifier_identity(),
        "call_seed": seed,
        "raw_output": content,
        "reasoning_content": reasoning,
        "token_logprobs": usage.pop("token_logprobs", None),
        "v_report": report,
        "report_parse_status": "ok" if report is not None else "fallback",
        "registered_action": "KEEP" if report is None else report["action"],
        "field_spans": field_spans,
        "usage": v_usage(usage),
        "created_at_utc": utc_now(),
    }
    capture = capture_shared.wait_for_same_forward_capture(
        Path(args.capture_sidecar), capture_id, timeout_seconds=30
    )
    native = build_native_record(
        run_id=args.logit_run_id, v_record=v_record, capture=capture
    )
    return v_record, native


def stage_smoke(args: argparse.Namespace) -> None:
    args.source_g_run_id = "c018-smoke-source"
    args.replicate_id = REPLICATE_ID
    args.logit_run_id = args.run_id + "__logits"
    v_record, native = _call_verifier(
        args=args,
        problem_id="smoke-1",
        problem="Determine whether the proposed answer to 2+2 is correct.",
        y0="We compute $2+2=4$, so the final answer is \\boxed{4}.",
        seed=1,
    )
    if v_record["reasoning_content"] not in (None, ""):
        raise RuntimeError("thinking-off smoke unexpectedly returned separate reasoning")
    if v_record["report_parse_status"] != "ok":
        raise RuntimeError(
            "smoke report did not parse in the registered tagged format; refusing to "
            "spend a GPU window on a run whose reports would all fall back to KEEP"
        )
    if native["status"] != "ok":
        raise RuntimeError(f"same-forward smoke failed: {native['status']}")
    print(canonical_json({
        "stage": "smoke",
        "status": "passed",
        "report_parse_status": v_record["report_parse_status"],
        "tokens": len(native["positions"]),
        "identity_max_abs_error": native["identity_max_abs_error"],
    }))


def stage_verify(args: argparse.Namespace) -> None:
    g_rows = list(read_jsonl(Path(args.g_records)))
    if len(g_rows) != int(args.expected_rows):
        raise RuntimeError(f"expected {args.expected_rows} source G rows, got {len(g_rows)}")
    output = Path(args.output)
    native_output = Path(args.logit_output)
    done = completed_keys(output, ("problem_id",))
    native_done = completed_keys(native_output, ("problem_id",))
    if done != native_done:
        raise RuntimeError("V and native-logit resume sets differ")
    written = 0
    for g_row in g_rows:
        problem_id = g_row["problem_id"]
        if (problem_id,) in done:
            continue
        y0 = g_row["outputs"][0]["raw_output"] or ""
        seed = derive_call_seed(
            REPLICATE_PLAN_ID, "verify", args.replicate_id,
            int(args.base_seed), SEED_CELL_ID, problem_id, 0,
        )
        v_record, native = _call_verifier(
            args=args,
            problem_id=problem_id,
            problem=g_row["problem_text"],
            y0=y0,
            seed=seed,
        )
        if native["status"] != "ok":
            raise RuntimeError(f"native trace failed for {problem_id}: {native['status']}")
        append_jsonl(output, v_record)
        append_jsonl(native_output, native)
        written += 1
    print(canonical_json({"stage": "verify", "written": written, "total": len(g_rows)}))


def stage_evidence(args: argparse.Namespace) -> None:
    g_rows = {row["problem_id"]: row for row in read_jsonl(Path(args.g_records))}
    g_native = {row["problem_id"]: row for row in read_jsonl(Path(args.g_logit_traces))}
    v_rows = {row["problem_id"]: row for row in read_jsonl(Path(args.v_records))}
    v_native = {row["problem_id"]: row for row in read_jsonl(Path(args.v_logit_traces))}
    expected = int(args.expected_rows)
    if not all(len(rows) == expected for rows in (g_rows, g_native, v_rows, v_native)):
        raise RuntimeError("c018 evidence inputs are incomplete")
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for problem_id, g_row in g_rows.items():
        if (problem_id,) in done:
            continue
        v_row = v_rows[problem_id]
        report = v_row["v_report"]
        g_sample = g_row["outputs"][0]
        g_text = g_sample["raw_output"] or ""
        g_answer = g_sample.get("extracted_answer")
        v_answer = None
        equivalent = None
        if report is not None:
            v_answer = base.math_parse(report["final_answer"])[1]
            if g_answer is not None and v_answer is not None:
                # Compare the parsed texts; G's extracted_answer is a sympy string that
                # does not re-parse (docs/analysis/c018_report_format.md).
                equivalent = base.math_equivalent(
                    base.answer_text(g_sample), report["final_answer"])[0]
        audit = derive_verifier_scalar_evidence(
            g_text=g_text,
            g_trace=g_sample.get("token_logprobs") or {},
            g_native=g_native[problem_id],
            g_answer=g_answer,
            v_text=v_row["raw_output"],
            v_trace=v_row.get("token_logprobs") or {},
            v_native=v_native[problem_id],
            report=report,
            field_spans=v_row["field_spans"],
            v_answer=v_answer,
            answers_equivalent=equivalent,
        )
        append_jsonl(output, {
            "schema_version": "math-verifier-scalar-evidence-record-v1",
            "record_id": stable_id(args.run_id, problem_id, "v-scalar-evidence"),
            "run_id": args.run_id,
            "source_g_run_id": g_row["run_id"],
            "source_g_logit_run_id": g_native[problem_id]["run_id"],
            "source_v_run_id": v_row["run_id"],
            "source_v_logit_run_id": v_native[problem_id]["run_id"],
            "condition_id": CONDITION_ID,
            "condition_hash": condition_hash(),
            "problem_id": problem_id,
            "parser_id": "math_expression_spans_v1",
            **audit,
            "created_at_utc": utc_now(),
        })
        written += 1
    print(canonical_json({"stage": "evidence", "written": written}))


def _summary(values: list[float | int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def stage_analyze(args: argparse.Namespace) -> None:
    v_rows = list(read_jsonl(Path(args.v_records)))
    evidence = list(read_jsonl(Path(args.evidence)))
    expected = int(args.expected_rows)
    if len(v_rows) != expected or len(evidence) != expected:
        raise RuntimeError(f"incomplete analysis inputs: {len(v_rows)}, {len(evidence)}")
    parsed = [row for row in v_rows if row["report_parse_status"] == "ok"]
    evidence_by_problem = {row["problem_id"]: row for row in evidence}
    verdict_by_location: dict[str, Counter[str]] = {}
    for row in parsed:
        status = evidence_by_problem[row["problem_id"]]["location_match_status"]
        verdict_by_location.setdefault(row["v_report"]["verdict"], Counter())[status] += 1
    feature_summaries: dict[str, Any] = {}
    missingness: dict[str, int] = {}
    for name in FEATURE_NAMES:
        values = [row["features"][name] for row in evidence
                  if row["features"][name] is not None]
        feature_summaries[name] = None if not values else _summary(values)
        missingness[name] = expected - len(values)
    report = {
        "condition_id": CONDITION_ID,
        "run_id": args.run_id,
        "rows": expected,
        "report_parse_status_distribution": dict(Counter(
            row["report_parse_status"] for row in v_rows
        )),
        "verdict_distribution": dict(Counter(
            row["v_report"]["verdict"] for row in parsed
        )),
        "action_distribution": dict(Counter(
            row["registered_action"] for row in v_rows
        )),
        "location_match_status_distribution": dict(Counter(
            row["location_match_status"] for row in evidence
        )),
        "verdict_by_location_status": {
            verdict: dict(counts) for verdict, counts in sorted(verdict_by_location.items())
        },
        "candidate_answer_parseable": sum(
            row["features"]["v_candidate_answer_parseable"] == 1 for row in evidence
        ),
        "gv_final_answer_equivalent": sum(
            row["features"]["gv_final_answer_equivalent"] is True for row in evidence
        ),
        "rows_with_v_math_tokens": sum(bool(row["v_math_token_indices"])
                                       for row in evidence),
        "v_math_span_count": sum(len(row["v_math_spans"]) for row in evidence),
        "v_math_token_count": sum(len(row["v_math_token_indices"]) for row in evidence),
        "api_output_tokens": sum(int(row["usage"].get("output_tokens") or 0)
                                 for row in v_rows),
        "scalar_missingness": missingness,
        "scalar_summaries": feature_summaries,
        "development_only": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = "\n".join([
        f"# {CONDITION_ID} neutral verifier scalar summary", "",
        f"- Rows / parsed reports: {expected} / {len(parsed)}",
        f"- Verdicts: {json.dumps(report['verdict_distribution'], sort_keys=True)}",
        f"- Actions: {json.dumps(report['action_distribution'], sort_keys=True)}",
        f"- Parseable V final answers: {report['candidate_answer_parseable']}/{expected}",
        f"- G/V mathematically equivalent final answers: {report['gv_final_answer_equivalent']}/{expected}",
        f"- Error-location mapping: {json.dumps(report['location_match_status_distribution'], sort_keys=True)}",
        f"- Verdict x location: {json.dumps(report['verdict_by_location_status'], sort_keys=True)}",
        f"- Rows / spans / tokens with V math: {report['rows_with_v_math_tokens']} / "
        f"{report['v_math_span_count']} / {report['v_math_token_count']}",
        f"- V API output tokens: {report['api_output_tokens']}",
        f"- Scalar missingness: {json.dumps(missingness, sort_keys=True)}", "",
        "The prompt is correctness-neutral. Every quoted location is classified under any "
        "verdict; token-level location features come only from a unique exact quotation. "
        "Raw max-logit features are pre-softmax vocabulary maxima, not logarithms.", "",
    ])
    (output.parent / "summary.md").write_text(summary, encoding="utf-8")
    print(canonical_json(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-runnable")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--run-id", required=True)
    smoke.add_argument("--api-base", required=True)
    smoke.add_argument("--served-model", required=True)
    smoke.add_argument("--capture-sidecar", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--logit-run-id", required=True)
    verify.add_argument("--source-g-run-id", required=True)
    verify.add_argument("--api-base", required=True)
    verify.add_argument("--served-model", required=True)
    verify.add_argument("--g-records", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--capture-sidecar", required=True)
    verify.add_argument("--logit-output", required=True)
    verify.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    verify.add_argument("--replicate-id", default=REPLICATE_ID)
    verify.add_argument("--base-seed", type=int, default=20260923)
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--run-id", required=True)
    evidence.add_argument("--g-records", required=True)
    evidence.add_argument("--g-logit-traces", required=True)
    evidence.add_argument("--v-records", required=True)
    evidence.add_argument("--v-logit-traces", required=True)
    evidence.add_argument("--output", required=True)
    evidence.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--v-records", required=True)
    analyze.add_argument("--evidence", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "check-runnable":
        model = require_runnable()
        print(canonical_json({
            "runnable": True,
            "condition_id": CONDITION_ID,
            "revision": model["revision"],
            "features": len(FEATURE_NAMES),
        }))
    elif args.command == "smoke":
        stage_smoke(args)
    elif args.command == "verify":
        stage_verify(args)
    elif args.command == "evidence":
        stage_evidence(args)
    else:
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
