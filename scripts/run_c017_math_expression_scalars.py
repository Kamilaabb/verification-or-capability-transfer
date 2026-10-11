#!/usr/bin/env python3
"""c017: rerun c014 pass 1 and derive all-expression logit scalars."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c014_router as shared  # noqa: E402
import run_c002_dev30 as base  # noqa: E402

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.math_expression_spans import (  # noqa: E402
    content_token_indices,
    overlapping_token_indices,
    parse_math_spans,
)
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json, sha256_bytes, sha256_file, stable_id, utc_now,
)

CONDITION_ID = "c017"
CONDITION_PATH = ROOT / "conditions/c017/condition.yaml"
DATASET_ID = "math500-r6e4ed1-l3to5-dev300-s20260910"
REPLICATE_PLAN_ID = "r009"
REPLICATE_ID = "p01"
STAGE = "math-expression-scalars"
SEED_CELL_ID = "c017"
EXPECTED_ROWS = 300


def configure_shared() -> None:
    shared.CONDITION_ID = CONDITION_ID
    shared.CONDITION_PATH = CONDITION_PATH
    shared.DATASET_ID = DATASET_ID
    shared.REPLICATE_PLAN_ID = REPLICATE_PLAN_ID
    shared.STAGE = STAGE
    shared.SEED_CELL_ID = SEED_CELL_ID
    shared.EXPECTED_ROWS = EXPECTED_ROWS
    shared.configure()


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

    behavioral = filtered(resolved)
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

    find_paths(behavioral)
    return sha256_bytes(canonical_json({"condition": behavioral,
                                        "referenced_files": referenced}).encode("utf-8"))


def require_runnable() -> dict[str, Any]:
    model = condition()["model"]
    missing = [name for name in ("revision", "tokenizer_revision", "server_path",
                                  "chat_template_hash") if not model.get(name)]
    if missing:
        raise SystemExit(f"{CONDITION_ID} unresolved model identity: {missing}")
    if model.get("thinking") is not False:
        raise SystemExit(f"{CONDITION_ID} requires thinking=false")
    if int(model["pass1_decoding"]["max_new_tokens"]) != 3072:
        raise SystemExit(f"{CONDITION_ID} requires c014's 3072-token pass1 budget")
    if condition()["features"]["count"] != 7:
        raise SystemExit(f"{CONDITION_ID} must register exactly seven requested scalars")
    return model


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def derive_evidence(
    pass1: Mapping[str, Any], native: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    output = pass1["outputs"][0]
    text = output["raw_output"]
    trace = output.get("token_logprobs") or {}
    spans = parse_math_spans(text)
    math_indices = overlapping_token_indices(trace, text, spans)
    output_indices = content_token_indices(trace)
    source_by_index = {int(token["index"]): token for token in trace.get("tokens", [])}
    native_by_index = {int(position["index"]): position
                       for position in native.get("positions", [])}

    def logprobs(indices: list[int]) -> list[float]:
        return [float(source_by_index[index]["logprob"])
                for index in indices if index in source_by_index]

    def max_logits(indices: list[int]) -> list[float]:
        return [float(native_by_index[index]["max_raw_logit"])
                for index in indices if index in native_by_index]

    math_lp, math_max = logprobs(math_indices), max_logits(math_indices)
    output_lp, output_max = logprobs(output_indices), max_logits(output_indices)
    if len(math_lp) != len(math_indices) or len(math_max) != len(math_indices):
        math_lp, math_max = [], []
    if len(output_lp) != len(output_indices) or len(output_max) != len(output_indices):
        output_lp, output_max = [], []

    features: dict[str, int | float | None] = {
        "g_math_expression_mean_max_logit": _mean(math_max),
        "g_math_expression_mean_logprob": _mean(math_lp),
        "g_answer_parseable": int(output.get("extracted_answer") is not None),
        "problem_char_count": len(pass1["problem_text"]),
        "g_output_char_count": len(text),
        "g_output_mean_max_logit": _mean(output_max),
        "g_output_mean_logprob": _mean(output_lp),
    }
    missingness = {name: value is None for name, value in features.items()}
    return {
        "schema_version": "math-expression-scalar-evidence-record-v1",
        "record_id": stable_id(run_id, pass1["problem_id"], "math-expression-evidence"),
        "run_id": run_id,
        "source_pass1_run_id": pass1["run_id"],
        "source_logit_run_id": native["run_id"],
        "condition_id": CONDITION_ID,
        "condition_hash": condition_hash(),
        "problem_id": pass1["problem_id"],
        "parser_id": "math_expression_spans_v1",
        "features": features,
        "missingness": missingness,
        "math_spans": [{"start": span.start, "end": span.end, "kind": span.kind}
                       for span in spans],
        "math_token_indices": math_indices,
        "math_token_count": len(math_indices),
        "output_token_count": len(output_indices),
        "created_at_utc": utc_now(),
    }


def stage_evidence(args: argparse.Namespace) -> None:
    natives = {row["problem_id"]: row for row in read_jsonl(Path(args.logit_traces))}
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for row in read_jsonl(Path(args.pass1_records)):
        if (row["problem_id"],) in done:
            continue
        native = natives.get(row["problem_id"])
        if native is None or native.get("status") != "ok":
            raise RuntimeError(f"missing valid native trace for {row['problem_id']}")
        append_jsonl(output, derive_evidence(row, native, run_id=args.run_id))
        written += 1
    print(canonical_json({"stage": "evidence", "written": written}))


def stage_grade(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(Path(args.pass1_records)))
    ids = [row["problem_id"] for row in rows]
    sources = base.load_source_rows(ids)
    base.audit_gold_answers(sources)
    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "outcome"))
    for row in rows:
        problem_id = row["problem_id"]
        if (problem_id, "u0") in done:
            continue
        # Grade the parsed text, not extracted_answer, whose sympy string does not
        # re-parse (docs/analysis/c018_report_format.md).
        prediction = base.answer_text(row["outputs"][0])
        value, status = base.grade_prediction(sources[problem_id]["gold"], prediction)
        append_jsonl(output, {"problem_id": problem_id, "outcome": "u0", "u": value,
                              "grader_status": status, "source_record_id": row["record_id"]})
    print(canonical_json({"stage": "grade", "rows": len(rows)}))


def _summary(values: list[float | int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": ordered[0], "max": ordered[-1]}


def stage_analyze(args: argparse.Namespace) -> None:
    pass1 = list(read_jsonl(Path(args.pass1_records)))
    evidence = list(read_jsonl(Path(args.evidence)))
    grades = {row["problem_id"]: row for row in read_jsonl(Path(args.grades))}
    expected = int(args.expected_rows)
    if not (len(pass1) == len(evidence) == len(grades) == expected):
        raise RuntimeError(f"incomplete inputs: {len(pass1)}, {len(evidence)}, {len(grades)}")
    if any(row["u"] is None for row in grades.values()):
        raise RuntimeError("grader_error present; accuracy undefined")

    feature_names = list(evidence[0]["features"])
    feature_summaries = {}
    missingness = {}
    for name in feature_names:
        values = [row["features"][name] for row in evidence
                  if row["features"][name] is not None]
        feature_summaries[name] = None if not values else _summary(values)
        missingness[name] = expected - len(values)
    correct = sum(int(row["u"]) for row in grades.values())
    output_tokens = [int(row["outputs"][0]["usage"].get("output_tokens") or 0)
                     for row in pass1]
    report = {
        "condition_id": CONDITION_ID, "run_id": args.run_id, "rows": expected,
        "correct": correct, "accuracy": correct / expected,
        "answer_parseable": sum(row["features"]["g_answer_parseable"] for row in evidence),
        "rows_with_math_tokens": sum(row["math_token_count"] > 0 for row in evidence),
        "math_span_count": sum(len(row["math_spans"]) for row in evidence),
        "math_token_count": sum(row["math_token_count"] for row in evidence),
        "output_content_token_count": sum(row["output_token_count"] for row in evidence),
        "api_output_tokens": {"sum": sum(output_tokens), "mean": statistics.mean(output_tokens)},
        "scalar_missingness": missingness,
        "scalar_summaries": feature_summaries,
        "parse_status_distribution": dict(Counter(
            row["outputs"][0]["parse_status"] for row in pass1)),
        "grader_status_distribution": dict(Counter(
            row["grader_status"] for row in grades.values())),
        "parser_id": "math_expression_spans_v1",
        "development_only": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = "\n".join([
        f"# {CONDITION_ID} math-expression scalar summary", "",
        f"- Rows / correct / accuracy: {expected} / {correct} / {correct / expected:.4f}",
        f"- Parseable answers: {report['answer_parseable']}/{expected}",
        f"- Rows with parsed math tokens: {report['rows_with_math_tokens']}/{expected}",
        f"- Parsed math spans / tokens: {report['math_span_count']} / {report['math_token_count']}",
        f"- Output content tokens: {report['output_content_token_count']}",
        f"- Scalar missingness: {json.dumps(missingness, sort_keys=True)}", "",
        "All scalar evidence was written before grading. Math-token aggregates use every "
        "token overlapping any span from math_expression_spans_v1; they are not restricted "
        "to the final extracted answer.", "",
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
    generate = sub.add_parser("generate")
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--api-base", required=True)
    generate.add_argument("--served-model", required=True)
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    generate.add_argument("--output", required=True)
    generate.add_argument("--capture-sidecar", required=True)
    generate.add_argument("--logit-output", required=True)
    generate.add_argument("--logit-run-id", required=True)
    generate.add_argument("--replicate-id", default=REPLICATE_ID)
    generate.add_argument("--base-seed", type=int, default=20260922)
    ev = sub.add_parser("evidence")
    ev.add_argument("--run-id", required=True)
    ev.add_argument("--pass1-records", required=True)
    ev.add_argument("--logit-traces", required=True)
    ev.add_argument("--output", required=True)
    grade = sub.add_parser("grade")
    grade.add_argument("--pass1-records", required=True)
    grade.add_argument("--output", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--pass1-records", required=True)
    analyze.add_argument("--evidence", required=True)
    analyze.add_argument("--grades", required=True)
    analyze.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    analyze.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "check-runnable":
        model = require_runnable()
        print(canonical_json({"runnable": True, "condition_id": CONDITION_ID,
                              "revision": model["revision"], "features": 7}))
    elif args.command == "smoke":
        configure_shared()
        shared.stage_smoke(args)
    elif args.command == "generate":
        configure_shared()
        shared.stage_pass1(args)
    elif args.command == "evidence":
        stage_evidence(args)
    elif args.command == "grade":
        stage_grade(args)
    else:
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
