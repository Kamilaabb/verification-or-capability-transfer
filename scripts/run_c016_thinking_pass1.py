#!/usr/bin/env python3
"""c016: Qwen3-4B thinking-on, single pass, 4096-token math dev300 diagnostic."""

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

import run_c002_dev30 as base  # noqa: E402

from authority_allocation.api import api_call  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.prompts import render, split_prompt  # noqa: E402
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json, sha256_bytes, sha256_file, stable_id, utc_now,
)

CONDITION_ID = "c016"
CONDITION_PATH = ROOT / "conditions/c016/condition.yaml"
DATASET_ID = "math500-r6e4ed1-l3to5-dev300-s20260910"
REPLICATE_PLAN_ID = "r008"
REPLICATE_ID = "p01"
STAGE = "pilot300"
EXPECTED_ROWS = 300
CALL_SEED = 42


def condition() -> dict[str, Any]:
    return resolve_condition(CONDITION_PATH, repository_root=ROOT)


def condition_hash() -> str:
    resolved = condition()
    excluded = set(resolved["hash_policy"]["exclude"])

    def filtered(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: filtered(v) for k, v in value.items() if k not in excluded}
        if isinstance(value, list):
            return [filtered(v) for v in value]
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
                exists = candidate.is_file()
            except OSError:
                exists = False
            if exists:
                referenced[value] = sha256_file(candidate)

    find_paths(behavioral)
    return sha256_bytes(canonical_json({
        "condition": behavioral, "referenced_files": referenced,
    }).encode("utf-8"))


def model_block() -> dict[str, Any]:
    return condition()["model"]


def require_runnable() -> dict[str, Any]:
    cfg = condition()
    model = cfg["model"]
    missing = [k for k in ("revision", "tokenizer_revision", "server_path",
                            "chat_template_hash") if not model.get(k)]
    if missing:
        raise SystemExit(f"c016 unresolved model identity: {missing}")
    if model.get("thinking") is not True:
        raise SystemExit("c016 requires thinking=true")
    if model.get("reasoning_parser") != "qwen3":
        raise SystemExit("c016 requires the qwen3 reasoning parser")
    decoding = model["pass1_decoding"]
    if float(decoding["temperature"]) != 0.0 or int(decoding["max_new_tokens"]) != 4096:
        raise SystemExit("c016 requires temperature=0 and max_new_tokens=4096")
    if cfg["replicates"]["call_seed"] != CALL_SEED:
        raise SystemExit("c016 requires constant call seed 42")
    if model["finalizer"]["enabled"] or model["retry_generation"]:
        raise SystemExit("c016 forbids finalizer and generation retry")
    if cfg["grading"]["length_finish_policy"] != "grade_if_answer_extractable_otherwise_wrong":
        raise SystemExit("c016 length-finish grading policy drifted")
    return model


def compact_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "latency_ms": usage.get("latency_ms"),
        "finish_reason": usage.get("finish_reason"),
        "retries": usage.get("retries", 0),
    }


def safe_extract(content: str) -> str | None:
    if not content:
        return None
    try:
        return base.math_parse(content)[1]
    except Exception:
        return None


def manifest_ids(path: Path, expected: int) -> list[str]:
    ids = [base.manifest_fields(row)[0] for row in read_jsonl(path)]
    if len(ids) != expected:
        raise ValueError(f"expected {expected} manifest rows, got {len(ids)}")
    return ids


def model_identity(prompt_path: Path) -> dict[str, Any]:
    model = require_runnable()
    return {
        "model_id": model["model_id"], "revision": model["revision"],
        "server_path": model["server_path"], "backend": model["backend"],
        "backend_version": model["backend_version"], "dtype": model["dtype"],
        "quantization": model["quantization"],
        "prompt_sha256": sha256_file(prompt_path),
        "chat_template_sha256": model["chat_template_hash"],
        "thinking": True, "reasoning_parser": "qwen3",
    }


def stage_generate(args: argparse.Namespace) -> None:
    model = require_runnable()
    ids = manifest_ids(Path(args.manifest), int(args.expected_rows))
    sources = base.load_source_rows(ids)
    base.audit_gold_answers(sources)
    prompt = ROOT / model["prompt_pass1"]
    system, user = split_prompt(prompt)
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for problem_id in ids:
        if (problem_id,) in done:
            continue
        content, reasoning, usage = "", None, {
            "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
            "latency_ms": None, "finish_reason": None, "retries": 0,
        }
        status = "ok"
        try:
            content, reasoning, usage = api_call(
                api_base=args.api_base, served_model=args.served_model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": render(
                              user, problem=sources[problem_id]["problem"])}],
                decoding=model["pass1_decoding"], seed=CALL_SEED, thinking=True,
            )
        except Exception as exc:  # noqa: BLE001
            status = "call_error"
            print(canonical_json({"problem_id": problem_id,
                                  "call_error": str(exc)[:200]}))
        answer = safe_extract(content)
        if status == "ok" and answer is None:
            status = "model_invalid"
        append_jsonl(output, {
            "schema_version": "thinking-pass1-record-v1",
            "record_id": stable_id(args.run_id, problem_id, "pass1"),
            "run_id": args.run_id, "condition_id": CONDITION_ID,
            "condition_hash": condition_hash(), "dataset_id": DATASET_ID,
            "replicate_id": REPLICATE_ID, "stage": STAGE,
            "problem_id": problem_id, "problem_text": sources[problem_id]["problem"],
            "generator": model_identity(prompt),
            "output": {"sample_index": 0, "call_seed": CALL_SEED,
                       "raw_reasoning": reasoning, "raw_output": content,
                       "extracted_answer": answer, "parse_status": status,
                       "usage": compact_usage(usage)},
            "created_at_utc": utc_now(),
        })
        written += 1
    print(canonical_json({"stage": "pass1", "written": written, "total": len(ids)}))


def grade_one(gold: str, record: Mapping[str, Any]) -> tuple[int | None, str]:
    """Length is diagnostic only: an extractable answer is graded normally.

    The raw output is graded, not extracted_answer, which is the sympy string of the
    parse and does not re-parse (docs/analysis/c018_report_format.md).
    """
    text = base.answer_text(record["output"])
    if text is None:
        return 0, "model_invalid"
    return base.grade_prediction(gold, text)


def stage_grade(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(Path(args.records)))
    ids = [row["problem_id"] for row in rows]
    sources = base.load_source_rows(ids)
    base.audit_gold_answers(sources)
    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "outcome"))
    for row in rows:
        problem_id = row["problem_id"]
        if (problem_id, "u0") in done:
            continue
        value, status = grade_one(sources[problem_id]["gold"], row)
        append_jsonl(output, {
            "problem_id": problem_id, "outcome": "u0", "u": value,
            "grader_status": status, "source_record_id": row["record_id"],
        })
    print(canonical_json({"stage": "grade", "rows": len(rows)}))


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def stage_analyze(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(Path(args.records)))
    grades = {r["problem_id"]: r for r in read_jsonl(Path(args.grades))
              if r["outcome"] == "u0"}
    if len(rows) != int(args.expected_rows) or len(grades) != len(rows):
        raise RuntimeError(f"incomplete analysis inputs: records={len(rows)} grades={len(grades)}")
    if any(g["u"] is None for g in grades.values()):
        raise RuntimeError("grader_error present; accuracy is not defined")
    outputs = [r["output"] for r in rows]
    length_rows = [r for r in rows if r["output"]["usage"]["finish_reason"] == "length"]
    length_extractable = [r for r in length_rows if r["output"]["extracted_answer"] is not None]
    tokens = [int(o["usage"]["output_tokens"] or 0) for o in outputs]
    correct = sum(int(grades[r["problem_id"]]["u"]) for r in rows)
    report = {
        "run_id": args.run_id, "condition_id": CONDITION_ID, "rows": len(rows),
        "correct": correct, "accuracy": correct / len(rows),
        "finish_reason_distribution": dict(Counter(
            str(o["usage"]["finish_reason"]) for o in outputs)),
        "parse_status_distribution": dict(Counter(o["parse_status"] for o in outputs)),
        "grader_status_distribution": dict(Counter(
            grades[r["problem_id"]]["grader_status"] for r in rows)),
        "length_finish_count": len(length_rows),
        "length_finish_extractable": len(length_extractable),
        "length_finish_correct": sum(
            int(grades[r["problem_id"]]["u"]) for r in length_extractable),
        "unextractable_count": sum(o["extracted_answer"] is None for o in outputs),
        "reasoning_present_count": sum(bool(o["raw_reasoning"]) for o in outputs),
        "output_tokens": {"sum": sum(tokens), "mean": statistics.mean(tokens),
                          "median": statistics.median(tokens),
                          "p90": percentile(tokens, 0.90), "max": max(tokens)},
        "grading_rule": "length finishes are graded normally when an answer is extractable",
        "finalizer_calls": 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = "\n".join([
        "# c016 thinking-on pass1 summary", "",
        f"- Rows / correct / accuracy: {len(rows)} / {correct} / {correct / len(rows):.4f}",
        f"- Length finishes: {len(length_rows)}; extractable: {len(length_extractable)}; "
        f"correct: {report['length_finish_correct']}",
        f"- Unextractable: {report['unextractable_count']}",
        f"- Reasoning preserved: {report['reasoning_present_count']}/{len(rows)}",
        f"- Output tokens: {sum(tokens)} total; median {statistics.median(tokens):g}; "
        f"p90 {percentile(tokens, 0.90)}; max {max(tokens)}", "",
        "Length finishes were not automatically marked wrong: an extractable final answer "
        "was graded normally. No finalizer was used.", "",
    ])
    (output.parent / "summary.md").write_text(summary, encoding="utf-8")
    print(canonical_json(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-runnable")
    gen = sub.add_parser("generate")
    gen.add_argument("--run-id", required=True)
    gen.add_argument("--api-base", required=True)
    gen.add_argument("--served-model", required=True)
    gen.add_argument("--manifest", required=True)
    gen.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    gen.add_argument("--output", required=True)
    grade = sub.add_parser("grade")
    grade.add_argument("--records", required=True)
    grade.add_argument("--output", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--records", required=True)
    analyze.add_argument("--grades", required=True)
    analyze.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    analyze.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "check-runnable":
        model = require_runnable()
        print(canonical_json({"runnable": True, "revision": model["revision"],
                              "thinking": True, "max_new_tokens": 4096,
                              "call_seed": CALL_SEED}))
        return 0
    {"generate": stage_generate, "grade": stage_grade,
     "analyze": stage_analyze}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
