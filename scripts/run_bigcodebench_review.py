#!/usr/bin/env python3
"""GV on BigCodeBench development (docs/plan/code_bigcodebench_study.md): one model reviews the
other's archived solo output, for the pair Qwen3-4B x Ministral-8B.
c110: Ministral-8B reviews c100 (Qwen3-4B); c111: Qwen3-4B reviews c105 (Ministral-8B).

Usage: run_bigcodebench_review.py --condition-id {c110,...,c114,c148,c149,c199,c278,...,c302} <command> [args]
c112: Qwen3-8B reviews c100 (Qwen3-4B); c113: Qwen3-4B reviews c101 (Qwen3-8B).
c199: Qwen3-8B reviews c102 (Llama-3.2-3B); c148: Llama-3.2-3B reviews c101 (Qwen3-8B).
c149: Qwen2.5-Coder-7B reviews c104 (Qwen2.5-7B); c278: Qwen2.5-7B reviews c108 (Qwen2.5-Coder-7B).
c279: Seed-Coder-8B reviews c101 (Qwen3-8B); c280: Qwen3-8B reviews c109 (Seed-Coder-8B).
c281-c300: the stratified code pairs (plan section 14); each odd/even id pair is one model pair, both directions.

Commands:
    check-runnable  the frozen review prompt, the reviewer's identity (its solo condition's,
                    itself checked by run_bigcodebench_solo), the filled generator source, and
                    that the condition differs from its base only where registered
    smoke           one review call on a toy task: exact token alignment and a paired capture
    review          one review call per row on the reviewed model's complete output; resumable
    reconcile       keep only rows present and valid in both the record and trace files
    evidence        code_solve_scalar_v1 of the reviewer's own output, before grading
    grade-requests  the (record_id, task_id, text) file for the evaluate container
    grade           u_review, u_gv (primary: the reviewed program when the reviewer's output has
                    none) and u_gv_strict; any grader_error stops here
    analyze         reports/analysis.json and reports/summary.md

The solo runner is imported, not changed, so solo runs still pending use exactly the code they
were registered with.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_bigcodebench_solo as solo  # noqa: E402
import run_c014_router as shared  # noqa: E402
import run_mathfull_solo as mathfull  # noqa: E402

from authority_allocation import bigcodebench_study, derive_call_seed  # noqa: E402
from authority_allocation.code_solve_features import FEATURE_CONTRACT, derive_features  # noqa: E402
from authority_allocation.prompts import split_prompt  # noqa: E402
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.serialization import canonical_json, sha256_bytes, stable_id, utc_now  # noqa: E402

REVIEWS = {"c110": {"base": "c105", "g": "c100"},
           "c111": {"base": "c100", "g": "c105"},
           # The Qwen3-4B x Qwen3-8B pair, as in the math and knowledge domains (plan section 10).
           "c112": {"base": "c101", "g": "c100"},
           "c113": {"base": "c100", "g": "c101"},
           # c113 with the registered empty-token-bytes repair (plan section 10): c113 stopped on
           # a token vLLM returned without bytes.
           "c114": {"base": "c100", "g": "c101", "repair": True},
           # The Qwen3-8B x Llama-3.2-3B pair (decided 2026-10-01), the large-gap pair also run in the
           # knowledge and NL-reasoning domains: c199 Qwen3-8B reviews c102, c148 Llama-3.2-3B reviews c101.
           "c199": {"base": "c101", "g": "c102"},
           "c148": {"base": "c102", "g": "c101"},
           # The two coder pairs pre-planned in plan section 2 (decided 2026-10-02, plan section 13).
           "c149": {"base": "c108", "g": "c104"},
           "c278": {"base": "c104", "g": "c108"},
           "c279": {"base": "c109", "g": "c101"},
           "c280": {"base": "c101", "g": "c109"},
           # The stratified code pairs (decided 2026-10-02, plan section 14), both directions of each.
           "c281": {"base": "c106", "g": "c107"},
           "c282": {"base": "c107", "g": "c106"},
           "c283": {"base": "c106", "g": "c109"},
           "c284": {"base": "c109", "g": "c106"},
           "c285": {"base": "c105", "g": "c108"},
           "c286": {"base": "c108", "g": "c105"},
           "c287": {"base": "c103", "g": "c104"},
           "c288": {"base": "c104", "g": "c103"},
           "c289": {"base": "c102", "g": "c104"},
           "c290": {"base": "c104", "g": "c102"},
           "c291": {"base": "c106", "g": "c101"},
           "c292": {"base": "c101", "g": "c106"},
           "c293": {"base": "c106", "g": "c105"},
           "c294": {"base": "c105", "g": "c106"},
           "c295": {"base": "c107", "g": "c104"},
           "c296": {"base": "c104", "g": "c107"},
           "c297": {"base": "c109", "g": "c102"},
           "c298": {"base": "c102", "g": "c109"},
           "c299": {"base": "c100", "g": "c104"},
           "c300": {"base": "c104", "g": "c100"},
           # c279 with the registered empty-token-bytes repair (plan section 13): c279 stopped on a
           # token vLLM returned without bytes, as c113 did.
           "c301": {"base": "c109", "g": "c101", "repair": True},
           # c297 with the registered empty-token-bytes repair (plan section 14): c297 stopped on a
           # token vLLM returned without bytes, as c279 did.
           "c302": {"base": "c109", "g": "c102", "repair": True}}
REVIEW_SECTIONS = {"condition_id", "status", "purpose", "gate", "runnable_gate", "model", "replicates",
                   "protocol", "generator_source", "delivery", "stages"}
REVIEW_MODEL_KEYS = {"prompt_pass1", "prompt_pass1_provenance", "prompt_inputs"}
REPAIR_KEY, REPAIR_RULE = "empty_token_bytes_repair", "vocabulary_bytes_by_token_id_v1"
REPAIRS_FILE = "token_byte_repairs.jsonl"
CONDITION_ID = "c110"
ROLE = "review"
STAGE = "bigcodebench-review"
SMOKE_Y0 = "```python\ndef task_func(a, b):\n    return a - b\n```"
FINAL_BLOCK = re.compile(r"```[ \t]*(?:python3?|py)[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def configure(condition_id: str) -> None:
    global CONDITION_ID
    if condition_id not in REVIEWS:
        raise SystemExit(f"unknown condition {condition_id}")
    CONDITION_ID = condition_id
    shared.CONDITION_ID = condition_id
    shared.CONDITION_PATH = ROOT / "conditions" / condition_id / "condition.yaml"
    shared.DATASET_ID = solo.DATASET_ID
    shared.REPLICATE_PLAN_ID = solo.REPLICATE_PLAN_ID
    shared.STAGE = STAGE
    shared.SEED_CELL_ID = condition_id
    shared.EXPECTED_ROWS = solo.EXPECTED_ROWS
    shared.configure()


def require_registered(condition_id: str) -> dict[str, Any]:
    spec = REVIEWS[condition_id]
    solo.require_registered(spec["base"])  # the reviewer's own identity, dataset and decoding
    cond, parent = solo.condition(condition_id), solo.condition(spec["base"])
    model = cond["model"]
    if model["prompt_pass1"] != bigcodebench_study.CODE_REVIEW_PROMPT:
        raise SystemExit(f"{condition_id} must use the frozen prompt {bigcodebench_study.CODE_REVIEW_PROMPT}")
    bigcodebench_study.require_frozen_prompt(ROOT, model["prompt_pass1"])
    for key in sorted(set(cond) | set(parent)):
        if key not in REVIEW_SECTIONS and cond.get(key) != parent.get(key):
            raise SystemExit(f"{condition_id} {key} differs from {spec['base']}")
    for key in sorted(set(model) | set(parent["model"])):
        if key == "pass1_logit_capture" and spec.get("repair"):
            continue
        if key not in REVIEW_MODEL_KEYS and model.get(key) != parent["model"].get(key):
            raise SystemExit(f"{condition_id} model.{key} differs from {spec['base']}")
    capture, parent_capture = model["pass1_logit_capture"], parent["model"]["pass1_logit_capture"]
    if spec.get("repair"):
        if capture.get(REPAIR_KEY) != REPAIR_RULE or \
                {k: v for k, v in capture.items() if k != REPAIR_KEY} != parent_capture:
            raise SystemExit(f"{condition_id} must add only {REPAIR_KEY}: {REPAIR_RULE} to the capture block")
    elif REPAIR_KEY in capture:
        raise SystemExit(f"{condition_id} does not register {REPAIR_KEY}")
    if cond["replicates"]["seed_cell_id"] != condition_id:
        raise SystemExit(f"{condition_id} must use its own seed cell")
    source = cond["generator_source"]
    if source["condition_id"] != spec["g"]:
        raise SystemExit(f"{condition_id} must review {spec['g']}'s outputs")
    if "REQUIRED_BEFORE_RUN" in str(source["artifact_path"]) + str(source["pass1_run_id"]):
        raise SystemExit(f"{condition_id} generator_source is not filled in yet")
    if source["model_id"] != solo.condition(spec["g"])["model"]["model_id"]:
        raise SystemExit(f"{condition_id} generator_source.model_id differs from {spec['g']}'s model")
    return model


# ------------------------------------------------------------------ empty-token-bytes repair (c114)

def byte_level_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 bytes_to_unicode table that byte-level BPE vocabularies are written in."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def vocabulary_bytes(model: Mapping[str, Any]):
    """token id -> the exact bytes its vocabulary entry spells, from the served snapshot."""
    vocab = json.loads((Path(model["server_path"]) / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    by_id, decode = {int(i): tok for tok, i in vocab.items()}, byte_level_decoder()
    return lambda token_id: bytes(decode[ch] for ch in by_id[int(token_id)])


def repair_empty_token_bytes(trace: Mapping[str, Any] | None, content: str,
                             lookup) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]]]:
    """Fill bytes only for tokens returned without any, then locate the content with the unchanged
    rule of generated_token_logprobs.build_generated_token_trace. A trace that already aligned, or
    has no empty token, is returned untouched; a repaired trace that still does not align keeps
    status text_mismatch, so the run stops exactly as before."""
    if not trace or trace.get("status") == "ok":
        return trace, []
    tokens = [dict(t) for t in trace.get("tokens") or []]
    empty = [t for t in tokens if not t["bytes"] and t.get("token_id") is not None]
    if not empty:
        return trace, []
    repaired, reconstructed = [], bytearray()
    for t in tokens:
        if not t["bytes"] and t.get("token_id") is not None:
            fill = lookup(t["token_id"])
            t["bytes"] = list(fill)
            repaired.append({"index": t["index"], "token_id": t["token_id"], "bytes_hex": fill.hex()})
        t["start_byte"] = len(reconstructed)
        reconstructed.extend(bytes(t["bytes"]))
        t["end_byte"] = len(reconstructed)
    raw, content_bytes = bytes(reconstructed), content.encode("utf-8")
    start = None
    if content_bytes and raw.endswith(content_bytes):
        start = len(raw) - len(content_bytes)
    elif content_bytes and raw.count(content_bytes) == 1:
        start = raw.index(content_bytes)
    fixed = {**trace, "tokens": tokens, "status": "ok" if start is not None else "text_mismatch",
             "reconstructed_text_sha256": sha256_bytes(raw), "content_start_byte": start,
             "content_end_byte": None if start is None else start + len(content_bytes)}
    return fixed, repaired


def build_messages(model: Mapping[str, Any], problem: str, y0: str) -> list[dict[str, str]]:
    system, user = split_prompt(ROOT / model["prompt_pass1"])
    return [{"role": "system", "content": system},
            {"role": "user", "content": bigcodebench_study.user_message(user, problem=problem, y0=y0)}]


def stage_smoke(args: argparse.Namespace) -> None:
    model = require_registered(CONDITION_ID)
    capture_id = stable_id(args.run_id, "smoke", "same_forward_logit_capture")
    content, usage = solo.call_model(args, model, build_messages(model, solo.SMOKE_TASK, SMOKE_Y0), 1, capture_id)
    trace = usage.get("token_logprobs") or {}
    if REVIEWS[CONDITION_ID].get("repair"):
        trace, _ = repair_empty_token_bytes(trace, content or "", vocabulary_bytes(model))
        trace = trace or {}
    record = {"run_id": args.run_id, "problem_id": "smoke",
              "outputs": [{"raw_output": content, "token_logprobs": trace}]}
    capture = shared.wait_for_same_forward_capture(Path(args.capture_sidecar), capture_id, timeout_seconds=60)
    native = shared.build_same_forward_logit_record(run_id=args.run_id, pass1_record=record, capture=capture)
    problems = []
    if not content:
        problems.append("empty content")
    if trace.get("status") != "ok":
        problems.append(f"trace {trace.get('status')}")
    if native["status"] != "ok":
        problems.append(f"native capture {native['status']}")
    print(canonical_json({"stage": "smoke", "condition_id": CONDITION_ID, "native_status": native["status"],
                          "python_blocks": bigcodebench_study.python_block_count(content),
                          "finish_reason": usage.get("finish_reason"), "problems": problems}))
    if problems:
        raise SystemExit("smoke failed: " + "; ".join(problems))


def stage_review(args: argparse.Namespace) -> None:
    model = require_registered(CONDITION_ID)
    g_rows = {r["problem_id"]: r for r in read_jsonl(Path(args.g_records))}
    source_run = solo.condition(CONDITION_ID)["generator_source"]["pass1_run_id"]
    foreign = sorted({r["run_id"] for r in g_rows.values()} - {source_run})
    if foreign:
        raise RuntimeError(f"reviewed records come from {foreign}, not the registered {source_run}")
    manifest = solo.manifest_rows(Path(args.manifest), int(args.expected_rows))
    ids = [str(r["problem_id"]) for r in manifest]
    by_id = {str(r["problem_id"]): r for r in manifest}
    sources = solo.load_bigcodebench_rows(ids)
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    lookup = vocabulary_bytes(model) if REVIEWS[CONDITION_ID].get("repair") else None
    written = 0
    for problem_id in ids:
        if (problem_id,) in done:
            continue
        row, g_row = sources[problem_id], g_rows[problem_id]
        y0 = g_row["outputs"][0]["raw_output"] or ""
        messages = build_messages(model, row["problem_text"], y0)
        seed = derive_call_seed(solo.REPLICATE_PLAN_ID, "review", args.replicate_id, int(args.base_seed),
                                CONDITION_ID, problem_id, 0)
        capture_id = stable_id(args.run_id, problem_id, "same_forward_logit_capture")
        content, usage = solo.call_model(args, model, messages, seed, capture_id)
        trace = usage.pop("token_logprobs", None)
        if lookup is not None:
            trace, repaired = repair_empty_token_bytes(trace, content or "", lookup)
            if repaired:
                append_jsonl(output.parent / REPAIRS_FILE, {
                    "problem_id": problem_id, "record_id": stable_id(args.run_id, problem_id, "review"),
                    "trace_status_after": trace["status"], "repaired_tokens": repaired})
        record = {
            "schema_version": "code-review-record-v1",
            "record_id": stable_id(args.run_id, problem_id, "review"),
            "run_id": args.run_id, "condition_id": CONDITION_ID,
            "condition_hash": shared.condition_hash(), "dataset_id": solo.DATASET_ID,
            "replicate_id": args.replicate_id, "stage": STAGE,
            "problem_id": problem_id, "problem_text": row["problem_text"],
            "problem_metadata": {"entry_point": row["entry_point"],
                                 "libs": str(by_id[problem_id].get("libs", ""))},
            "generator": solo.model_identity(model),
            "outputs": [{"sample_index": 0, "call_seed": seed, "raw_output": content,
                         "python_block_count": bigcodebench_study.python_block_count(content),
                         "token_logprobs": trace, "usage": shared.base.compact_usage(usage)}],
            "y0_sample_index": 0, "created_at_utc": utc_now(),
            "source_g_run_id": g_row["run_id"], "source_g_record_id": g_row["record_id"],
            "source_g_output_sha256": sha256_bytes(y0.encode("utf-8")),
        }
        append_jsonl(output, record)
        capture = shared.wait_for_same_forward_capture(Path(args.capture_sidecar), capture_id, timeout_seconds=60)
        native = shared.build_same_forward_logit_record(run_id=args.logit_run_id, pass1_record=record,
                                                        capture=capture)
        append_jsonl(Path(args.logit_output), native)
        if native["status"] != "ok":
            raise RuntimeError(f"same-forward logit capture failed for {problem_id}: {native['status']}")
        written += 1
    print(canonical_json({"stage": "review", "written": written, "total": len(ids)}))


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


def stage_grade(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.records))
    results = {r["record_id"]: r for r in read_jsonl(Path(args.results))}
    if len(results) != len(records):
        raise RuntimeError(f"{len(results)} grader results for {len(records)} records")
    errors = [r for r in results.values() if r["status"] == "grader_error"]
    if errors:
        raise RuntimeError(f"{len(errors)} grader_error rows, e.g. {errors[0]}; the run is rejected")
    g_grades = {r["problem_id"]: r for r in read_jsonl(Path(args.g_grades)) if r["outcome"] == "u0"}
    output = Path(args.output)
    output.unlink(missing_ok=True)
    for record in records:
        pid = record["problem_id"]
        result = results.get(record["record_id"])
        if result is None or result["task_id"] != pid:
            raise RuntimeError(f"no grader result for {pid}")
        base = {"problem_id": pid, "source_record_id": record["record_id"]}
        u_review = int(result["u"])
        append_jsonl(output, {**base, "outcome": "u_review", "u": u_review, "status": result["status"],
                              "detail": result["detail"], "seconds": result["seconds"]})
        fallback = result["status"] == "model_invalid"
        append_jsonl(output, {**base, "outcome": "u_gv", "u": int(g_grades[pid]["u"]) if fallback else u_review,
                              "delivered_from": "g_fallback" if fallback else "review"})
        append_jsonl(output, {**base, "outcome": "u_gv_strict", "u": u_review})
    print(canonical_json({"stage": "grade", "rows": len(records)}))


def final_block(text: str) -> str | None:
    blocks = FINAL_BLOCK.findall(text or "")
    return re.sub(r"\s+", " ", blocks[-1]).strip() if blocks else None


def stage_analyze(args: argparse.Namespace) -> None:
    records = {r["problem_id"]: r for r in read_jsonl(Path(args.records))}
    evidence = {r["problem_id"]: r for r in read_jsonl(Path(args.evidence))}
    grades: dict[str, dict[str, Mapping[str, Any]]] = collections.defaultdict(dict)
    for row in read_jsonl(Path(args.grades)):
        grades[row["outcome"]][row["problem_id"]] = row
    g_grades = {r["problem_id"]: r for r in read_jsonl(Path(args.g_grades)) if r["outcome"] == "u0"}
    g_text = {r["problem_id"]: r["outputs"][0]["raw_output"] for r in read_jsonl(Path(args.g_records))}
    expected = int(args.expected_rows)
    if not (len(records) == len(evidence) == len(grades["u_gv"]) == expected):
        raise RuntimeError(f"incomplete inputs: {len(records)}, {len(evidence)}, {len(grades['u_gv'])}")
    ids = sorted(records)
    total = lambda outcome: sum(int(grades[outcome][p]["u"]) for p in ids)  # noqa: E731
    g_correct = sum(int(g_grades[p]["u"]) for p in ids)
    repair = sum(int(grades["u_gv"][p]["u"]) and not int(g_grades[p]["u"]) for p in ids)
    harm = sum(not int(grades["u_gv"][p]["u"]) and int(g_grades[p]["u"]) for p in ids)
    fallback = sum(grades["u_gv"][p]["delivered_from"] == "g_fallback" for p in ids)
    details = collections.Counter(grades["u_review"][p]["detail"] for p in ids)
    relation = collections.Counter(
        "reviewer_no_fenced_block" if final_block(records[p]["outputs"][0]["raw_output"]) is None
        else "same_final_block_as_reviewed"
        if final_block(records[p]["outputs"][0]["raw_output"]) == final_block(g_text[p])
        else "changed_final_block" for p in ids)
    usage = [records[p]["outputs"][0]["usage"] for p in ids]
    tokens = [int(u.get("output_tokens") or 0) for u in usage]
    truncated = [p for p, u in zip(ids, usage) if u.get("finish_reason") == "length"]
    loops = [p for p in truncated if solo.repetition_share(records[p]["outputs"][0]["raw_output"]) >= 0.5]
    missing = {name: sum(evidence[p]["missingness"][name] for p in ids) for name in evidence[ids[0]]["features"]}
    repairs_path = Path(args.records).parent / REPAIRS_FILE
    kept = {records[p]["record_id"] for p in ids}
    repairs = [r for r in (read_jsonl(repairs_path) if repairs_path.exists() else []) if r["record_id"] in kept]
    report_repairs = {"rows": len(repairs), "tokens": sum(len(r["repaired_tokens"]) for r in repairs),
                      "problem_ids": [r["problem_id"] for r in repairs]}
    report: dict[str, Any] = {
        "condition_id": CONDITION_ID, "run_id": args.run_id, "role": ROLE, "rows": expected,
        "review_passed": total("u_review"), "gv_primary_passed": total("u_gv"),
        "gv_strict_passed": total("u_gv_strict"), "fallback_rows": fallback,
        "reviewed_alone_passed_same_rows": g_correct, "gv_vs_reviewed_repair": repair,
        "gv_vs_reviewed_harm": harm, "review_grader_detail_counts": dict(details),
        "final_block_relation_to_reviewed": dict(relation),
        "truncated_rows": len(truncated), "truncated_repetition_loops": len(loops),
        "output_tokens": {"sum": sum(tokens), "mean": statistics.mean(tokens),
                          "median": statistics.median(tokens), "max": max(tokens)},
        "scalar_missingness": missing, "feature_contract": FEATURE_CONTRACT, "development_only": True,
    }
    repair_enabled = bool(REVIEWS[CONDITION_ID].get("repair"))
    if repair_enabled:
        report["empty_token_bytes_repaired"] = report_repairs
    lines = [f"# {CONDITION_ID} BigCodeBench review summary", "",
             f"- Rows: {expected}. GV delivered passed (primary): {total('u_gv')}/{expected}; strict: "
             f"{total('u_gv_strict')}/{expected}; fallback rows: {fallback}.",
             f"- Reviewed model alone on the same rows: {g_correct}/{expected}; GV vs it repair/harm {repair}/{harm}.",
             f"- Reviewer grader detail: {json.dumps(dict(details), sort_keys=True)}.",
             f"- Reviewer's final fenced block relative to the reviewed output's (whitespace-normalized): "
             f"{dict(relation)}.",
             f"- Output tokens: sum {sum(tokens)}, mean {statistics.mean(tokens):.1f}, max {max(tokens)}; "
             f"truncated at {solo.BUDGET}: {len(truncated)} rows, {len(loops)} of them repetition loops.",
             f"- Scalar missingness: {json.dumps({k: v for k, v in missing.items() if v}, sort_keys=True)}",
             *([f"- Tokens returned without bytes and filled from the vocabulary ({REPAIR_RULE}): "
                f"{report_repairs['tokens']} tokens on {report_repairs['rows']} rows {report_repairs['problem_ids']}."]
               if repair_enabled else []),
             "- The comparison with the reviewer alone and max(G, V) is a separate analysis over the archives.",
             "", "Evidence was written before grading. Grading runs the task's own tests in the pinned "
             "BigCodeBench evaluate container with networking disabled.", ""]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output.parent / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-runnable")
    smoke = sub.add_parser("smoke")
    for name in ("--run-id", "--api-base", "--served-model", "--capture-sidecar"):
        smoke.add_argument(name, required=True)
    review = sub.add_parser("review")
    for name in ("--run-id", "--api-base", "--served-model", "--manifest", "--output", "--g-records",
                 "--capture-sidecar", "--logit-output", "--logit-run-id"):
        review.add_argument(name, required=True)
    review.add_argument("--expected-rows", type=int, default=solo.EXPECTED_ROWS)
    review.add_argument("--replicate-id", default=solo.REPLICATE_ID)
    review.add_argument("--base-seed", type=int, default=solo.BASE_SEED)
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
    for name in ("--records", "--results", "--g-grades", "--output"):
        grade.add_argument(name, required=True)
    analyze = sub.add_parser("analyze")
    for name in ("--run-id", "--records", "--evidence", "--grades", "--g-records", "--g-grades", "--output"):
        analyze.add_argument(name, required=True)
    analyze.add_argument("--expected-rows", type=int, default=solo.EXPECTED_ROWS)
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
                              "reviews": REVIEWS[CONDITION_ID]["g"], "max_new_tokens": solo.BUDGET,
                              "dataset_id": solo.DATASET_ID}))
    elif args.command == "smoke":
        stage_smoke(args)
    elif args.command == "review":
        stage_review(args)
    elif args.command == "reconcile":
        print(canonical_json({"stage": "reconcile", **mathfull.reconcile(Path(args.records), Path(args.traces))}))
    elif args.command == "evidence":
        stage_evidence(args)
    elif args.command == "grade-requests":
        solo.stage_grade_requests(args)
    elif args.command == "grade":
        stage_grade(args)
    else:
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
