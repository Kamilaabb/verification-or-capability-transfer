#!/usr/bin/env python3
"""c014: cheap controlled self-verification pilot on the math dev300 manifest.

Built on run_c002_dev30 for everything the domain already solved - the frozen manifest
join, the math-verify grading contract and the API plumbing - and
adds what router_v1 needs and authority_v6 has no place for: a second pass that always
emits an answer, a paired vLLM-native raw-logit capture, two-arm counterfactuals carrying per-row
token costs, and an analysis whose primary outcome is cost.

There is no authority cap and no action vocabulary here. The self-check's verdict is
recorded for analysis and never gates execution, because a row where the model says
"correct" and still changes the answer is the harm channel this pilot exists to count.

Every stage is resumable by key, and the model identity is read from the condition
rather than hardcoded, so this file is self-testable on CPU before Qwen3-4B is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c002_dev30 as base  # noqa: E402

from authority_allocation import derive_call_seed  # noqa: E402
from authority_allocation.api import api_call, chat_template_kwargs_for  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.prompts import render, split_prompt  # noqa: E402
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.router_semantics import (  # noqa: E402
    best_fixed_arm,
    bootstrap_policy_ci,
    deployed_outcomes,
    fixed_arm_values,
    oracle_headroom,
    policy_value,
)
from authority_allocation.serialization import (  # noqa: E402
    canonical_json,
    sha256_bytes,
    sha256_file,
    stable_id,
    utc_now,
)

CONDITION_ID = "c014"
CONDITION_PATH = ROOT / "conditions" / CONDITION_ID / "condition.yaml"
DATASET_ID = "math500-r6e4ed1-l3to5-dev300-s20260910"
REPLICATE_PLAN_ID = "r007"
STAGE = "pilot300"
SEED_CELL_ID = "c014"
EXPECTED_ROWS = 300
RAW_TOP1_TOLERANCE = 1e-6
IDENTITY_TOLERANCE = 1e-4

SELF_CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reasoning", "final_answer"],
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect", "uncertain"]},
        "reasoning": {"type": "string"},
        "final_answer": {"type": "string"},
    },
}


def configure() -> None:
    """Point the reused c002 infrastructure at c014 without editing it."""
    base.CONDITION_ID = CONDITION_ID
    base.CONDITION_PATH = CONDITION_PATH
    base.DATASET_ID = DATASET_ID
    base.REPLICATE_PLAN_ID = REPLICATE_PLAN_ID
    base.STAGE = STAGE
    base.EXPECTED_ROWS = EXPECTED_ROWS


def condition() -> dict[str, Any]:
    return resolve_condition(CONDITION_PATH, repository_root=ROOT)


def condition_hash() -> str:
    """Hash c014's resolved behavior and every registered repository file.

    The older math runner assumes a ``protocol.schemas_directory``. router_v1 instead
    registers its heterogeneous schemas under ``protocol.record_schemas``, so c014 owns
    this small adapter while retaining the same canonical payload and exclusion rules.
    """
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
                # Behavioral prose is also string-valued and is not a repository path.
                # Linux raises ENAMETOOLONG where Windows simply returns false.
                is_file = False
            if is_file:
                referenced[value] = sha256_file(candidate)

    find_paths(behavioral)
    payload = {"condition": behavioral, "referenced_files": referenced}
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def model_block() -> dict[str, Any]:
    return condition()["model"]


def require_frozen_model() -> dict[str, Any]:
    """The runnable gate. A parameterised runner is fine; an unpinned Slurm job is not.

    The paired native capture is meaningful only against the exact weights, tokenizer,
    chat template and greedy settings of the original call, so the identity fields are
    checked here rather than discovered halfway through a GPU allocation.
    """
    model = model_block()
    missing = [key for key in ("revision", "server_path", "chat_template_hash")
               if not model.get(key)]
    if missing:
        raise SystemExit(
            f"{CONDITION_ID} is not runnable: {CONDITION_PATH.relative_to(ROOT)} "
            f"model.{'/'.join(missing)} "
            "is still null. Freeze the Qwen3-4B snapshot and record its preflight first."
        )
    if model.get("thinking") is not False:
        raise SystemExit(f"{CONDITION_ID} requires thinking: false on both passes")
    for key in ("pass1_decoding", "pass2_decoding"):
        if float(model[key]["temperature"]) != 0.0:
            raise SystemExit(
                f"{CONDITION_ID} requires {key}.temperature == 0.0 for a recoverable trace")
    return model


def model_identity(prompt_path: Path) -> dict[str, Any]:
    """Build the pass-1 identity from c014's single shared model block."""
    model = require_frozen_model()
    return {
        "model_id": model["model_id"],
        "revision": model["revision"],
        "server_path": model["server_path"],
        "backend": model["backend"],
        "prompt_sha256": sha256_file(prompt_path),
        "chat_template_sha256": model["chat_template_hash"],
        "dtype": model["dtype"],
        "quantization": model["quantization"],
    }


def manifest_ids(path: Path, expected: int = EXPECTED_ROWS) -> list[str]:
    """Read the frozen manifest, asserting its size.

    `expected` is explicit rather than defeated with a flag, so a 5-row preflight slice
    states its size and the 300-row run still cannot silently run short.
    """
    ids = [base.manifest_fields(row)[0] for row in read_jsonl(path)]
    if len(ids) != expected:
        raise ValueError(f"expected {expected} manifest rows, got {len(ids)}")
    return ids


def extract_answer(content: str | None) -> tuple[str | None, str]:
    """The extracted answer and parse status written into a pass-1 or review record.

    math_verify gives up after its 10-second parsing budget on a degenerate output (c030 row
    15 repeated "= \\boxed{\\frac{12}{2}}" several hundred times) and raises, which once
    killed a whole job. Such a row keeps its raw output and trace, records no answer and
    is marked model_invalid, so it is graded wrong like any other output with no
    recoverable answer. Any other parser error still stops the run. Runs completed before
    this function existed never hit the timeout: it would have stopped them.

    The same holds for an integer too long for Python's int-from-string limit (c225 row
    615: Llama-3.2-3B wrote an 11,859-digit number; sympy raised SympifyError three times
    in a row and killed every attempt). Only that exact cause is caught; runs completed
    before this change never hit it, since it would have stopped them.
    """
    if not content:
        return None, "model_invalid"
    from math_verify.errors import TimeoutException
    from sympy.core.sympify import SympifyError

    try:
        return base.math_parse(content)[1], "ok"
    except TimeoutException:
        return None, "model_invalid"
    except SympifyError as exc:
        if "integer string conversion" not in str(exc):
            raise
        return None, "model_invalid"


def _identity(run_id: str) -> dict[str, Any]:
    return {"run_id": run_id, "condition_id": CONDITION_ID,
            "condition_hash": condition_hash(), "dataset_id": DATASET_ID}


def stage_smoke(args: argparse.Namespace) -> None:
    """Prove the served model can do what the archive later assumes it did.

    The paired native capture requires pass 1's content trace and token IDs, so those
    are checked here before any rows are spent.
    """
    model = require_frozen_model()
    system, user = split_prompt(ROOT / model["prompt_pass1"])
    decoding = dict(model["pass1_decoding"])
    decoding["output_logprobs"] = {"enabled": True, "mode": "raw_logprobs",
                                   "return_token_ids": True}
    content, _, usage = api_call(
        api_base=args.api_base, served_model=args.served_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": render(user, problem="What is $2+2$?")}],
        decoding=decoding, seed=1, thinking=False,
        chat_template_kwargs=chat_template_kwargs_for(model))
    trace = usage.get("token_logprobs") or {}
    if not content:
        raise RuntimeError("smoke returned empty content")
    if trace.get("status") != "ok":
        raise RuntimeError(f"smoke trace not exactly alignable: {trace.get('status')}")
    if any(token.get("token_id") is None for token in trace.get("tokens") or []):
        raise RuntimeError("smoke trace has no token ids; native capture cannot be paired")
    print(canonical_json({"stage": "smoke", "status": "passed",
                          "tokens": len(trace.get("tokens") or []),
                          "finish_reason": usage.get("finish_reason")}))


# --------------------------------------------------------------------------- pass 1

def stage_pass1(args: argparse.Namespace) -> None:
    model = require_frozen_model()
    ids = manifest_ids(Path(args.manifest), int(args.expected_rows))
    sources = base.load_source_rows(ids)
    base.audit_gold_answers(sources)
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    system, user = split_prompt(ROOT / model["prompt_pass1"])
    template_kwargs = chat_template_kwargs_for(model)
    written = 0
    for problem_id in ids:
        if (problem_id,) in done:
            continue
        seed = derive_call_seed(REPLICATE_PLAN_ID, "pass1", args.replicate_id,
                                int(args.base_seed), SEED_CELL_ID, problem_id, 0)
        capture_id = stable_id(args.run_id, problem_id, "same_forward_logit_capture")
        content, reasoning, usage = api_call(
            api_base=args.api_base, served_model=args.served_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": render(user, problem=sources[problem_id]["problem"])}],
            decoding={**model["pass1_decoding"],
                      "output_logprobs": {"enabled": True, "mode": "raw_logprobs",
                                          "return_token_ids": True},
                      "vllm_xargs": {"c014_capture_id": capture_id}},
            seed=seed, thinking=False,
            chat_template_kwargs=template_kwargs,
        )
        trace = usage.pop("token_logprobs", None)
        extracted, parse_status = extract_answer(content)
        pass1_record = {
            "schema_version": "g-record-single-v3",
            "record_id": stable_id(args.run_id, problem_id, "pass1"),
            **_identity(args.run_id), "replicate_id": args.replicate_id, "stage": STAGE,
            "problem_id": problem_id, "problem_text": sources[problem_id]["problem"],
            "generator": model_identity(ROOT / model["prompt_pass1"]),
            "outputs": [{"sample_index": 0, "call_seed": seed, "raw_output": content,
                         "extracted_answer": extracted, "parse_status": parse_status,
                         "token_logprobs": trace, "usage": base.compact_usage(usage)}],
            "y0_sample_index": 0, "created_at_utc": utc_now(),
        }
        append_jsonl(output, pass1_record)
        capture = wait_for_same_forward_capture(
            Path(args.capture_sidecar), capture_id, timeout_seconds=30)
        logit_record = build_same_forward_logit_record(
            run_id=args.logit_run_id, pass1_record=pass1_record, capture=capture)
        append_jsonl(Path(args.logit_output), logit_record)
        if logit_record["status"] != "ok":
            raise RuntimeError(
                f"same-forward logit capture failed for {problem_id}: "
                f"{logit_record['status']}")
        written += 1
    print(canonical_json({"stage": "pass1", "written": written, "total": len(ids)}))


# ------------------------------------------------------- same-forward raw logits

def wait_for_same_forward_capture(
    path: Path, capture_id: str, *, timeout_seconds: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            for row in read_jsonl(path):
                if row.get("capture_id") == capture_id:
                    return row
        time.sleep(0.05)
    raise RuntimeError(f"vLLM did not flush same-forward capture {capture_id}")


def build_same_forward_logit_record(
    *, run_id: str, pass1_record: Mapping[str, Any], capture: Mapping[str, Any]
) -> dict[str, Any]:
    problem_id = str(pass1_record["problem_id"])
    source = pass1_record["outputs"][0]
    trace = source.get("token_logprobs") or {}
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
            values = [raw.get(name) for name in (
                "max_raw_logit", "logsumexp", "entropy", "top1_top2_margin")]
            if (raw.get("sampled_token_id") != token.get("token_id")
                    or not isinstance(sampled_raw, (int, float))
                    or len(top_ids) != 20 or len(top_logits) != 20
                    or len(set(top_ids)) != 20
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in [sampled_raw, *values, *top_logits])):
                status = "raw_trace_missing"
                positions = []
                identity_error = None
                break
            sampled_logprob = float(token["logprob"])
            step_error = abs(sampled_logprob - (
                float(sampled_raw) - float(raw["logsumexp"])))
            identity_error = max(identity_error, step_error)
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
        "schema_version": "native-logit-trace-record-v1",
        "record_id": stable_id(run_id, problem_id, "same_forward_logit_capture"),
        "run_id": run_id, "source_pass1_run_id": pass1_record["run_id"],
        "condition_id": CONDITION_ID, "condition_hash": condition_hash(),
        "problem_id": problem_id,
        "capture_id": str(capture.get("capture_id", "")),
        "capture_backend": "vllm_same_forward_logits_processor_v1",
        "status": status, "same_forward": True,
        "exact_token_match": exact_tokens,
        "identity_max_abs_error": identity_error,
        "positions": positions, "created_at_utc": utc_now(),
    }


# --------------------------------------------------------------------------- pass 2

def pass2_messages(problem: str, prior_solution: str) -> list[dict[str, str]]:
    """Render the active condition's frozen self-check prompt as one user turn."""
    path = ROOT / model_block()["prompt_pass2"]
    template = path.read_text(encoding="utf-8")
    return [{"role": "user", "content": render(
        template, problem=problem, prior_solution=prior_solution)}]


def pass2_answer_text(check: Mapping[str, Any]) -> str | None:
    """The final_answer text a self-check's extracted_answer was parsed from."""

    if check.get("extracted_answer") is None:
        return None
    try:
        return json.loads(check["raw_output"])["final_answer"]
    except (TypeError, ValueError, KeyError):
        return None


def stage_pass2(args: argparse.Namespace) -> None:
    model = require_frozen_model()
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for record in read_jsonl(Path(args.pass1_records)):
        problem_id = record["problem_id"]
        if (problem_id,) in done:
            continue
        sample = record["outputs"][0]
        seed = derive_call_seed(REPLICATE_PLAN_ID, "pass2", args.replicate_id,
                                int(args.base_seed), SEED_CELL_ID, problem_id, 0)
        content = verdict = answer = None
        status = "ok"
        try:
            content, _, usage = api_call(
                api_base=args.api_base, served_model=args.served_model,
                messages=pass2_messages(
                    record["problem_text"], sample["raw_output"] or ""),
                decoding=model["pass2_decoding"], seed=seed, thinking=False,
                schema=SELF_CHECK_SCHEMA)
        except Exception as error:  # noqa: BLE001 - a dead call is data, not a crash
            status, usage = "call_error", {"retries": 0, "finish_reason": None}
            print(canonical_json({"problem_id": problem_id, "call_error": str(error)[:200]}))
        else:
            try:
                parsed = json.loads(content)
                verdict = parsed["verdict"]
                answer = base.math_parse(parsed["final_answer"])[1]
                if answer is None:
                    status = "model_invalid"
            except Exception:  # noqa: BLE001
                status = "model_invalid"

        # Compare the texts that were parsed, not their sympy strings, which do not
        # re-parse (docs/analysis/c018_report_format.md).
        prior = base.answer_text(sample)
        changed = None
        if answer is not None and prior is not None:
            equivalent, left_ok, right_ok = base.math_equivalent(
                prior, parsed["final_answer"])
            changed = None if equivalent is None else (not equivalent)

        usage.pop("token_logprobs", None)
        append_jsonl(output, {
            "schema_version": "self-check-record-v1",
            "record_id": stable_id(args.run_id, problem_id, "pass2"),
            "run_id": args.run_id, "source_pass1_run_id": record["run_id"],
            **{k: v for k, v in _identity(args.run_id).items() if k != "run_id"},
            "replicate_id": args.replicate_id, "stage": STAGE, "cell_id": CONDITION_ID,
            "problem_id": problem_id, "call_seed": seed, "raw_output": content,
            "verdict": verdict, "extracted_answer": answer, "answer_changed": changed,
            "parse_status": status, "token_logprobs": None,
            "usage": base.compact_usage(usage), "created_at_utc": utc_now(),
        })
        written += 1
    print(canonical_json({"stage": "pass2", "written": written}))


# ------------------------------------------------------------- grading + two arms

def stage_grade(args: argparse.Namespace) -> None:
    """Grade y0 and y_self against the frozen math-verify contract."""
    ids = manifest_ids(Path(args.manifest), int(args.expected_rows))
    sources = base.load_source_rows(ids)
    checks = {row["problem_id"]: row for row in read_jsonl(Path(args.pass2_records))}
    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "outcome"))
    counts: dict[str, int] = {}
    for record in read_jsonl(Path(args.pass1_records)):
        problem_id = record["problem_id"]
        gold = sources[problem_id]["gold"]
        # Grade the texts the answers were parsed from; extracted_answer holds their
        # sympy strings, which do not re-parse (docs/analysis/c018_report_format.md).
        wanted = [("u0", base.answer_text(record["outputs"][0]))]
        check = checks.get(problem_id)
        if check is not None and check["parse_status"] in ("ok", "retry_ok"):
            wanted.append(("u_self", pass2_answer_text(check)))
        for outcome, prediction in wanted:
            if (problem_id, outcome) in done:
                continue
            value, grader_status = base.grade_prediction(gold, prediction)
            append_jsonl(output, {"problem_id": problem_id, "outcome": outcome,
                                  "value": value, "grader_status": grader_status,
                                  "created_at_utc": utc_now()})
            counts[grader_status] = counts.get(grader_status, 0) + 1
    print(canonical_json({"stage": "grade", "by_status": counts}))


def stage_counterfactual(args: argparse.Namespace) -> None:
    """Build the two-arm records, each carrying both arms' token costs."""
    grades: dict[tuple[str, str], dict[str, Any]] = {
        (row["problem_id"], row["outcome"]): row for row in read_jsonl(Path(args.grades))}
    checks = {row["problem_id"]: row for row in read_jsonl(Path(args.pass2_records))}
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for record in read_jsonl(Path(args.pass1_records)):
        problem_id = record["problem_id"]
        if (problem_id,) in done:
            continue
        check = checks[problem_id]
        u0_row = grades.get((problem_id, "u0"), {})
        self_row = grades.get((problem_id, "u_self"), {})
        u0, u_self = u0_row.get("value"), self_row.get("value")
        status = check["parse_status"]
        arms = deployed_outcomes(u0=u0, u_self=u_self, self_check_status=status)
        append_jsonl(output, {
            "schema_version": "router-counterfactual-v1",
            "record_id": stable_id(args.run_id, problem_id, "counterfactual"),
            "run_id": args.run_id, "source_pass1_run_id": record["run_id"],
            "source_pass2_run_id": check["run_id"], "condition_id": CONDITION_ID,
            "condition_hash": condition_hash(), "problem_id": problem_id,
            "u0": u0, "u_self": u_self, **arms, "self_check_status": status,
            "grader_status": {"y0": u0_row.get("grader_status", "not_run"),
                              "y_self": self_row.get("grader_status", "not_run")},
            "tokens_pass1": (record["outputs"][0]["usage"] or {}).get("output_tokens"),
            "tokens_pass2": (check["usage"] or {}).get("output_tokens"),
            "created_at_utc": utc_now(),
        })
        written += 1
    print(canonical_json({"stage": "counterfactual", "written": written}))


# -------------------------------------------------------------------------- analyze

def oof_frontier(rows: list[dict[str, Any]], scores: Mapping[str, float]) -> list[dict[str, Any]]:
    """The whole out-of-fold frontier, reported entire.

    No champion operating point is selected. Cross-fitting makes each row's score come
    from a model that never saw it, but reading the frontier and then reporting its best
    threshold as the result is policy-selection optimism, which cross-fitting does not
    remove. This pilot registers no confirmatory claim, so the curve is the deliverable.
    """
    ranked = sorted(rows, key=lambda row: scores.get(row["problem_id"], 0.0), reverse=True)
    frontier = []
    for k in range(0, len(ranked) + 1):
        called = [row["problem_id"] for row in ranked[:k]]
        value = policy_value(rows, called)
        frontier.append({"k": k, "call_fraction": value["call_fraction"],
                         "correct": value["correct"], "accuracy": value["accuracy"],
                         "output_tokens": value["output_tokens"]})
    return frontier


def stage_analyze(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.counterfactuals))
    traces = {row["problem_id"]: row for row in read_jsonl(Path(args.logit_traces))} \
        if args.logit_traces else {}

    head = oracle_headroom(rows)
    arms = fixed_arm_values(rows)
    best = best_fixed_arm(rows)
    decision_relevant = head["decision_relevant"]

    report: dict[str, Any] = {
        "condition_id": CONDITION_ID, "run_id": args.run_id, "rows": len(rows),
        "pools": head, "fixed_arms": arms, "best_fixed_arm": best,
        "gate_0": {
            "G1_decision_relevant_at_least_20": {
                "value": decision_relevant, "threshold": 20,
                "passed": decision_relevant >= 20,
                "meaning": "the only stop criterion; below it there is nothing to route",
            },
            "superiority_testing_permitted": {
                "value": head["headroom_over_best_fixed"], "threshold": 6,
                "permitted": head["headroom_over_best_fixed"] >= 6,
                "meaning": (
                    "min(R,H) >= 6 is what McNemar(min(R,H), 0) needs, and that is a "
                    "PERFECT router taking every win with no losses. It says an accuracy "
                    "improvement is detectable in principle, never that one exists. "
                    "Whether it does is decided by a held-out paired test."
                ),
            },
            "no_average_effect_criterion": (
                "A1 - A0 is reported as description only. At R == H the arms tie and "
                "McNemar is maximally insignificant while the headroom is at its largest, "
                "so an average-effect gate would reject exactly the most routable case."
            ),
        },
        "a1_minus_a0_descriptive": {
            "repair": head["repair"], "harm": head["harm"],
            "net": head["repair"] - head["harm"],
        },
        "self_check_behaviour": {},
        "native_logit_capture": {},
    }

    if traces:
        ok = [t for t in traces.values() if t["status"] == "ok"]
        worst_identity = max((t["identity_max_abs_error"] or 0.0)
                             for t in traces.values())
        exact = all(t["same_forward"] and t["exact_token_match"] for t in ok)
        report["native_logit_capture"] = {
            "ok": len(ok), "total": len(traces),
            "capture_backend": "vllm_same_forward_logits_processor_v1",
            "identity_max_abs_error": worst_identity,
            "tolerance": IDENTITY_TOLERANCE,
            "r2_features_valid": len(ok) == len(traces) and exact
                                 and worst_identity <= IDENTITY_TOLERANCE,
            "note": ("R2 uses online reductions of the same raw-logit tensor from "
                     "which vLLM generated each pass-1 token. Full-vocabulary logits "
                     "never leave the GPU and are never written to disk."),
        }

    bootstrap_seed = int(args.bootstrap_seed)
    report["bootstrap_a1_vs_a0"] = bootstrap_policy_ci(
        rows, [row["problem_id"] for row in rows], baseline=[],
        seed=bootstrap_seed, repetitions=int(args.bootstrap_repetitions))

    if args.router_scores:
        scores = {row["problem_id"]: float(row["score"])
                  for row in read_jsonl(Path(args.router_scores))}
        report["oof_frontier"] = oof_frontier(rows, scores)
        report["oof_frontier_is_exploratory"] = (
            "Reported entire. No champion threshold is selected and no confirmatory "
            "claim is made from it. A future confirmatory condition must pick the "
            "threshold inside the training folds and evaluate on the outer held-out "
            "fold, or pick it on development and run the frozen test exactly once.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    gate = report["gate_0"]["G1_decision_relevant_at_least_20"]
    capture = report["native_logit_capture"]
    summary = "\n".join([
        f"# {CONDITION_ID} Gate 0 summary",
        "",
        f"- Rows: {report['rows']}",
        f"- Repair / harm: {head['repair']} / {head['harm']}",
        f"- Decision-relevant rows: {decision_relevant}",
        f"- Gate 0: {'PASS' if gate['passed'] else 'STOP'} "
        f"({gate['value']} vs threshold {gate['threshold']})",
        f"- Best fixed arm: {best['name']} ({best['correct']} correct)",
        f"- Native raw-logit capture: {capture.get('ok', 0)}/"
        f"{capture.get('total', 0)} valid; R2 valid={capture.get('r2_features_valid', False)}",
        "",
        "The complete machine-readable result is `analysis.json`. No champion router "
        "threshold is selected in this Gate 0 run.",
        "",
    ])
    (output.parent / "summary.md").write_text(summary, encoding="utf-8")
    print(canonical_json({k: report[k] for k in ("pools", "best_fixed_arm", "gate_0")}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p: argparse.ArgumentParser, api: bool = False,
               replicate: str = "p01") -> argparse.ArgumentParser:
        p.add_argument("--run-id", required=True)
        p.add_argument("--replicate-id", default=replicate)
        p.add_argument("--base-seed", default=20260910)
        if api:
            p.add_argument("--api-base", required=True)
            p.add_argument("--served-model", required=True)
        return p

    sm = common(sub.add_parser("smoke"), api=True)
    sm.set_defaults(function=stage_smoke)

    p1 = common(sub.add_parser("pass1"), api=True)
    p1.add_argument("--manifest", required=True)
    p1.add_argument("--expected-rows", default=EXPECTED_ROWS)
    p1.add_argument("--output", required=True)
    p1.add_argument("--capture-sidecar", required=True)
    p1.add_argument("--logit-output", required=True)
    p1.add_argument("--logit-run-id", required=True)
    p1.set_defaults(function=stage_pass1)

    p2 = common(sub.add_parser("pass2"), api=True, replicate="s01")
    p2.add_argument("--pass1-records", required=True)
    p2.add_argument("--output", required=True)
    p2.set_defaults(function=stage_pass2)

    gr = sub.add_parser("grade")
    gr.add_argument("--manifest", required=True)
    gr.add_argument("--expected-rows", default=EXPECTED_ROWS)
    gr.add_argument("--pass1-records", required=True)
    gr.add_argument("--pass2-records", required=True)
    gr.add_argument("--output", required=True)
    gr.set_defaults(function=stage_grade)

    cf = common(sub.add_parser("counterfactual"))
    cf.add_argument("--pass1-records", required=True)
    cf.add_argument("--pass2-records", required=True)
    cf.add_argument("--grades", required=True)
    cf.add_argument("--output", required=True)
    cf.set_defaults(function=stage_counterfactual)

    an = sub.add_parser("analyze")
    an.add_argument("--run-id", required=True)
    an.add_argument("--counterfactuals", required=True)
    an.add_argument("--logit-traces", default=None)
    an.add_argument("--router-scores", default=None)
    an.add_argument("--bootstrap-seed", default=20260921)
    an.add_argument("--bootstrap-repetitions", default=10000)
    an.add_argument("--output", required=True)
    an.set_defaults(function=stage_analyze)

    gate = sub.add_parser("check-runnable")
    gate.set_defaults(function=lambda args: print(canonical_json(
        {"runnable": True, "revision": require_frozen_model()["revision"]})))
    return parser


def main() -> int:
    configure()
    args = build_parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
