#!/usr/bin/env python3
"""Gate 1: add rewrite over the full development split and label (z0, z1, z2).

Built on run_c005_gate0, the way run_c006_guidance is, because generate, verify and the
grading ingest are unchanged. What Gate 1 adds is the rewrite call, a third graded
outcome, and the counterfactual labels that Gate 0 could not produce.

Reuse is by key, not by copying logic. The sbatch seeds the work directory with c006's
300 generator and verifier rows; stage_generate and stage_verify skip what is already
recorded, so they run only rows 301-832 of the same manifest. Those reused rows keep
their own schema_version and condition_id, which is why conditions/c007/condition.yaml
registers two record generations.

Extraction stays the grader's job for y1 exactly as it is for y0 and y2. The rewrite
stage therefore records deferred_to_grader and cannot say whether a program came back;
stage_counterfactual resolves that into ok or model_invalid from the grading result,
which is the first place it is visible.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c005_gate0 as base  # noqa: E402

from authority_allocation import derive_call_seed  # noqa: E402
from authority_allocation.api import api_call  # noqa: E402
from authority_allocation.cap_outcomes import build_cap_outcome_target  # noqa: E402
from authority_allocation.prompts import render, split_prompt  # noqa: E402
from authority_allocation.records import (  # noqa: E402
    append_jsonl,
    completed_keys,
    read_jsonl,
)
from authority_allocation.semantics import counterfactual_outcomes  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json,
    stable_id,
    utc_now,
)

REWRITE_RECORD_SCHEMA_VERSION = "rewrite-record-v2"
COUNTERFACTUAL_RECORD_SCHEMA_VERSION = "counterfactual-record-v2"
REWRITE_REPLICATE_ID = "w01"
REWRITE_BASE_SEED = 20260917


def configure_c007() -> None:
    """Point the base runner at c007 without disturbing c005 or c006.

    The replicate plan moves to r005 because Gate 1 needs a rewrite scope and r004
    declares that scope absent. r004 cannot gain one: c006 is a completed run that
    references it. derive_call_seed hashes the plan id, so the 532 new rows draw seeds
    from a different derivation than the 300 reused ones. replicates/r005.yaml records
    why that is unavoidable and why it does not bias anything.
    """

    base.CONDITION_ID = "c007"
    base.CONDITION_PATH = ROOT / "conditions/c007/condition.yaml"
    base.REPLICATE_PLAN_ID = "r005"
    base.STAGE = "gate1"
    base.EXPECTED_ROWS = 832
    base.G_RECORD_SCHEMA_VERSION = "g-record-single-v3"
    base.V_RECORD_SCHEMA_VERSION = "v-record-v3"
    base.PROMPTS["g_rewrite"] = ROOT / "conditions/c007/prompts/g_rewrite_v1.txt"


# Applied at import, not in main(): this module reconfigures shared state in the base
# runner, so importing it and then calling a stage directly must see c007 too.
#
# The mutation is confined to the copy of the base module this file imported. Tests
# reach the base runner as scripts.run_c005_gate0 while this imports it as the
# top-level run_c005_gate0, and Python keeps those as separate module objects, so
# c005's and c006's own entry points are untouched. test_c007_gate1_runner pins that.
configure_c007()


# ------------------------------------------------------------------------- rewrite


def stage_rewrite(args: argparse.Namespace) -> None:
    """Run G against its own program plus the critique, wherever the request is not KEEP.

    The rewriter receives problem, y0 and critique and never the verifier's candidate
    program; that isolation is protocols/authority_v5/README.md section 45 and it lives
    at this call site, not in the prompt.
    """

    output = Path(args.output)
    done = completed_keys(output, ("problem_id", "cell_id"))
    g_by_problem = {row["problem_id"]: row for row in read_jsonl(Path(args.g_records))}
    system, template = split_prompt(base.PROMPTS["g_rewrite"])
    config = base.model_config("g")
    decoding = config["rewrite_decoding"]
    current_condition_hash = base.condition_hash()

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
            status = "not_requested"
            fallback = None
            usage = None
        else:
            report = v_row["v_report"]
            if report is None:
                raise ValueError(
                    f"{problem_id}: non-KEEP registered request without a V report"
                )
            call_seed = derive_call_seed(
                base.REPLICATE_PLAN_ID, "rewrite", REWRITE_REPLICATE_ID,
                REWRITE_BASE_SEED, cell_id, problem_id, 0,
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
                # Deliberately no extraction here. See the module docstring.
                status = "deferred_to_grader"
                fallback = None
                usage = base.compact_usage(call_usage)
            except Exception:
                raw_output = None
                status = "call_error"
                fallback = "KEEP"
                usage = None

        append_jsonl(output, {
            "schema_version": REWRITE_RECORD_SCHEMA_VERSION,
            "record_id": stable_id(args.run_id, cell_id, problem_id),
            "run_id": args.run_id,
            "source_v_run_id": v_row["run_id"],
            "source_v_record_id": v_row["record_id"],
            "condition_id": base.CONDITION_ID,
            "condition_hash": current_condition_hash,
            "dataset_id": base.DATASET_ID,
            "replicate_id": REWRITE_REPLICATE_ID,
            "stage": base.STAGE,
            "cell_id": cell_id,
            "problem_id": problem_id,
            "problem_metadata": g_row.get("problem_metadata", {}),
            "registered_request": registered_request,
            "rewrite_ran": registered_request != "KEEP",
            "call_seed": call_seed,
            "raw_output": raw_output,
            "extracted_answer": None,
            "status": status,
            "failure_fallback_action": fallback,
            "usage": usage,
            "created_at_utc": utc_now(),
        })


# ------------------------------------------------------------------- grade requests


def stage_grade_requests(args: argparse.Namespace) -> None:
    """Emit one grading request per outcome Gate 1 needs: u0, u1 and u2.

    u0 and u2 follow Gate 0 exactly, including its rule that a report which did not
    parse gets no u2 request, because M2 deploys y0 there and grading an absent
    candidate as a wrong program would depress mean(u2) with a parsing failure.

    u1 is requested wherever the rewrite call returned text. A KEEP row has no y1 by
    construction and a call_error produced none, so neither is requested; both surface
    as u1 = null with the M1 branch falling back to u0.
    """

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    v_by_problem = {row["problem_id"]: row for row in read_jsonl(Path(args.v_records))}
    w_by_problem = {row["problem_id"]: row for row in read_jsonl(Path(args.rewrite_records))}

    rows = []
    for g_row in read_jsonl(Path(args.g_records)):
        task_id = g_row["problem_id"]
        rows.append({
            "record_id": stable_id(args.run_id, task_id, "y0"),
            "task_id": task_id,
            "outcome": "u0",
            "text": g_row["outputs"][0]["raw_output"],
        })

        w_row = w_by_problem.get(task_id)
        if w_row is not None and w_row["raw_output"] is not None:
            rows.append({
                "record_id": stable_id(args.run_id, task_id, "y1"),
                "task_id": task_id,
                "outcome": "u1",
                "text": w_row["raw_output"],
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
    counts = collections.Counter(row["outcome"] for row in rows)
    print(canonical_json({
        "requests": len(rows), "by_outcome": dict(sorted(counts.items())),
        "output": str(output),
    }))


# ------------------------------------------------------------------ counterfactuals


def resolve_rewrite_status(record_status: str, u1_grade_status: str | None) -> str:
    """Turn the rewrite record's deferred status into the post-grading vocabulary.

    not_requested and call_error carry over: both are known before grading. A row that
    reached the grader resolves to ok or model_invalid by whether a usable program came
    back. A grader_error is a pipeline fault, not a model outcome, and the protocol
    refuses to validate a run on top of one.
    """

    if record_status in ("not_requested", "call_error"):
        return record_status
    if record_status != "deferred_to_grader":
        raise ValueError(f"unexpected rewrite status {record_status!r}")
    if u1_grade_status in ("ok", "model_invalid"):
        return u1_grade_status
    raise RuntimeError(
        f"rewrite deferred to the grader but grading returned {u1_grade_status!r}; "
        "a grader error cannot be resolved into a model outcome"
    )


def stage_counterfactual(args: argparse.Namespace) -> None:
    """Join the graded outcomes into (z0, z1, z2) under the protocol's failure policy."""

    output = Path(args.output)
    v_by_problem = {row["problem_id"]: row for row in read_jsonl(Path(args.v_records))}
    w_by_problem = {row["problem_id"]: row for row in read_jsonl(Path(args.rewrite_records))}

    graded: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(Path(args.grade_records)):
        graded[(row["problem_id"], row["outcome"])] = row

    current_condition_hash = base.condition_hash()
    for g_row in read_jsonl(Path(args.g_records)):
        task_id = g_row["problem_id"]
        v_row = v_by_problem[task_id]
        w_row = w_by_problem[task_id]

        u0_row = graded[(task_id, "u0")]
        u1_row = graded.get((task_id, "u1"))
        u2_row = graded.get((task_id, "u2"))

        rewrite_status = resolve_rewrite_status(
            w_row["status"], u1_row["status"] if u1_row is not None else None
        )
        u0 = u0_row["u"]
        u1 = u1_row["u"] if (u1_row is not None and rewrite_status == "ok") else None
        u2 = u2_row["u"] if u2_row is not None else None

        outcomes = counterfactual_outcomes(
            v_row["registered_request"], u0=u0, u1=u1, u2=u2,
            rewrite_status=rewrite_status,
        )
        append_jsonl(output, {
            "schema_version": COUNTERFACTUAL_RECORD_SCHEMA_VERSION,
            "record_id": stable_id(args.run_id, task_id, "cf"),
            "run_id": args.run_id,
            "source_g_run_id": g_row["run_id"],
            "source_v_run_id": v_row["run_id"],
            "source_rewrite_run_id": w_row["run_id"] if w_row["rewrite_ran"] else None,
            # Gate 1 registers no feature contract and runs no evidence stage, so there
            # is no evidence record to point at. Null says that; any other hash would
            # only look like provenance.
            "source_evidence_record_id": None,
            "evidence_sha256": None,
            "condition_id": base.CONDITION_ID,
            "condition_hash": current_condition_hash,
            "stage": base.STAGE,
            "cell_id": v_row["cell_id"],
            "problem_id": task_id,
            "registered_request": v_row["registered_request"],
            "rewrite_status": rewrite_status,
            "u0": u0, "u1": u1, "u2": u2,
            "z0": outcomes["z0"], "z1": outcomes["z1"], "z2": outcomes["z2"],
            "grader_status": {
                "y0": u0_row["status"],
                "y1": u1_row["status"] if u1_row is not None else "not_run",
                "y2": u2_row["status"] if u2_row is not None else "not_run",
            },
            "created_at_utc": utc_now(),
        })


# ------------------------------------------------------------------------- analyze


def stage_analyze(args: argparse.Namespace) -> None:
    """Report the eight (z0, z1, z2) patterns and the two counterfactual margins.

    protocols/authority_v6/README.md section 3 requires all eight patterns and the
    repair/harm counts for z1 - z0 and z2 - z1, and requires sparse difference to be
    reported as non-identifiability rather than tuned away.
    """

    rows = list(read_jsonl(Path(args.counterfactual_records)))
    complete = [r for r in rows if None not in (r["z0"], r["z1"], r["z2"])]

    patterns = collections.Counter(
        f'{r["z0"]}{r["z1"]}{r["z2"]}' for r in complete
    )
    all_patterns = {
        f"{a}{b}{c}": patterns.get(f"{a}{b}{c}", 0)
        for a in (0, 1) for b in (0, 1) for c in (0, 1)
    }

    def margin(lo: str, hi: str) -> dict[str, int]:
        repair = sum(1 for r in complete if r[lo] == 0 and r[hi] == 1)
        harm = sum(1 for r in complete if r[lo] == 1 and r[hi] == 0)
        return {"repair": repair, "harm": harm, "net": repair - harm}

    requests = collections.Counter(r["registered_request"] for r in rows)
    override = [r for r in complete if r["registered_request"] == "OVERRIDE"]
    m1_m2_divergent = sum(1 for r in override if r["z1"] != r["z2"])

    oracle = collections.Counter(
        build_cap_outcome_target(r["z0"], r["z1"], r["z2"])["oracle_class"]
        for r in complete
    )

    summary = {
        "run_id": args.run_id,
        "condition_id": base.CONDITION_ID,
        "stage": base.STAGE,
        "n": len(rows),
        "n_complete": len(complete),
        "z_patterns": all_patterns,
        "z1_minus_z0": margin("z0", "z1"),
        "z2_minus_z1": margin("z1", "z2"),
        "z2_minus_z0": margin("z0", "z2"),
        "request_distribution": dict(sorted(requests.items())),
        "override_rows_complete": len(override),
        "m1_m2_divergent_on_override": m1_m2_divergent,
        "oracle_class_distribution": dict(sorted(oracle.items())),
        "rewrite_status_distribution": dict(sorted(
            collections.Counter(r["rewrite_status"] for r in rows).items()
        )),
        "identifiability_note": (
            "M1 and M2 can only differ on OVERRIDE rows. If "
            "m1_m2_divergent_on_override is small, that is the Gate 1 result and must "
            "be reported as non-identifiability, not retuned away."
        ),
        "created_at_utc": utc_now(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")

    lines = [
        f"# Gate 1 ({base.CONDITION_ID})",
        "",
        f"- rows: {summary['n']} ({summary['n_complete']} fully graded)",
        f"- requests: {summary['request_distribution']}",
        "",
        "## (z0, z1, z2) patterns",
        "",
        "|pattern|count|",
        "|---|---:|",
        *(f"|{name}|{count}|" for name, count in sorted(all_patterns.items())),
        "",
        "## counterfactual margins",
        "",
        "|margin|repair|harm|net|",
        "|---|---:|---:|---:|",
        *(
            f"|{label}|{m['repair']}|{m['harm']}|{m['net']}|"
            for label, m in (
                ("z1 - z0", summary["z1_minus_z0"]),
                ("z2 - z1", summary["z2_minus_z1"]),
                ("z2 - z0", summary["z2_minus_z0"]),
            )
        ),
        "",
        "## identifiability",
        "",
        f"- OVERRIDE rows fully graded: {summary['override_rows_complete']}",
        f"- of those, M1 and M2 disagree on: {summary['m1_m2_divergent_on_override']}",
        "",
        summary["identifiability_note"],
        "",
        f"- rewrite status: {summary['rewrite_status_distribution']}",
        f"- oracle classes: {summary['oracle_class_distribution']}",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(canonical_json(summary))


# ---------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Declare Gate 1's stages explicitly rather than reaching into the base parser.

    The shared stages delegate to the base functions, which is where the reuse belongs;
    the argument declarations are repeated because argparse exposes no supported way to
    add a subcommand to a parser someone else built, and a private-attribute reach would
    break silently on an argparse change.
    """

    parser = argparse.ArgumentParser(description="Gate 1 for the code domain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--role", choices=("g", "v"), required=True)
    base.add_api_arguments(smoke)
    smoke.set_defaults(function=base.stage_smoke)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--output", required=True)
    base.add_api_arguments(generate)
    generate.set_defaults(function=base.stage_generate)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--g-records", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--output", required=True)
    base.add_api_arguments(verify)
    verify.set_defaults(function=base.stage_verify)

    grade = subparsers.add_parser("grade")
    grade.add_argument("--requests", required=True)
    grade.add_argument("--results", required=True)
    grade.add_argument("--output", required=True)
    grade.add_argument("--run-id", required=True)
    grade.set_defaults(function=base.stage_grade)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--bundle-dir", required=True)
    finalize.add_argument("--dataset-manifest", required=True)
    finalize.add_argument("--started-at-utc", required=True)
    finalize.add_argument("--source-run-id", action="append")
    finalize.set_defaults(function=base.stage_finalize)

    rewrite = subparsers.add_parser("rewrite")
    base.add_api_arguments(rewrite)
    rewrite.add_argument("--g-records", required=True)
    rewrite.add_argument("--v-records", required=True)
    rewrite.add_argument("--output", required=True)
    rewrite.add_argument("--run-id", required=True)
    rewrite.set_defaults(function=stage_rewrite)

    requests = subparsers.add_parser("grade-requests-gate1")
    requests.add_argument("--g-records", required=True)
    requests.add_argument("--v-records", required=True)
    requests.add_argument("--rewrite-records", required=True)
    requests.add_argument("--output", required=True)
    requests.add_argument("--run-id", required=True)
    requests.set_defaults(function=stage_grade_requests)

    counterfactual = subparsers.add_parser("counterfactual")
    counterfactual.add_argument("--g-records", required=True)
    counterfactual.add_argument("--v-records", required=True)
    counterfactual.add_argument("--rewrite-records", required=True)
    counterfactual.add_argument("--grade-records", required=True)
    counterfactual.add_argument("--output", required=True)
    counterfactual.add_argument("--run-id", required=True)
    counterfactual.set_defaults(function=stage_counterfactual)

    analyze = subparsers.add_parser("analyze-gate1")
    analyze.add_argument("--counterfactual-records", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--run-id", required=True)
    analyze.set_defaults(function=stage_analyze)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
