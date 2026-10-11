#!/usr/bin/env python3
"""Run only the y1 rewrite branch on c011's validated 300-row Gate-0 archive."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c007_gate1 as gate1  # noqa: E402

from authority_allocation.records import read_jsonl  # noqa: E402
from authority_allocation.serialization import canonical_json, stable_id  # noqa: E402

CONDITION_ID = "c013"
DATASET_ID = "bigcodebench-b74c0d0-complete-s20260917"
PROBLEM_FIELD = "complete_prompt"
REPLICATE_PLAN_ID = "r005"
STAGE = "gate1_300_diagnostic"
EXPECTED_ROWS = 300


def configure_c013() -> None:
    """Configure the delegated c007 stages inside this CLI process.

    This is deliberately called from main rather than at import time: run_c007 and this
    runner share their low-level base module, and import-time mutation would make the
    repository's unit tests order-dependent.
    """
    gate1.base.CONDITION_ID = CONDITION_ID
    gate1.base.CONDITION_PATH = ROOT / "conditions/c013/condition.yaml"
    gate1.base.DATASET_ID = DATASET_ID
    gate1.base.PROBLEM_FIELD = PROBLEM_FIELD
    gate1.base.REPLICATE_PLAN_ID = REPLICATE_PLAN_ID
    gate1.base.STAGE = STAGE
    gate1.base.EXPECTED_ROWS = EXPECTED_ROWS
    gate1.base.G_RECORD_SCHEMA_VERSION = "g-record-single-v3"
    gate1.base.V_RECORD_SCHEMA_VERSION = "v-record-v3"
    gate1.base.PROMPTS["g_rewrite"] = ROOT / "conditions/c007/prompts/g_rewrite_v1.txt"

def stage_grade_requests_y1(args: argparse.Namespace) -> None:
    rows = []
    for rewrite in read_jsonl(Path(args.rewrite_records)):
        if rewrite["raw_output"] is None:
            continue
        task_id = rewrite["problem_id"]
        rows.append({
            "record_id": stable_id(args.run_id, task_id, "y1"),
            "task_id": task_id,
            "outcome": "u1",
            "text": rewrite["raw_output"],
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(canonical_json({"requests": len(rows), "by_outcome": {"u1": len(rows)}}))


def stage_merge_grades(args: argparse.Namespace) -> None:
    merged = []
    seen: set[tuple[str, str]] = set()
    counts: collections.Counter[str] = collections.Counter()
    for source in args.input:
        for row in read_jsonl(Path(source)):
            key = (row["problem_id"], row["outcome"])
            if key in seen:
                raise ValueError(f"duplicate grade for {key}")
            seen.add(key)
            counts[row["outcome"]] += 1
            merged.append(row)
    Path(args.output).write_text(
        "".join(canonical_json(row) + "\n" for row in merged), encoding="utf-8"
    )
    print(canonical_json({"rows": len(merged), "by_outcome": dict(sorted(counts.items()))}))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "grade-requests-y1":
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--rewrite-records", required=True)
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--output", required=True)
        stage_grade_requests_y1(parser.parse_args())
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "merge-grades":
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--input", action="append", required=True)
        parser.add_argument("--output", required=True)
        stage_merge_grades(parser.parse_args())
        return 0
    configure_c013()
    return gate1.main()


if __name__ == "__main__":
    raise SystemExit(main())
