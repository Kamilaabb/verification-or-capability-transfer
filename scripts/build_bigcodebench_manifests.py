"""Partition BigCodeBench into frozen development and test manifests.

Written to run once. The split is drawn before any model has seen any task, which
is the whole point: a test set carved out afterwards from tasks already run is not
independent, and the project's dataset policy requires an independent test after any
tuning. Doing it now costs nothing and cannot be done later.

Manifests are id-only, like the math ones — content is joined from the pinned dataset
revision at run time, so the manifest cannot drift from the data.

Nesting is deliberate: the Gate 0 manifest is a prefix of the development manifest in
the same shuffled order, so Gate 1 extends the same run rather than starting over, and
the Gate 0 records stay reusable.

Because every manifest is a prefix slice of one fixed permutation, moving the
development/test boundary only moves which tasks fall either side of it — it does not
reshuffle anything. The boundary moved from 732/400 to 832/300 on 2026-09-17, before
any model had seen any task, so gate0 and the first 732 development rows are bit
identical to the first draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATASET_REPO = "bigcode/bigcodebench"
DATASET_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
# Split v0.1.4 was verified to hold exactly the id set the container's
# get_bigcodebench(subset="full") returns, so the manifest and the grader cannot
# disagree about which tasks exist.
SPLIT = "v0.1.4"
EXPECTED_ROWS = 1140

# Measured by the ground-truth check (jobs 65076891 / 65081056), not assumed: four
# tasks whose own canonical solution fails, four that pass only with network access.
EXCLUDED_TASKS = (
    "BigCodeBench/205", "BigCodeBench/363", "BigCodeBench/590", "BigCodeBench/1005",
    "BigCodeBench/101", "BigCodeBench/176", "BigCodeBench/314", "BigCodeBench/1012",
)

SEED = 20260917
GATE0_ROWS = 300
DEVELOPMENT_ROWS = 832
TEST_ROWS = 300


def task_number(task_id: str) -> int:
    return int(task_id.split("/")[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default=SPLIT)
    args = parser.parse_args()

    dataset = load_dataset(DATASET_REPO, revision=DATASET_REVISION, split=args.split)
    if len(dataset) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {len(dataset)}")
    excluded = set(EXCLUDED_TASKS)

    rows = {}
    for row in dataset:
        task_id = str(row["task_id"])
        if task_id in excluded:
            continue
        rows[task_id] = {"problem_id": task_id, "libs": row.get("libs")}

    missing = excluded - {str(row["task_id"]) for row in dataset}
    if missing:
        raise RuntimeError(f"excluded tasks absent from the revision: {sorted(missing)}")

    usable = sorted(rows, key=task_number)
    print("total rows      :", len(dataset))
    print("excluded        :", len(excluded))
    print("usable          :", len(usable))
    if len(usable) < DEVELOPMENT_ROWS + TEST_ROWS:
        raise RuntimeError(f"only {len(usable)} usable tasks; split needs "
                           f"{DEVELOPMENT_ROWS + TEST_ROWS}")

    order = np.random.Generator(np.random.PCG64(SEED)).permutation(len(usable))
    shuffled = [usable[index] for index in order]

    development = shuffled[:DEVELOPMENT_ROWS]
    test = shuffled[DEVELOPMENT_ROWS:DEVELOPMENT_ROWS + TEST_ROWS]
    reserve = shuffled[DEVELOPMENT_ROWS + TEST_ROWS:]
    gate0 = development[:GATE0_ROWS]

    if set(development) & set(test):
        raise RuntimeError("development and test overlap")
    if not set(gate0) <= set(development):
        raise RuntimeError("gate0 is not nested inside development")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, members in (
        ("gate0", gate0),
        ("development", development),
        ("test", test),
        ("reserve", reserve),
    ):
        path = args.output_dir / f"{name}.jsonl"
        payload = "".join(
            json.dumps(rows[task_id], sort_keys=True) + "\n" for task_id in members
        )
        path.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        print(f"{name:<12} rows={len(members):<5} sha256={digest}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
