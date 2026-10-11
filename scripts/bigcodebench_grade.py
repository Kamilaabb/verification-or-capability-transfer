"""Grade candidate programs against BigCodeBench tests.

The code-domain analogue of the math grader. `grade_prediction(gold, text)` compares
two strings and is pure; correctness here is defined by execution, so this extracts a
program from the model's text and runs the task's own test suite against it.

Batched on purpose. One container start and one dataset load serve a whole grading
stage; a per-call container would spend more time starting than grading. Runs inside
the pinned evaluate image — see `slurm/` for the invocation, which MUST pass
`--net --network=none`, since four tasks otherwise pass only by reaching the network.

Status vocabulary matches the math grader, so the contract does not change:

    ok            graded; u is 0 or 1
    model_invalid no program could be extracted; u is 0, as in the math domain
    grader_error  the grader itself failed; u is null and analysis rejects the run

The raw harness verdict is kept alongside in `detail` so the failure mix stays
auditable without widening the contract.

A timeout is recorded as u=0 with detail "timeout". That is a judgement, not a
measurement: a program that does not finish inside the frozen budget is unusable, but
three tasks (461, 1040, 1104) legitimately need over 60s, so a correct-but-slow
program on those is indistinguishable from a hang. Watch the timeout rate; if it is
material, the limits need revisiting rather than the label.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bigcodebench.data import get_bigcodebench
from bigcodebench.eval import untrusted_check
from bigcodebench.sanitize import sanitize, syntax_check

# Frozen with the dataset, not tuned per run: a pass rate that moves with machine
# load is not a measurement. Identical to the ground-truth check.
MAX_AS_LIMIT = 30720
MAX_DATA_LIMIT = 30720
MAX_STACK_LIMIT = 10
MIN_TIME_LIMIT = 5.0
GT_TIME_LIMIT = 60.0

# Measured by the ground-truth check (jobs 65076891 / 65081056): four tasks whose own
# canonical solution fails, four that pass only with network access. Grading one of
# these means the registered dataset is wrong, so it is an error rather than a skip.
EXCLUDED_TASKS = frozenset({
    "BigCodeBench/205", "BigCodeBench/363", "BigCodeBench/590", "BigCodeBench/1005",
    "BigCodeBench/101", "BigCodeBench/176", "BigCodeBench/314", "BigCodeBench/1012",
})


def defines_entry_point(program: str, entry_point: str) -> bool:
    """True when the program actually defines the function the tests will call."""

    try:
        tree = ast.parse(program)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in ast.walk(tree)
    )


def extract_program(text: str | None, entry_point: str) -> str | None:
    """Return a syntactically valid program defining entry_point, or None.

    Syntactic validity alone is not enough. tree-sitter parses tolerantly, so a
    malformed answer can yield a fragment that compiles but contains no function at
    all — the imports off the top of a one-line blob, say. Accepting that would grade
    "produced no program" as "produced a program that failed", which moves items out of
    model_invalid and into the wrong-answer count.
    """

    if not text or not text.strip():
        return None
    try:
        program = sanitize(text, entrypoint=entry_point)
    except Exception:
        return None
    if not program or not program.strip():
        return None
    if not syntax_check(program):
        return None
    if not defines_entry_point(program, entry_point):
        return None
    return program


def grade_program(task: dict[str, Any], text: str | None) -> dict[str, Any]:
    """Grade one candidate. Mirrors grade_prediction's (u, status) contract."""

    program = extract_program(text, task["entry_point"])
    if program is None:
        return {"u": 0, "status": "model_invalid", "detail": "no_program", "seconds": 0.0}

    started = time.time()
    try:
        harness_status, details = untrusted_check(
            program,
            task["test"],
            task["entry_point"],
            max_as_limit=MAX_AS_LIMIT,
            max_data_limit=MAX_DATA_LIMIT,
            max_stack_limit=MAX_STACK_LIMIT,
            min_time_limit=MIN_TIME_LIMIT,
            gt_time_limit=GT_TIME_LIMIT,
        )
    except Exception as exc:
        return {
            "u": None,
            "status": "grader_error",
            "detail": "driver_error:" + type(exc).__name__,
            "seconds": round(time.time() - started, 2),
        }
    seconds = round(time.time() - started, 2)

    harness_status = str(harness_status)
    if harness_status == "pass":
        passed = details is not None and all(bool(x) for x in details)
        if not passed:
            return {
                "u": None,
                "status": "grader_error",
                "detail": "pass_with_failing_details",
                "seconds": seconds,
            }
        return {"u": 1, "status": "ok", "detail": "pass", "seconds": seconds}
    if harness_status in ("fail", "timeout"):
        return {"u": 0, "status": "ok", "detail": harness_status, "seconds": seconds}
    return {"u": None, "status": "grader_error", "detail": harness_status, "seconds": seconds}


def grade_row(problems: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    task_id = row["task_id"]
    if task_id in EXCLUDED_TASKS:
        raise RuntimeError(f"{task_id} is excluded by the registered dataset; refusing to grade")
    task = problems.get(task_id)
    if task is None:
        raise RuntimeError(f"{task_id} is not in the BigCodeBench task set")
    result = grade_program(task, row.get("text"))
    return {"record_id": row["record_id"], "task_id": task_id, **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True, help="JSONL of record_id, task_id, text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset", default="full")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.requests).open(encoding="utf-8") if line.strip()]
    problems = get_bigcodebench(subset=args.subset)
    print("requests: {0}  workers: {1}".format(len(rows), args.workers), flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = pool.map(lambda row: grade_row(problems, row), rows)
        for index, result in enumerate(futures, start=1):
            results.append(result)
            if index % 50 == 0:
                print("  {0}/{1}".format(index, len(rows)), flush=True)

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")

    graded = [r for r in results if r["status"] == "ok"]
    counts = collections.Counter(r["detail"] for r in results)
    print()
    print("detail counts:", json.dumps(dict(counts), sort_keys=True))
    print("grader_error :", sum(1 for r in results if r["status"] == "grader_error"))
    print("model_invalid:", sum(1 for r in results if r["status"] == "model_invalid"))
    if graded:
        print("u=1: {0}/{1} = {2:.4f}".format(
            sum(r["u"] for r in graded), len(graded), sum(r["u"] for r in graded) / len(graded)))
    print("wrote", output)


if __name__ == "__main__":
    main()
