#!/usr/bin/env python3
"""c020: re-derive c018's evidence and analysis from its archive with the fixed parser path.

c018 computed gv_final_answer_equivalent from G's stored extracted_answer, the sympy
string of an already-parsed answer, which does not re-parse
(docs/analysis/c018_report_format.md). Commit 430d65c compares the parsed texts instead.
c020 reruns only c018's evidence and analysis stages on the byte-identical c017 G and
c018 V records, so no model is called and no gold is read. The compare stage then fails
unless every evidence field other than that one feature equals c018's archive.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c018_math_verifier_scalars as c018  # noqa: E402

CONDITION_ID = "c020"
CONDITION_PATH = ROOT / "conditions/c020/condition.yaml"
CHANGED_FEATURE = "gv_final_answer_equivalent"
# Fields that name the run that wrote a row rather than describe the row.
RUN_FIELDS = {"record_id", "run_id", "condition_id", "condition_hash", "created_at_utc"}


def configure() -> None:
    c018.CONDITION_ID = CONDITION_ID
    c018.CONDITION_PATH = CONDITION_PATH


def unexpected_differences(new: Mapping[str, Any], old: Mapping[str, Any]) -> list[str]:
    """Every difference between two evidence rows outside run identity and the fixed feature."""

    found: list[str] = []
    for key in sorted(set(new) | set(old)):
        if key in RUN_FIELDS:
            continue
        if key in ("features", "missingness"):
            left, right = new.get(key) or {}, old.get(key) or {}
            found += [f"{key}.{name}" for name in sorted(set(left) | set(right))
                      if name != CHANGED_FEATURE and left.get(name) != right.get(name)]
        elif new.get(key) != old.get(key):
            found.append(key)
    return found


def stage_compare(args: argparse.Namespace) -> None:
    new = {row["problem_id"]: row for row in c018.read_jsonl(Path(args.evidence))}
    old = {row["problem_id"]: row for row in c018.read_jsonl(Path(args.c018_evidence))}
    if set(new) != set(old) or len(new) != int(args.expected_rows):
        raise RuntimeError("c020 and c018 evidence do not cover the same rows")
    bad = {pid: diff for pid in new if (diff := unexpected_differences(new[pid], old[pid]))}
    if bad:
        sample = dict(list(bad.items())[:5])
        raise RuntimeError(f"c020 evidence differs from c018 beyond {CHANGED_FEATURE}: {sample}")
    transitions = Counter(
        (str(old[pid]["features"][CHANGED_FEATURE]), str(new[pid]["features"][CHANGED_FEATURE]))
        for pid in new
    )
    report = {
        "condition_id": CONDITION_ID,
        "rows": len(new),
        "changed_feature": CHANGED_FEATURE,
        "every_other_field_identical": True,
        "c018_to_c020": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(c018.canonical_json(report))


def main() -> int:
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command in ("smoke", "verify"):
        raise SystemExit("c020 calls no model; only check-runnable, evidence, analyze and compare run")
    if command == "compare":
        parser = argparse.ArgumentParser(description="compare c020 evidence with c018's archive")
        parser.add_argument("--evidence", required=True)
        parser.add_argument("--c018-evidence", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--expected-rows", type=int, default=c018.EXPECTED_ROWS)
        stage_compare(parser.parse_args(sys.argv[2:]))
        return 0
    return c018.main()


if __name__ == "__main__":
    raise SystemExit(main())
