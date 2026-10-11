"""Build the frozen manifest for the OlympiadBench one-pair extension of the math study.

Written to run once, before any model has seen any of these problems
(docs/plan/olympiadbench_extension.md).

Source is Hothan/OlympiadBench at a pinned revision, configuration OE_TO_maths_en_COMP: the
English, text-only, open-ended competition maths problems (674 rows). The pair run on it is
fixed in advance and nothing is chosen on it, so there is no development/test split: one
manifest, "eval", holds every included row.

Exclusions are decided from the source data alone, never from any model output:

    multiple_answer         is_multiple_answer is true; the frozen prompts ask for one boxed answer
    has_unit                the reference answer carries a unit, which the grader does not compare
    source_error_flag       the source marks the row with an error note
    duplicate_id            an id that occurs more than once; every copy is dropped
    duplicate_problem_text  exact question text that occurs more than once; every copy is dropped
    empty_answer            no reference answer
    gold_unparseable        the registered grader cannot parse the reference answer
                            (run_c002_dev30.math_parse_gold), so no output could be graded
    gold_nonfinite          it parses to a non-finite value (zoo, nan), against which every
                            comparison raises (the mathfull ungradeable-gold rule, applied up front)

The rows are sorted by id and put in one fixed permutation, used only for batching. The
manifest is id-only; content is joined from the pinned revision at run time.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c002_dev30 as base  # noqa: E402
import run_mathfull_solo as solo  # noqa: E402

EXPECTED_SOURCE_ROWS = 674
ORDER_SEED = 20261001


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def gold_status(answer: str) -> str:
    try:
        parsed, _ = base.math_parse_gold(answer)
    except Exception:
        return "gold_unparseable"
    if not parsed:
        return "gold_unparseable"
    if str(parsed[0]) in {"zoo", "nan"}:
        return "gold_nonfinite"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; manifests are written once")

    source = load_dataset(solo.OLYMPIAD_REPO, solo.OLYMPIAD_CONFIG, revision=solo.OLYMPIAD_REVISION,
                          split="train")
    if len(source) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"expected {EXPECTED_SOURCE_ROWS} source rows, got {len(source)}")
    id_count = collections.Counter(str(i) for i in source["id"])
    text_count = collections.Counter(text_hash(q) for q in source["question"])
    excluded: dict[str, list[str]] = collections.defaultdict(list)
    rows = []
    for row in source:
        pid = f"{solo.OLYMPIAD_ID_PREFIX}{row['id']}"
        if row["modality"] != "Text-only" or row["language"] != "English" or row["question_type"] != "Open-ended":
            raise RuntimeError(f"{pid} is not an English text-only open-ended problem")
        if row["is_multiple_answer"]:
            excluded["multiple_answer"].append(pid)
            continue
        if (row["unit"] or "").strip():
            excluded["has_unit"].append(pid)
            continue
        if (row["error"] or "").strip():
            excluded["source_error_flag"].append(pid)
            continue
        if id_count[str(row["id"])] > 1:
            excluded["duplicate_id"].append(pid)
            continue
        if text_count[text_hash(row["question"])] > 1:
            excluded["duplicate_problem_text"].append(pid)
            continue
        if len(row["final_answer"] or []) != 1 or not solo.olympiad_gold(row["final_answer"]):
            excluded["empty_answer"].append(pid)
            continue
        status = gold_status(solo.olympiad_gold(row["final_answer"]))
        if status != "ok":
            excluded[status].append(pid)
            continue
        rows.append({"problem_id": pid, "subject": row["subfield"], "answer_type": row["answer_type"]})

    rows.sort(key=lambda r: int(r["problem_id"][len(solo.OLYMPIAD_ID_PREFIX):]))
    order = np.random.Generator(np.random.PCG64(ORDER_SEED)).permutation(len(rows))
    ordered = [rows[i] for i in order]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval.jsonl").write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    report = {"source": {"repo": solo.OLYMPIAD_REPO, "config": solo.OLYMPIAD_CONFIG,
                         "revision": solo.OLYMPIAD_REVISION, "rows": len(source)},
              "numpy_version": np.__version__, "order_seed": ORDER_SEED,
              "excluded": {k: sorted(v) for k, v in excluded.items()},
              "excluded_counts": {k: len(v) for k, v in excluded.items()},
              "manifests": {"eval": {
                  "rows": len(ordered), "sha256": digest,
                  "counts_by_subject": dict(sorted(collections.Counter(r["subject"] for r in ordered).items())),
                  "counts_by_answer_type": dict(sorted(collections.Counter(r["answer_type"] for r in ordered).items()))}}}
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("excluded_counts", "manifests")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
