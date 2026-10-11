"""Build the frozen dev-order and sealed test manifests for MMLU-Pro.

Written to run once, before any model has seen any of these questions. The design is
registered in docs/plan/mmlu_pro_study.md section 3.

Source is TIGER-Lab/MMLU-Pro at a pinned revision. Only its 12,032-row test split is used;
the 70 validation rows are the official few-shot exemplars. Exclusions are decided from the
source data alone, never from any model output:

    duplicate_id                an id that occurs more than once; every copy is dropped
    duplicate_question          identical question text and options on several rows; every
                                copy is dropped, so one question never sits on both sides
    empty_question              no question text
    options_not_strings         options are not a list of non-empty strings
    option_count_out_of_range   fewer than 2 or more than 10 options
    answer_not_an_option        the reference letter is not one of this row's option letters,
                                or disagrees with answer_index

Split: the sealed test takes 3,000 rows, the development pool the rest. Strata are
(category, source origin), the origin being the src prefix (ori_mmlu, stemez, theoremQA,
scibench). Test quotas are proportional to stratum size by largest remainder. Rows are sorted
by id and each stratum, in sorted key order, is permuted once by one PCG64 stream; its first
quota rows go to test. The development pool and the test are then each permuted once more:
development rows are consumed as prefixes of dev_order (dev1000 first), and a smaller
confirmatory test, if one is chosen, is a prefix of test_order fixed before any test row runs.

Manifests are id-only; content is joined from the pinned revision at run time.
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

from authority_allocation import mmlu_pro  # noqa: E402
from authority_allocation.mmmu_pro import allocate_largest_remainder  # noqa: E402

REPO = "TIGER-Lab/MMLU-Pro"
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
EXPECTED_ROWS = 12032
TEST_ROWS = 3000
DEV_PREFIXES = (1000,)
SPLIT_SEED = 20260927
DEV_ORDER_SEED = 20260928
TEST_ORDER_SEED = 20260929


def write_manifest(path: Path, rows: list[dict]) -> dict:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"{path.name:<16} rows={len(rows):<6} sha256={digest}")
    return {"rows": len(rows), "sha256": digest,
            "counts_by_category": dict(sorted(collections.Counter(r["category"] for r in rows).items())),
            "counts_by_origin": dict(sorted(collections.Counter(r["origin"] for r in rows).items())),
            "counts_by_n_options": dict(sorted(collections.Counter(str(r["n_options"]) for r in rows).items()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; manifests are written once")

    source = load_dataset(REPO, revision=REVISION, split="test")
    if len(source) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, found {len(source)}")

    rows = list(source)
    id_count = collections.Counter(str(r["question_id"]) for r in rows)
    key_count = collections.Counter(
        mmlu_pro.question_key(r["question"] or "", r["options"]) for r in rows
        if isinstance(r["options"], list) and all(isinstance(o, str) for o in r["options"]))
    excluded: dict[str, list[str]] = collections.defaultdict(list)
    kept: list[dict] = []
    for row in rows:
        pid = str(row["question_id"])
        options = row["options"]
        if id_count[pid] > 1:
            excluded["duplicate_id"].append(pid)
            continue
        if not (row["question"] or "").strip():
            excluded["empty_question"].append(pid)
            continue
        if not isinstance(options, list) or not all(isinstance(o, str) and o.strip() for o in options):
            excluded["options_not_strings"].append(pid)
            continue
        if key_count[mmlu_pro.question_key(row["question"], options)] > 1:
            excluded["duplicate_question"].append(pid)
            continue
        if not 2 <= len(options) <= len(mmlu_pro.OPTION_LETTERS):
            excluded["option_count_out_of_range"].append(pid)
            continue
        answer = str(row["answer"]).strip()
        if (not mmlu_pro.answer_is_option(answer, len(options))
                or mmlu_pro.OPTION_LETTERS.index(answer) != int(row["answer_index"])):
            excluded["answer_not_an_option"].append(pid)
            continue
        kept.append(mmlu_pro.manifest_row(row))

    kept.sort(key=lambda r: int(r["problem_id"]))
    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in kept:
        strata[(row["category"], row["origin"])].append(row)
    quotas = allocate_largest_remainder({k: len(v) for k, v in strata.items()}, TEST_ROWS)
    rng = np.random.Generator(np.random.PCG64(SPLIT_SEED))
    test, development = [], []
    for key in sorted(strata):
        members = strata[key]
        order = rng.permutation(len(members))
        chosen = [members[i] for i in order]
        test.extend(chosen[:quotas[key]])
        development.extend(chosen[quotas[key]:])
    if {r["problem_id"] for r in test} & {r["problem_id"] for r in development} or len(test) != TEST_ROWS:
        raise RuntimeError("split is inconsistent")
    ordered = {}
    for name, members, seed in (("development", development, DEV_ORDER_SEED), ("test", test, TEST_ORDER_SEED)):
        members = sorted(members, key=lambda r: int(r["problem_id"]))
        order = np.random.Generator(np.random.PCG64(seed)).permutation(len(members))
        ordered[name] = [members[i] for i in order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": {"repo": REPO, "revision": REVISION, "split": "test", "rows": len(rows)},
        "numpy_version": np.__version__,
        "seeds": {"split": SPLIT_SEED, "dev_order": DEV_ORDER_SEED, "test_order": TEST_ORDER_SEED},
        "excluded": {k: sorted(v, key=int) for k, v in excluded.items()},
        "excluded_counts": {k: len(v) for k, v in excluded.items()},
        "rows_kept": len(kept),
        "strata": len(strata),
        "test_quota_by_stratum": {f"{c}|{o}": q for (c, o), q in sorted(quotas.items())},
        "manifests": {},
    }
    report["manifests"]["test"] = write_manifest(args.output_dir / "test.jsonl", ordered["test"])
    report["manifests"]["dev_order"] = write_manifest(args.output_dir / "dev_order.jsonl",
                                                      ordered["development"])
    for size in DEV_PREFIXES:
        report["manifests"][f"dev{size}"] = write_manifest(args.output_dir / f"dev{size}.jsonl",
                                                           ordered["development"][:size])
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("excluded:", report["excluded_counts"], "kept:", len(kept), "strata:", len(strata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
