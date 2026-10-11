"""Build the frozen development-order and sealed test manifests for full MATH, levels 3-5.

Written to run once, before any model has seen any of these problems.

Source is nlile/hendrycks-MATH-benchmark at a pinned revision. Its 12,000-row train split is
the PRM800K split of Hendrycks MATH: the 7,500 original training problems (unique_id
"train/...") plus the 4,500 original test problems that are not in MATH-500 (unique_id
"test/..."); its 500-row test split is exactly MATH-500. The builder verifies the latter
and that no train row overlaps MATH-500, so nothing here touches dev30, pilot300/dev300 or
unused37.

Two pools, split along MATH's own train/test boundary:

    test pool         original MATH test problems not in MATH-500   -> sealed test
    development pool  original MATH training problems              -> development order

Exclusions are decided from the source data alone, never from any model output:

    duplicate_unique_id     an id that occurs more than once; every copy is dropped
    duplicate_problem_text  a problem whose exact text occurs more than once anywhere in
                            the 12,500 problems, MATH-500 included; every copy is dropped,
                            so identical text can never sit on both sides of the split
    empty_answer            no reference answer
    gold_unparseable        the registered grader cannot parse the reference answer
                            (run_c002_dev30.math_parse_gold), so no output could be graded

Development rows are consumed as prefixes of one fixed permutation (dev1000 first, then
2000, then 3000 if needed), so enlarging development never reshuffles or selects rows.
The test order is also a fixed permutation, used only for batching.

Manifests are id-only, like the MATH-500 ones; content is joined from the pinned revision
at run time.
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

REPO = "nlile/hendrycks-MATH-benchmark"
REVISION = "465bcdb36f5962aa3512891498966df785fc3c18"
MATH500_REPO = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
ALLOWED_LEVELS = (3, 4, 5)
DEV_SEED = 20260924
TEST_SEED = 20260925
DEV_PREFIXES = (1000,)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def gold_parses(answer: str) -> bool:
    try:
        parsed, _ = base.math_parse_gold(answer)
    except Exception:
        return False
    return bool(parsed)


def write_manifest(path: Path, rows: list[dict]) -> dict:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    levels = collections.Counter(str(row["level"]) for row in rows)
    print(f"{path.name:<18} rows={len(rows):<5} levels={dict(sorted(levels.items()))} sha256={digest}")
    return {"rows": len(rows), "sha256": digest, "counts_by_level": dict(sorted(levels.items()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; manifests are written once")

    source = load_dataset(REPO, revision=REVISION)
    math500 = load_dataset(MATH500_REPO, revision=MATH500_REVISION, split="test")
    train, heldout = source["train"], source["test"]
    if len(train) != 12000 or len(heldout) != 500 or len(math500) != 500:
        raise RuntimeError("unexpected split sizes")
    m500_ids, m500_text = set(math500["unique_id"]), {text_hash(p) for p in math500["problem"]}
    if set(heldout["unique_id"]) != m500_ids or {text_hash(p) for p in heldout["problem"]} != m500_text:
        raise RuntimeError("the source's test split is not exactly MATH-500")
    if set(train["unique_id"]) & m500_ids:
        raise RuntimeError("train rows share ids with MATH-500")

    id_count = collections.Counter(train["unique_id"])
    text_count = collections.Counter([text_hash(p) for p in train["problem"]]
                                     + [text_hash(p) for p in math500["problem"]])
    excluded: dict[str, list[str]] = collections.defaultdict(list)
    pools: dict[str, list[dict]] = {"test": [], "development": []}
    for row in train:
        pid = str(row["unique_id"])
        level = int(row["level"])
        if level not in ALLOWED_LEVELS:
            continue
        if id_count[pid] > 1:
            excluded["duplicate_unique_id"].append(pid)
            continue
        if text_count[text_hash(row["problem"])] > 1:
            excluded["duplicate_problem_text"].append(pid)
            continue
        answer = (row["answer"] or "").strip()
        if not answer:
            excluded["empty_answer"].append(pid)
            continue
        if not gold_parses(answer):
            excluded["gold_unparseable"].append(pid)
            continue
        pool = "test" if pid.startswith("test/") else "development" if pid.startswith("train/") else None
        if pool is None:
            raise RuntimeError(f"unexpected id prefix: {pid}")
        pools[pool].append({"problem_id": pid, "level": level, "subject": row["subject"]})

    ordered = {}
    for pool, seed in (("development", DEV_SEED), ("test", TEST_SEED)):
        rows = sorted(pools[pool], key=lambda r: r["problem_id"])
        order = np.random.Generator(np.random.PCG64(seed)).permutation(len(rows))
        ordered[pool] = [rows[i] for i in order]
    if {r["problem_id"] for r in ordered["development"]} & {r["problem_id"] for r in ordered["test"]}:
        raise RuntimeError("development and test overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"source": {"repo": REPO, "revision": REVISION},
              "numpy_version": np.__version__,
              "excluded": {k: sorted(v) for k, v in excluded.items()},
              "excluded_counts": {k: len(v) for k, v in excluded.items()},
              "manifests": {}}
    report["manifests"]["test"] = write_manifest(args.output_dir / "test.jsonl", ordered["test"])
    report["manifests"]["dev_order"] = write_manifest(args.output_dir / "dev_order.jsonl",
                                                      ordered["development"])
    for size in DEV_PREFIXES:
        report["manifests"][f"dev{size}"] = write_manifest(
            args.output_dir / f"dev{size}.jsonl", ordered["development"][:size])
    print("excluded:", report["excluded_counts"])
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
