"""Build the frozen manifest for the natural-language-reasoning domain of the GV vs max(G, V)
study (docs/plan/nl_reasoning_study.md).

Written to run once, on the server, before any model has seen any of these problems. The three
benchmarks are the natural-language-reasoning set of Variation in Verification (ICLR 2026,
arXiv 2509.17995): ReClor validation (500), FOLIO v2 validation (203) and GPQA Diamond (198),
901 rows together. They are pooled into one domain and every row is run; nothing is fit, tuned
or chosen on them, so there is no development/test split: one manifest, "eval", holds every
included row.

Sources are the authors' own releases (no Hugging Face token needed), checked by sha256 before
anything is read:

    GPQA     github.com/idavidrein/gpqa dataset.zip (password in the README), gpqa_diamond.csv
    ReClor   github.com/yuweihao/reclor releases/v1 reclor_data.zip (password in the README), val.json
    FOLIO    huggingface.co/datasets/tasksource/folio folio_v2_validation.jsonl at a pinned revision,
             the public copy of the v2 release the FOLIO README recommends

Each row becomes one multiple-choice question for the frozen MMLU-Pro prompts:

    ReClor   the passage, a blank line, the question; its four answers in source order
    FOLIO    the premises and the conclusion, then "true, false, or uncertain?"; options
             True / False / Uncertain in that order
    GPQA     the question; the correct and three incorrect answers in one fixed permutation per row,
             drawn from OPTION_SEED in sorted record-id order (GPQA stores the correct answer first)

Exclusions are decided from the source data alone, never from any model output:

    duplicate_id              an id that occurs more than once; every copy is dropped
    duplicate_question        identical question text and options on several rows; every copy is dropped
    empty_question            no question text, or an empty option
    gold_ambiguous            the correct answer's text equals another option's text
    label_invalid             a reference label outside the row's options

The rows are sorted by problem id and put in one fixed permutation (ORDER_SEED), used only for
batching. The manifest is id-only; the question texts, options and reference letters are written
once to source_rows.jsonl next to it, whose sha256 the dataset registration records and the
runner checks before every read. Neither file is ever committed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ORDER_SEED = 20261001
OPTION_SEED = 20261002
LETTERS = "ABCDEFGHIJ"
FOLIO_OPTIONS = ("True", "False", "Uncertain")
INPUT_SHA256 = {
    "gpqa_diamond_csv": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
    "reclor_val_json": "49a43973d7cf00d5e0d645daad864efca8e10497ced241f32afa4bd260233159",
    "folio_v2_validation_jsonl": "5273a939ad5ffff5b8f1e21b6a81f07b0653be11bead084f40e067269b7b1aaf",
}
EXPECTED_SOURCE_ROWS = {"gpqa_diamond": 198, "reclor": 500, "folio": 203}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folio_question(premises: str, conclusion: str) -> str:
    return (f"Premises:\n{premises.strip()}\n\nConclusion:\n{conclusion.strip()}\n\n"
            "Based only on the premises, is the conclusion true, false, or uncertain?")


def reclor_question(context: str, question: str) -> str:
    return f"{context.strip()}\n\n{question.strip()}"


def gpqa_rows(path: Path) -> list[dict[str, Any]]:
    csv.field_size_limit(10 ** 9)
    with path.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    source.sort(key=lambda r: r["Record ID"])
    rng = np.random.Generator(np.random.PCG64(OPTION_SEED))
    rows = []
    for record in source:
        answers = [record["Correct Answer"].strip()] + [record[f"Incorrect Answer {i}"].strip()
                                                       for i in (1, 2, 3)]
        permutation = [int(i) for i in rng.permutation(4)]
        rows.append({"problem_id": f"gpqa/{record['Record ID']}", "category": "gpqa_diamond",
                     "origin": "gpqa_" + record["High-level domain"].strip().lower(),
                     "question": record["Question"].strip(),
                     "options": [answers[i] for i in permutation],
                     "answer": LETTERS[permutation.index(0)],
                     "correct_text": answers[0], "option_permutation": permutation})
    return rows


def reclor_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for record in json.loads(path.read_text(encoding="utf-8")):
        options = [a.strip() for a in record["answers"]]
        label = record["label"]
        rows.append({"problem_id": f"reclor/{record['id_string']}", "category": "reclor",
                     "origin": "reclor", "question": reclor_question(record["context"], record["question"]),
                     "options": options,
                     "answer": LETTERS[label] if isinstance(label, int) and 0 <= label < len(options) else None,
                     "correct_text": options[label] if isinstance(label, int) and 0 <= label < len(options) else None})
    return rows


def folio_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = str(record["label"]).strip()
        rows.append({"problem_id": f"folio/{record['example_id']}", "category": "folio", "origin": "folio",
                     "question": folio_question(str(record["premises"]), str(record["conclusion"])),
                     "options": list(FOLIO_OPTIONS),
                     "answer": LETTERS[FOLIO_OPTIONS.index(label)] if label in FOLIO_OPTIONS else None,
                     "correct_text": label if label in FOLIO_OPTIONS else None})
    return rows


def exclusions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    excluded: dict[str, list[str]] = collections.defaultdict(list)
    id_count = collections.Counter(r["problem_id"] for r in rows)
    key = lambda r: json.dumps([r["question"], r["options"]], ensure_ascii=False)  # noqa: E731
    question_count = collections.Counter(key(r) for r in rows)
    for row in rows:
        pid = row["problem_id"]
        if id_count[pid] > 1:
            excluded["duplicate_id"].append(pid)
        elif question_count[key(row)] > 1:
            excluded["duplicate_question"].append(pid)
        elif not row["question"] or any(not o for o in row["options"]):
            excluded["empty_question"].append(pid)
        elif row["answer"] is None or row["answer"] not in LETTERS[:len(row["options"])]:
            excluded["label_invalid"].append(pid)
        elif row["options"].count(row["correct_text"]) != 1:
            excluded["gold_ambiguous"].append(pid)
    return {k: sorted(set(v)) for k, v in excluded.items()}


def build(gpqa_csv: Path, reclor_val: Path, folio_val: Path, *,
          input_sha256: dict[str, str] = INPUT_SHA256,
          expected_rows: dict[str, int] = EXPECTED_SOURCE_ROWS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The ordered rows and the build report. The defaults are the registered inputs; tests
    pass their own synthetic files with their hashes and sizes."""
    inputs = {"gpqa_diamond_csv": gpqa_csv, "reclor_val_json": reclor_val,
              "folio_v2_validation_jsonl": folio_val}
    for name, path in inputs.items():
        if sha256_file(path) != input_sha256[name]:
            raise SystemExit(f"{path} does not have the registered sha256 for {name}")
    by_source = {"gpqa_diamond": gpqa_rows(gpqa_csv), "reclor": reclor_rows(reclor_val),
                 "folio": folio_rows(folio_val)}
    for source, rows in by_source.items():
        if len(rows) != expected_rows[source]:
            raise RuntimeError(f"expected {expected_rows[source]} {source} rows, got {len(rows)}")
    rows = [r for source_rows in by_source.values() for r in source_rows]
    excluded = exclusions(rows)
    dropped = {pid for ids in excluded.values() for pid in ids}
    kept = sorted((r for r in rows if r["problem_id"] not in dropped), key=lambda r: r["problem_id"])
    order = np.random.Generator(np.random.PCG64(ORDER_SEED)).permutation(len(kept))
    ordered = [kept[int(i)] for i in order]
    report = {
        "input_sha256": dict(input_sha256), "order_seed": ORDER_SEED, "option_seed": OPTION_SEED,
        "numpy_version": np.__version__, "rows_before_exclusion": len(rows),
        "rows_after_exclusion": len(ordered), "excluded": excluded,
        "counts_by_category": dict(sorted(collections.Counter(r["category"] for r in ordered).items())),
        "counts_by_origin": dict(sorted(collections.Counter(r["origin"] for r in ordered).items())),
        "counts_by_n_options": {str(k): v for k, v in sorted(
            collections.Counter(len(r["options"]) for r in ordered).items())},
        "gold_letter_counts": dict(sorted(collections.Counter(r["answer"] for r in ordered).items())),
    }
    return ordered, report


def manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"problem_id": row["problem_id"], "category": row["category"], "origin": row["origin"],
            "n_options": len(row["options"])}


def source_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {"problem_id": row["problem_id"], "category": row["category"], "origin": row["origin"],
           "question": row["question"], "options": row["options"], "answer": row["answer"]}
    if "option_permutation" in row:
        out["option_permutation"] = row["option_permutation"]
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                    encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpqa-csv", type=Path, required=True)
    parser.add_argument("--reclor-val", type=Path, required=True)
    parser.add_argument("--folio-val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; manifests are written once")
    ordered, report = build(args.gpqa_csv, args.reclor_val, args.folio_val)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "eval.jsonl", [manifest_row(r) for r in ordered])
    write_jsonl(args.output_dir / "source_rows.jsonl", [source_row(r) for r in ordered])
    report["eval_sha256"] = sha256_file(args.output_dir / "eval.jsonl")
    report["source_rows_sha256"] = sha256_file(args.output_dir / "source_rows.jsonl")
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    report["build_report_sha256"] = sha256_file(args.output_dir / "build_report.json")
    print(json.dumps({k: report[k] for k in ("rows_after_exclusion", "excluded", "counts_by_category",
                                             "counts_by_n_options", "gold_letter_counts", "eval_sha256",
                                             "source_rows_sha256", "build_report_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
