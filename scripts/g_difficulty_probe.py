#!/usr/bin/env python3
"""Measure acc(G) under both BigCodeBench prompt fields, paired on identical tasks.

Gate 1 found M1 and M2 indistinguishable on 8 of 195 OVERRIDE rows. The diagnosis in
docs/analysis/gate1_c007.md section 5 is that acc_G = 0.3678 sits below the 0.40-0.55
band c003 registered in advance, and that the both-fail cell grew from 30.6% to 68.7%
as a result. One of the two levers for moving acc_G is the prompt field: the dataset
registers instruct_prompt, whose stated rationale was that it is harder.

That rationale is now measurably weak. The OVERRIDE denominator barely moves with
acc_G — projected 195 rows at 0.368 against 173 at 0.55 — because this verifier
over-corrects, requesting OVERRIDE on 14.4% of rows the generator already got right.
So the cost the harder field was bought for is small, and what it costs in return is
the discriminating cell.

This probe answers only the cheap precondition: does complete_prompt land acc_G inside
the band. If it does not, the dataset lever is dead and the question becomes which
generator to swap in, without spending five hours on a verifier pilot to find out.

Paired on purpose. Both arms generate for the same task ids in the same order, so the
comparison carries no sampling difference at all, and the per-task disagreement is
directly readable. Engineering, not a research run: nothing is registered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.api import api_call  # noqa: E402
from authority_allocation.prompts import render, split_prompt  # noqa: E402
from authority_allocation.serialization import canonical_json  # noqa: E402

PROMPT = ROOT / "conditions/c005/prompts/g_initial_v1.txt"
DATASET_REPO = "bigcode/bigcodebench"
DATASET_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
DATASET_SPLIT = "v0.1.4"

FIELDS = ("instruct_prompt", "complete_prompt")

# c006's registered generator decoding, unchanged: the probe must not also be measuring
# a decoding difference.
DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": None,
    "max_new_tokens": 3072,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def load_rows(task_ids: list[str]) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset

    wanted = set(task_ids)
    indexed: dict[str, dict[str, Any]] = {}
    dataset = load_dataset(DATASET_REPO, revision=DATASET_REVISION, split=DATASET_SPLIT)
    for row in dataset:
        task_id = str(row["task_id"])
        if task_id in wanted:
            indexed[task_id] = {
                "instruct_prompt": row["instruct_prompt"],
                "complete_prompt": row["complete_prompt"],
                "entry_point": row["entry_point"],
            }
    missing = sorted(wanted - set(indexed))
    if missing:
        raise RuntimeError(f"manifest ids absent from the pinned revision: {missing[:5]}")
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument(
        "--fields", default=",".join(FIELDS),
        help="prompt fields to run, comma separated; one field screens a candidate "
             "generator, two compare presentations of the same task",
    )
    parser.add_argument("--requests-out", required=True)
    parser.add_argument("--generations-out", required=True)
    args = parser.parse_args()

    task_ids = [
        json.loads(line)["problem_id"]
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.tasks]
    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise SystemExit(f"unknown prompt fields: {sorted(unknown)}")
    source = load_rows(task_ids)
    system, template = split_prompt(PROMPT)

    # Checked after the FIRST generation, not after all of them. The first version of
    # this guard ran at the end and cost 52 minutes of H100 generating 200 outputs that
    # were thrown away: deepseek-coder-6.7b-instruct returns byte-level BPE tokens under
    # the frozen container, with or without --tokenizer-mode slow. A broken serving path
    # is visible in one call.
    markers = ("Ġ", "Ċ")

    def refuse_if_undecoded(text: str, task_id: str) -> None:
        if any(m in text for m in markers):
            raise SystemExit(
                f"{task_id}: the generation carries byte-level BPE markers, so the "
                "server is not detokenizing. Stopping before spending a GPU hour on "
                "output the grader would score model_invalid and the summary would "
                "report as a capability measurement."
            )

    # Written as they are produced, not collected and dumped at the end. OpenCoder-8B
    # runs at roughly 54 seconds per task against the incumbent's 10, because it
    # generates to nearly the 3072-token cap; at 200 tasks that overruns a two-hour job
    # and a buffered writer would have lost all of it. Partial output is a result.
    requests_file = Path(args.requests_out).open("w", encoding="utf-8", newline="\n")
    generations_file = Path(args.generations_out).open("w", encoding="utf-8", newline="\n")

    requests = []
    generations = []
    for field in fields:
        for task_id in task_ids:
            content, _, usage = api_call(
                api_base=args.api_base,
                served_model=args.served_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": render(template, problem=source[task_id][field])},
                ],
                decoding=DECODING,
                seed=0,
                thinking=False,
            )
            refuse_if_undecoded(content, task_id)
            record_id = f"{field}::{task_id}"
            request_row = {
                "record_id": record_id, "task_id": task_id,
                "outcome": "u0", "text": content,
            }
            generation_row = {
                "record_id": record_id, "field": field, "task_id": task_id,
                "output_tokens": usage.get("output_tokens"),
                "finish_reason": usage.get("finish_reason"),
            }
            requests.append(request_row)
            generations.append(generation_row)
            requests_file.write(canonical_json(request_row) + "\n")
            requests_file.flush()
            generations_file.write(canonical_json(generation_row) + "\n")
            generations_file.flush()

    requests_file.close()
    generations_file.close()
    capped = sum(1 for g in generations if g["finish_reason"] == "length")
    print(canonical_json({
        "tasks": len(task_ids), "arms": list(fields), "requests": len(requests),
        "length_finishes": capped,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
