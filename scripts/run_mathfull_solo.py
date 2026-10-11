#!/usr/bin/env python3
"""Solo pass 1 on mathfull: G is Qwen3-4B, V is Qwen3-8B, both 4096 tokens.

c022/c023 use c017's solve prompt; c025/c026 use the frozen solve prompt v2
(docs/plan/gv_vs_max_g_v_study.md section 4.7) and differ from c022/c023 only in it.

Usage: run_mathfull_solo.py --condition-id <id> <c017 runner command | check-runnable | reconcile | manifest-info | shard-manifest | merge-shards | analyze> [args]

c017's runner is reused unchanged. This wrapper points it at the condition, the mathfull
dataset identity, the r012 seed plan and the 1,000-row dev1000 manifest, and replaces the
MATH-500 source loader with one for the pinned mathfull source. Two extra commands:

    check-runnable  proves each condition differs from its base only where registered
                    (c022 from c017: dataset, budget, seeds, evidence schema; c023 from
                    c021: dataset, seeds, evidence schema; c025/c026 from c022/c023:
                    prompt, seeds, evidence schema), that the manifest hash in the
                    condition equals the one in the dataset registration, and that a
                    frozen prompt still has its registered hash
    reconcile       makes a resumed run consistent before it continues: keeps only rows
                    present and valid in both pass1.jsonl and native_logit_trace.jsonl,
                    and drops a torn final line left by a job killed mid-write
    analyze         c017's analysis over every row except the problems registered in
                    datasets/<id>/ungradeable_gold.yaml (reference answer the grader
                    cannot compare against); any other grader_error still stops it
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c002_dev30 as base  # noqa: E402
import run_c017_math_expression_scalars as c017  # noqa: E402

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.serialization import canonical_json  # noqa: E402

MATHFULL_ID = "mathfull-r465bcdb-l3to5-s20260924"
SOURCE_REPO = "nlile/hendrycks-MATH-benchmark"
SOURCE_REVISION = "465bcdb36f5962aa3512891498966df785fc3c18"
# The second math dataset, run for one pair only (docs/plan/olympiadbench_extension.md):
# OlympiadBench's English, text-only, open-ended competition maths. Its single manifest is
# neither development nor test: the pair was fixed before any model saw these problems and
# nothing is chosen on it, so every included row is reported.
OLYMPIAD_ID = "olympiadbench-r91184b5-oe-to-maths-en-s20261001"
OLYMPIAD_REPO = "Hothan/OlympiadBench"
OLYMPIAD_CONFIG = "OE_TO_maths_en_COMP"
OLYMPIAD_REVISION = "91184b52131e7fc9455fef848035173aea8cc01a"
OLYMPIAD_ID_PREFIX = "olympiadbench/"
EXTENSION_MANIFESTS = {"eval"}
# The dataset a run uses is the one its condition registers; configure() points these at it.
DATASET_ID = MATHFULL_ID
DATASET_PATH = ROOT / "datasets" / DATASET_ID / "dataset.yaml"
REPLICATE_PLAN_ID = "r012"
BASE_SEED = 20260925
EXPECTED_ROWS = 1000
BUDGET = 4096

CONDITIONS = {
    "c022": {"base": "c017", "role": "G", "model_id": "Qwen/Qwen3-4B"},
    "c023": {"base": "c021", "role": "V", "model_id": "Qwen/Qwen3-8B"},
    "c025": {"base": "c022", "role": "G", "model_id": "Qwen/Qwen3-4B", "frozen_prompt": True},
    "c026": {"base": "c023", "role": "V", "model_id": "Qwen/Qwen3-8B", "frozen_prompt": True},
    "c029": {"base": "c025", "role": "solo", "model_id": "meta-llama/Llama-3.2-3B-Instruct",
             "frozen_prompt": True, "new_model": True},
    "c030": {"base": "c025", "role": "solo", "model_id": "Qwen/Qwen2.5-3B-Instruct",
             "frozen_prompt": True, "new_model": True},
    "c031": {"base": "c025", "role": "solo", "model_id": "Qwen/Qwen2.5-7B-Instruct",
             "frozen_prompt": True, "new_model": True},
    "c032": {"base": "c025", "role": "solo", "model_id": "mistralai/Ministral-8B-Instruct-2410",
             "frozen_prompt": True, "new_model": True},
    "c054": {"base": "c025", "role": "solo", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
             "frozen_prompt": True, "new_model": True},
    "c055": {"base": "c025", "role": "solo", "model_id": "Qwen/Qwen3-32B", "frozen_prompt": True, "new_model": True},
    "c066": {"base": "c025", "role": "solo", "model_id": "Qwen/Qwen2.5-72B-Instruct",
             "frozen_prompt": True, "new_model": True},
    "c067": {"base": "c025", "role": "solo", "model_id": "meta-llama/Llama-3.1-8B-Instruct",
             "frozen_prompt": True, "new_model": True},
    # The confirmatory test runs (docs/plan/test_mathfull_v2.md): c025/c026 on the sealed
    # test manifest, nothing else changed.
    "c040": {"base": "c025", "role": "G", "model_id": "Qwen/Qwen3-4B", "frozen_prompt": True, "test": True},
    "c041": {"base": "c026", "role": "V", "model_id": "Qwen/Qwen3-8B", "frozen_prompt": True, "test": True},
    # The Llama-3.2-3B / Qwen2.5-3B test (docs/plan/test_llama_qwen25.md): c029/c030 on the
    # first 1,500 test rows, nothing else changed.
    "c056": {"base": "c029", "role": "G", "model_id": "meta-llama/Llama-3.2-3B-Instruct",
             "frozen_prompt": True, "test": True},
    "c057": {"base": "c030", "role": "V", "model_id": "Qwen/Qwen2.5-3B-Instruct",
             "frozen_prompt": True, "test": True},
    # The Qwen3-4B / Qwen2.5-7B test (docs/plan/test_qwen3_4b_qwen25_7b.md): c025/c031 on the
    # first 1,500 test rows, nothing else changed.
    "c062": {"base": "c025", "role": "V", "model_id": "Qwen/Qwen3-4B", "frozen_prompt": True, "test": True},
    "c063": {"base": "c031", "role": "G", "model_id": "Qwen/Qwen2.5-7B-Instruct",
             "frozen_prompt": True, "test": True},
    # The Qwen3-8B / Mistral-Small-24B test (docs/plan/test_qwen3_8b_mistral24b.md): c026/c054
    # on the first 1,500 test rows, nothing else changed.
    "c092": {"base": "c026", "role": "G", "model_id": "Qwen/Qwen3-8B", "frozen_prompt": True, "test": True},
    "c093": {"base": "c054", "role": "V", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
             "frozen_prompt": True, "test": True},
    # Solo test runs of the remaining dev1000 models (2026-09-28): the dev
    # condition on the first 1,500 test rows, nothing else changed; no pair registered.
    "c096": {"base": "c032", "role": "solo", "model_id": "mistralai/Ministral-8B-Instruct-2410",
             "frozen_prompt": True, "test": True},
    "c097": {"base": "c055", "role": "solo", "model_id": "Qwen/Qwen3-32B", "frozen_prompt": True, "test": True},
    "c098": {"base": "c066", "role": "solo", "model_id": "Qwen/Qwen2.5-72B-Instruct",
             "frozen_prompt": True, "test": True},
    "c068": {"base": "c067", "role": "solo", "model_id": "meta-llama/Llama-3.1-8B-Instruct",
             "frozen_prompt": True, "test": True},
    # The re-split development set (decided 2026-09-29, docs/plan/testpool_dev1000.md):
    # each dev condition on testpool_dev1000, rows 1501-2500 of the test pool; nothing else changed.
    "c200": {"base": "c025", "role": "solo", "model_id": "Qwen/Qwen3-4B", "frozen_prompt": True},
    "c201": {"base": "c026", "role": "solo", "model_id": "Qwen/Qwen3-8B", "frozen_prompt": True},
    "c202": {"base": "c030", "role": "solo", "model_id": "Qwen/Qwen2.5-3B-Instruct", "frozen_prompt": True},
    "c203": {"base": "c031", "role": "solo", "model_id": "Qwen/Qwen2.5-7B-Instruct", "frozen_prompt": True},
    "c204": {"base": "c032", "role": "solo", "model_id": "mistralai/Ministral-8B-Instruct-2410", "frozen_prompt": True},
    "c205": {"base": "c054", "role": "solo", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501", "frozen_prompt": True},
    "c206": {"base": "c055", "role": "solo", "model_id": "Qwen/Qwen3-32B", "frozen_prompt": True},
    "c207": {"base": "c067", "role": "solo", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "frozen_prompt": True},
    "c208": {"base": "c029", "role": "solo", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "frozen_prompt": True},
    "c209": {"base": "c066", "role": "solo", "model_id": "Qwen/Qwen2.5-72B-Instruct", "frozen_prompt": True},
    # The OlympiadBench pair (docs/plan/olympiadbench_extension.md): the testpool_dev1000 solo
    # condition on the OlympiadBench manifest; nothing else changed.
    "c254": {"base": "c204", "role": "G", "model_id": "mistralai/Ministral-8B-Instruct-2410",
             "frozen_prompt": True, "dataset": OLYMPIAD_ID},
    "c255": {"base": "c201", "role": "V", "model_id": "Qwen/Qwen3-8B", "frozen_prompt": True,
             "dataset": OLYMPIAD_ID},
    # Its second pair, Qwen3-4B / Qwen3-8B (decided 2026-10-01, before any result).
    "c259": {"base": "c200", "role": "G", "model_id": "Qwen/Qwen3-4B", "frozen_prompt": True,
             "dataset": OLYMPIAD_ID},
}
# Registered test manifests: the sealed test and its prefixes (dataset.yaml test_use_log).
TEST_MANIFESTS = {"test", "test1500"}
# A condition that swaps in a new model may change only these model fields; the identity
# ones must equal the provenance file written from one scan of the snapshot.
MODEL_IDENTITY_KEYS = {"model_id", "revision", "tokenizer_revision", "server_path", "provenance",
                       "chat_template_hash", "chat_template_kwargs", "identity_reused_from",
                       "preflight_job", "preflight_reused_from", "preflight_reuse_reason",
                       "preflight_policy"}
PROVENANCE_FIELDS = ("revision", "tokenizer_revision", "server_path", "chat_template_hash")
SOLVE_PROMPT_V2 = "conditions/c025/prompts/g_solve_v2.txt"
REVIEW_PROMPT_V2 = "conditions/c027/prompts/v_reviewer_v2.txt"
# The commit-first review prompt of the prompt-variant sanity check (c390, c391;
# docs/plan/prompt_variant_test500.md); frozen 2026-10-09, used by no other condition.
REVIEW_PROMPT_COMMIT_FIRST = "conditions/c390/prompts/v_commit_first_v1.txt"
# Prompts frozen for the mathfull study, dev and test alike (docs/plan/gv_vs_max_g_v_study.md
# section 4.7). Hashes are over LF line endings, as the repository stores them. Changing a
# frozen prompt starts a new study; it is not a new condition on this dataset.
FROZEN_PROMPTS = {
    SOLVE_PROMPT_V2: "9df8fe147f4e6b3a6e35ed5337b7f7d19d5cf4c6ac1c86ce55b5f6d12f68148d",
    REVIEW_PROMPT_V2: "b523a22310bb041d12d789dfbf44d30e6bc97ec7a3100cc943bd83bb792b7446",
    REVIEW_PROMPT_COMMIT_FIRST: "e6804f3dfb2fb998d2eeec14e93c523be33893cfd3fd3075f2178a9f449c9d70",
}
# Top-level sections a condition may change relative to its base.
ALLOWED_SECTIONS = {"condition_id", "status", "purpose", "gate", "runnable_gate",
                    "comparison", "dataset", "replicates", "model", "protocol"}


def load_mathfull_rows(problem_ids: Iterable[str]) -> dict[str, dict[str, str]]:
    """Join an id-only mathfull manifest to the pinned source revision."""

    from datasets import load_dataset

    wanted = set(problem_ids)
    indexed: dict[str, dict[str, str]] = {}
    for row in load_dataset(SOURCE_REPO, revision=SOURCE_REVISION, split="train"):
        problem_id = str(row["unique_id"])
        if problem_id not in wanted:
            continue
        if problem_id in indexed:
            raise ValueError(f"duplicate source row {problem_id!r}; it should have been excluded")
        problem, gold = row.get("problem"), row.get("answer")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"source row {problem_id!r} has no problem text")
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError(f"source row {problem_id!r} has no reference answer")
        indexed[problem_id] = {"problem": problem, "gold": gold}
    missing = wanted.difference(indexed)
    if missing:
        raise ValueError(f"manifest ids missing from the pinned source: {sorted(missing)[:5]}")
    return indexed


def olympiad_gold(final_answer: list[str]) -> str:
    """The reference answer as the grader receives it: the one final answer, without the
    $...$ delimiters a third of the source rows carry (math_parse_gold adds its own)."""

    if len(final_answer) != 1:
        raise ValueError(f"expected one final answer, got {len(final_answer)}")
    answer = final_answer[0].strip()
    if len(answer) >= 2 and answer.startswith("$") and answer.endswith("$"):
        answer = answer.strip("$").strip()
    return answer


def load_olympiad_rows(problem_ids: Iterable[str]) -> dict[str, dict[str, str]]:
    """Join an id-only OlympiadBench manifest to the pinned source revision."""

    from datasets import load_dataset

    wanted = set(problem_ids)
    indexed: dict[str, dict[str, str]] = {}
    for row in load_dataset(OLYMPIAD_REPO, OLYMPIAD_CONFIG, revision=OLYMPIAD_REVISION, split="train"):
        problem_id = f"{OLYMPIAD_ID_PREFIX}{row['id']}"
        if problem_id not in wanted:
            continue
        if problem_id in indexed:
            raise ValueError(f"duplicate source row {problem_id!r}; it should have been excluded")
        problem, gold = row.get("question"), olympiad_gold(row.get("final_answer") or [])
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"source row {problem_id!r} has no problem text")
        if not gold:
            raise ValueError(f"source row {problem_id!r} has no reference answer")
        indexed[problem_id] = {"problem": problem, "gold": gold}
    missing = wanted.difference(indexed)
    if missing:
        raise ValueError(f"manifest ids missing from the pinned source: {sorted(missing)[:5]}")
    return indexed


LOADERS = {MATHFULL_ID: load_mathfull_rows, OLYMPIAD_ID: load_olympiad_rows}


def configure(condition_id: str) -> None:
    """Point c017's runner at a condition and at the dataset that condition registers."""

    global DATASET_ID, DATASET_PATH, UNGRADEABLE_PATH
    condition_path = ROOT / "conditions" / condition_id / "condition.yaml"
    dataset_id = resolve_condition(condition_path, repository_root=ROOT)["dataset"]["id"]
    if dataset_id not in LOADERS:
        raise SystemExit(f"{condition_id} registers dataset {dataset_id}, which this runner cannot load")
    DATASET_ID = dataset_id
    DATASET_PATH = ROOT / "datasets" / DATASET_ID / "dataset.yaml"
    UNGRADEABLE_PATH = ROOT / "datasets" / DATASET_ID / "ungradeable_gold.yaml"
    c017.CONDITION_ID = condition_id
    c017.CONDITION_PATH = condition_path
    c017.DATASET_ID = DATASET_ID
    c017.REPLICATE_PLAN_ID = REPLICATE_PLAN_ID
    c017.SEED_CELL_ID = condition_id
    c017.EXPECTED_ROWS = EXPECTED_ROWS
    base.load_source_rows = LOADERS[DATASET_ID]


def registration(cond: dict[str, Any]) -> dict[str, Any]:
    """The dataset.yaml of the dataset a resolved condition registers."""

    if cond["dataset"]["id"] not in LOADERS:
        raise SystemExit(f"dataset {cond['dataset']['id']} is not one this runner can load")
    path = ROOT / "datasets" / cond["dataset"]["id"] / "dataset.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def require_frozen_prompt(relative: str) -> None:
    if relative not in FROZEN_PROMPTS:
        raise SystemExit(f"{relative} is not a frozen mathfull prompt")
    data = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != FROZEN_PROMPTS[relative]:
        raise SystemExit(f"frozen prompt {relative} no longer has its registered hash")


def require_registered_differences(condition_id: str) -> dict[str, Any]:
    spec = CONDITIONS[condition_id]
    cond = resolve_condition(ROOT / "conditions" / condition_id / "condition.yaml",
                             repository_root=ROOT)
    parent = resolve_condition(ROOT / "conditions" / spec["base"] / "condition.yaml",
                               repository_root=ROOT)
    for key in sorted(set(cond) | set(parent)):
        if key not in ALLOWED_SECTIONS and cond.get(key) != parent.get(key):
            raise SystemExit(f"{condition_id} {key} differs from {spec['base']}")
    model, parent_model = cond["model"], parent["model"]
    exempt = {"pass1_decoding", "pass1_budget_matched_to"}
    if spec.get("frozen_prompt"):
        if model["prompt_pass1"] != SOLVE_PROMPT_V2:
            raise SystemExit(f"{condition_id} must use the frozen solve prompt {SOLVE_PROMPT_V2}")
        require_frozen_prompt(model["prompt_pass1"])
        exempt |= {"prompt_pass1", "prompt_pass1_provenance"}
    if spec.get("new_model"):
        exempt |= MODEL_IDENTITY_KEYS
        provenance = json.loads((ROOT / model["provenance"]).read_text(encoding="utf-8"))
        if provenance["repo_id"] != model["model_id"]:
            raise SystemExit(f"{condition_id} provenance is for {provenance['repo_id']}")
        for field in PROVENANCE_FIELDS:
            if model.get(field) != provenance[field]:
                raise SystemExit(f"{condition_id} model.{field} disagrees with {model['provenance']}")
    for key in sorted(set(model) | set(parent_model)):
        if key in exempt:
            continue
        if model.get(key) != parent_model.get(key):
            raise SystemExit(f"{condition_id} model.{key} differs from {spec['base']}")
    decoding, parent_decoding = dict(model["pass1_decoding"]), dict(parent_model["pass1_decoding"])
    if int(decoding.pop("max_new_tokens")) != BUDGET:
        raise SystemExit(f"{condition_id} pass-1 budget must be {BUDGET}")
    parent_decoding.pop("max_new_tokens")
    if decoding != parent_decoding:
        raise SystemExit(f"{condition_id} decoding differs from {spec['base']} beyond the budget")
    if model["model_id"] != spec["model_id"]:
        raise SystemExit(f"{condition_id} must run {spec['model_id']}")
    if model.get("thinking") is not False or float(decoding["temperature"]) != 0.0:
        raise SystemExit(f"{condition_id} requires thinking=false and temperature 0")
    registered = registration(cond)
    manifest = registered["manifests"][cond["dataset"]["manifest"]]
    if cond["dataset"]["id"] != spec.get("dataset", MATHFULL_ID) \
            or cond["dataset"]["manifest_rows"] != manifest["rows"] \
            or cond["dataset"]["manifest_sha256"] != manifest["sha256"]:
        raise SystemExit(f"{condition_id} dataset block disagrees with its dataset.yaml")
    if spec.get("dataset", MATHFULL_ID) != MATHFULL_ID:
        if cond["dataset"]["manifest"] not in EXTENSION_MANIFESTS:
            raise SystemExit(f"{condition_id} must run the registered {sorted(EXTENSION_MANIFESTS)} manifest")
    elif spec.get("test"):
        if cond["dataset"]["manifest"] not in TEST_MANIFESTS:
            raise SystemExit(f"{condition_id} is a test condition and must run a sealed test manifest")
    elif manifest["rows"] != EXPECTED_ROWS or cond["dataset"]["manifest"] in TEST_MANIFESTS:
        raise SystemExit(f"{condition_id} must run the {EXPECTED_ROWS}-row development manifest")
    if cond["replicates"]["plan_id"] != REPLICATE_PLAN_ID \
            or cond["replicates"]["seed_cell_id"] != condition_id:
        raise SystemExit(f"{condition_id} must use {REPLICATE_PLAN_ID} with its own seed cell")
    if cond["features"]["count"] != 7:
        raise SystemExit(f"{condition_id} must keep the seven registered scalars")
    return model


def _intact_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index < len(lines) - 2:     # only a torn final line is tolerated
                raise
    return rows


def _rewrite(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def reconcile(pass1_path: Path, trace_path: Path) -> dict[str, int]:
    pass1, traces = _intact_rows(pass1_path), _intact_rows(trace_path)
    ok_traces = {r["problem_id"] for r in traces if r.get("status") == "ok"}
    kept_pass1 = [r for r in pass1 if r["problem_id"] in ok_traces]
    kept_ids = {r["problem_id"] for r in kept_pass1}
    kept_traces = [r for r in traces if r["problem_id"] in kept_ids and r.get("status") == "ok"]
    if len(kept_ids) != len(kept_pass1) or len({r["problem_id"] for r in kept_traces}) != len(kept_traces):
        raise SystemExit("duplicate problem ids in a resumed run; refusing to continue")
    if pass1_path.exists():
        _rewrite(pass1_path, kept_pass1)
    if trace_path.exists():
        _rewrite(trace_path, kept_traces)
    return {"pass1_before": len(pass1), "traces_before": len(traces), "kept": len(kept_pass1)}


def manifest_info(condition_path: Path) -> dict[str, Any]:
    """The server path, hash and row count of the manifest a condition registers."""
    cond = resolve_condition(condition_path, repository_root=ROOT)
    registered = registration(cond)
    manifest = registered["manifests"][cond["dataset"]["manifest"]]
    if cond["dataset"]["manifest_sha256"] != manifest["sha256"] \
            or cond["dataset"]["manifest_rows"] != manifest["rows"]:
        raise SystemExit(f"{condition_path} dataset block disagrees with its dataset.yaml")
    return {"path": manifest["server_path"], "sha256": manifest["sha256"], "rows": int(manifest["rows"])}


# Sharding splits one run's generation across several jobs. Shard i of K is a contiguous
# block of the manifest; every row keeps the call seed, record id and capture id it would
# have had in a single job, because none of them depends on the shard. A merge job then
# reassembles the records in manifest order and runs evidence, grading and archiving once.
def shard_bounds(rows: int, index: int, count: int) -> tuple[int, int]:
    if count < 1 or not 0 <= index < count:
        raise SystemExit(f"shard {index}/{count} is out of range")
    size, extra = divmod(rows, count)
    start = index * size + min(index, extra)
    return start, start + size + (1 if index < extra else 0)


def _manifest_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_shard_manifest(full: Path, index: int, count: int, out: Path) -> int:
    lines = _manifest_lines(full)
    start, end = shard_bounds(len(lines), index, count)
    out.write_text("".join(line + "\n" for line in lines[start:end]), encoding="utf-8")
    return end - start


def merge_shards(full: Path, shard_root: Path, count: int, out_dir: Path, names: list[str]) -> dict[str, int]:
    """Concatenate each shard's record files in manifest order, byte for byte.

    Every shard must hold exactly its block of problem ids, once each and fully parseable,
    and every native trace must have status ok; anything else stops the merge.
    """
    ids = [json.loads(line)["problem_id"] for line in _manifest_lines(full)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        by_id: dict[str, str] = {}
        for index in range(count):
            start, end = shard_bounds(len(ids), index, count)
            path = shard_root / str(index) / "records" / name
            if not path.exists():
                raise SystemExit(f"shard {index} has no {name}")
            seen: dict[str, str] = {}
            for line in _manifest_lines(path):
                row = json.loads(line)
                if row["problem_id"] in seen:
                    raise SystemExit(f"shard {index} {name} repeats {row['problem_id']}")
                if "trace" in name and row.get("status") != "ok":
                    raise SystemExit(f"shard {index} {name} has status {row.get('status')} for {row['problem_id']}")
                seen[row["problem_id"]] = line
            if set(seen) != set(ids[start:end]):
                raise SystemExit(f"shard {index} {name} does not hold exactly its {end - start} manifest rows")
            by_id.update(seen)
        (out_dir / name).write_text("".join(by_id[p] + "\n" for p in ids), encoding="utf-8")
    return {"rows": len(ids), "shards": count, "files": len(names)}


UNGRADEABLE_PATH = ROOT / "datasets" / MATHFULL_ID / "ungradeable_gold.yaml"


def ungradeable_ids() -> set[str]:
    registered = yaml.safe_load(UNGRADEABLE_PATH.read_text(encoding="utf-8"))
    return {row["problem_id"] for row in registered["ungradeable"]}


def analyze(argv: list[str]) -> None:
    """c017's analysis, run on every row except the registered ungradeable-gold problems.

    Rows are dropped only if listed in ungradeable_gold.yaml, whatever their grade. Any
    other grader_error stops the analysis, exactly as c017's own analysis does; an
    unparseable model output is graded wrong and stays in.
    """

    import argparse
    import tempfile

    parser = argparse.ArgumentParser()
    for name in ("--run-id", "--pass1-records", "--evidence", "--grades", "--output"):
        parser.add_argument(name, required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args(argv)
    listed = ungradeable_ids()
    sources = {name: list(c017.read_jsonl(Path(path))) for name, path in
               (("pass1", args.pass1_records), ("evidence", args.evidence), ("grades", args.grades))}
    unlisted_errors = sorted(r["problem_id"] for r in sources["grades"]
                             if r["u"] is None and r["problem_id"] not in listed)
    if unlisted_errors:
        raise SystemExit(f"grader_error on problems not registered as ungradeable: {unlisted_errors}")
    present = sorted({r["problem_id"] for r in sources["pass1"]} & listed)
    with tempfile.TemporaryDirectory() as tmp:
        kept = {}
        for name, rows in sources.items():
            path = Path(tmp) / f"{name}.jsonl"
            path.write_text("".join(canonical_json(r) + "\n" for r in rows
                                    if r["problem_id"] not in listed), encoding="utf-8")
            kept[name] = str(path)
        c017.stage_analyze(argparse.Namespace(
            run_id=args.run_id, pass1_records=kept["pass1"], evidence=kept["evidence"],
            grades=kept["grades"], expected_rows=args.expected_rows - len(present),
            output=args.output))
    output = Path(args.output)
    report = json.loads(output.read_text(encoding="utf-8"))
    manifest_name = resolve_condition(c017.CONDITION_PATH, repository_root=ROOT)["dataset"]["manifest"]
    report.update({"manifest_rows": args.expected_rows,
                   "manifest": manifest_name,
                   "development_only": manifest_name not in TEST_MANIFESTS | EXTENSION_MANIFESTS,
                   "excluded_ungradeable_gold": present,
                   "ungradeable_rule_file": str(UNGRADEABLE_PATH.relative_to(ROOT))})
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = output.parent / "summary.md"
    lines = summary.read_text(encoding="utf-8").split("\n")
    lines[1:1] = ["", f"- Manifest rows: {args.expected_rows}; excluded as ungradeable gold "
                  f"({UNGRADEABLE_PATH.relative_to(ROOT).as_posix()}): {len(present)} "
                  f"{present}. Every figure below is over the remaining "
                  f"{args.expected_rows - len(present)} rows."]
    summary.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "--condition-id" or argv[1] not in CONDITIONS:
        raise SystemExit(__doc__)
    condition_id, command, rest = argv[1], argv[2], argv[3:]
    configure(condition_id)
    if command == "check-runnable":
        model = require_registered_differences(condition_id)
        print(canonical_json({"runnable": True, "condition_id": condition_id,
                              "model_id": model["model_id"], "revision": model["revision"],
                              "max_new_tokens": BUDGET, "dataset_id": DATASET_ID}))
        return 0
    if command == "reconcile":
        pass1_path, trace_path = Path(rest[0]), Path(rest[1])
        print(canonical_json({"stage": "reconcile", **reconcile(pass1_path, trace_path)}))
        return 0
    if command == "manifest-info":
        info = manifest_info(ROOT / "conditions" / condition_id / "condition.yaml")
        print(info["path"], info["sha256"], info["rows"])
        return 0
    if command == "shard-manifest":
        full, index, count, out = Path(rest[0]), int(rest[1]), int(rest[2]), Path(rest[3])
        print(write_shard_manifest(full, index, count, out))
        return 0
    if command == "merge-shards":
        full, shard_root, count, out_dir = Path(rest[0]), Path(rest[1]), int(rest[2]), Path(rest[3])
        print(canonical_json({"stage": "merge-shards", **merge_shards(full, shard_root, count, out_dir, rest[4:])}))
        return 0
    if command == "analyze":
        analyze(rest)
        return 0
    if command == "generate" and "--base-seed" not in rest:
        rest = rest + ["--base-seed", str(BASE_SEED)]
    sys.argv = [sys.argv[0], command, *rest]
    return c017.main()


if __name__ == "__main__":
    raise SystemExit(main())
