#!/usr/bin/env python3
"""The MMLU-Pro knowledge study (docs/plan/mmlu_pro_study.md) on dev1000: Qwen3-4B alone (c070),
Qwen3-8B alone (c071), Qwen3-8B reviewing c070 (c072) and Qwen3-4B reviewing c071 (c073).

Usage: run_mmlu_pro.py --condition-id {c070,...,c079} <command> [args]
(c074-c080: the other mathfull models alone, see SOLO_MODELS; c081-c084: c070-c073 on the
test1500 prefix of the sealed test, see TEST_CONDITIONS; c089 and c120-c125: the other models
alone on test1500.)

Commands:
    check-runnable  the frozen prompt, the model identity (equal to the registered c025 or c026
                    identity), dataset hash, decoding and seed cell are the registered ones, and
                    c071-c073 differ from their base only where registered
    smoke           one call on a toy question: exact token alignment and a paired capture
    generate        c070/c071: one solve call per row with same-forward capture; resumable
    review          c072/c073: one review call per row on the other model's output; resumable
    reconcile       keep only rows present and valid in both the record and trace files
    evidence        text_mc_scalar_v1 for the condition's own output, before grading
    grade           letter equality with the reference answer (and, for a review, both GV
                    delivery readings)
    analyze         reports/analysis.json and reports/summary.md

The structure is scripts/run_mmmu_pro.py without images; that file is not imported so the
vision runs resuming on it cannot be affected by changes here.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c014_router as shared  # noqa: E402
import run_mathfull_solo as mathfull  # noqa: E402

from authority_allocation import derive_call_seed, mmlu_pro  # noqa: E402
from authority_allocation.api import api_call, chat_template_kwargs_for  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.prompts import split_prompt  # noqa: E402
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json, sha256_bytes, sha256_file, stable_id, utc_now,
)
from authority_allocation.text_mc_features import FEATURE_CONTRACT, derive_features  # noqa: E402

DATASET_ID = "mmlupro-rb189ec7-s20260927"
DATASET_PATH = ROOT / "datasets" / DATASET_ID / "dataset.yaml"
SOURCE_REPO = "TIGER-Lab/MMLU-Pro"
SOURCE_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
REPLICATE_PLAN_ID = "r014"
REPLICATE_ID = "p01"
BASE_SEED = 20260927
EXPECTED_ROWS = 1000
BUDGET = 4096
IDENTITY_FIELDS = ("model_id", "revision", "tokenizer_revision", "server_path", "chat_template_hash",
                   "chat_template_kwargs")

REVIEW_SECTIONS = {"condition_id", "status", "purpose", "gate", "runnable_gate", "model",
                   "replicates", "protocol", "generator_source", "delivery", "stages"}
REVIEW_MODEL_KEYS = {"prompt_pass1", "prompt_pass1_provenance", "prompt_inputs"}
CONDITIONS = {
    "c070": {"role": "solve", "model_id": "Qwen/Qwen3-4B", "identity": "c025",
             "prompt": mmlu_pro.MC_SOLVE_PROMPT},
    "c071": {"role": "solve", "model_id": "Qwen/Qwen3-8B", "identity": "c026",
             "prompt": mmlu_pro.MC_SOLVE_PROMPT, "base": "c070",
             "allowed_sections": {"condition_id", "status", "purpose", "gate", "runnable_gate",
                                  "model", "replicates"},
             "allowed_model_keys": {"model_id", "revision", "tokenizer_revision", "provenance",
                                    "identity_reference", "server_path", "chat_template_hash"}},
    "c072": {"role": "review", "model_id": "Qwen/Qwen3-8B", "identity": "c026",
             "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c071", "g": "c070",
             "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS},
    "c073": {"role": "review", "model_id": "Qwen/Qwen3-4B", "identity": "c025",
             "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c070", "g": "c071",
             "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS},
}
# Every other mathfull model alone (2026-09-27): c070 with only the model changed, to
# the registered identity of that model's mathfull solo condition, chat-template inputs included.
SOLO_MODELS = {"c074": ("meta-llama/Llama-3.2-3B-Instruct", "c029"),
               "c075": ("Qwen/Qwen2.5-3B-Instruct", "c030"),
               "c076": ("Qwen/Qwen2.5-7B-Instruct", "c031"),
               "c077": ("mistralai/Ministral-8B-Instruct-2410", "c032"),
               "c078": ("mistralai/Mistral-Small-24B-Instruct-2501", "c054"),
               "c079": ("Qwen/Qwen3-32B", "c055"),
               "c080": ("Qwen/Qwen2.5-72B-Instruct", "c066")}
for _cid, (_model, _ref) in SOLO_MODELS.items():
    CONDITIONS[_cid] = {"role": "solve", "model_id": _model, "identity": _ref,
                        "prompt": mmlu_pro.MC_SOLVE_PROMPT, "base": "c070",
                        "allowed_sections": {"condition_id", "status", "purpose", "gate", "runnable_gate",
                                             "model", "replicates", "protocol"},
                        "allowed_model_keys": {"model_id", "revision", "tokenizer_revision", "provenance",
                                               "identity_reference", "server_path", "chat_template_hash",
                                               "chat_template_kwargs"}}
# c070-c073 on the test1500 prefix (docs/plan/mmlu_pro_study.md section 10): each differs from
# its development condition only in the manifest, evidence schema id, seed cell and, for a
# review, the source of the reviewed outputs.
TEST_SECTIONS = {"condition_id", "status", "purpose", "gate", "runnable_gate", "replicates", "dataset",
                 "protocol", "generator_source"}
TEST_CONDITIONS = {"c081": ("c070", None), "c082": ("c071", None), "c083": ("c072", "c081"),
                   "c084": ("c073", "c082"),
                   # Mistral-Small-24B alone on the same rows, a stability check (plan section 12).
                   "c089": ("c078", None),
                   # The remaining dev1000 models alone on the same rows (plan section 13).
                   "c120": ("c074", None),
                   "c121": ("c075", None),
                   "c122": ("c076", None),
                   "c123": ("c077", None),
                   "c124": ("c079", None),
                   "c125": ("c080", None)}
for _cid, (_dev, _g) in TEST_CONDITIONS.items():
    CONDITIONS[_cid] = {**{k: v for k, v in CONDITIONS[_dev].items() if k not in ("base", "g")},
                        "base": _dev, "manifest": "test1500", "allowed_sections": TEST_SECTIONS,
                        "allowed_model_keys": set(), **({"g": _g} if _g else {})}
# The Qwen2.5-7B x Qwen3-4B pair, as in the math domain (plan section 11).
CONDITIONS["c085"] = {"role": "review", "model_id": "Qwen/Qwen2.5-7B-Instruct", "identity": "c031",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c076", "g": "c070",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c086"] = {"role": "review", "model_id": "Qwen/Qwen3-4B", "identity": "c025",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c070", "g": "c076",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
# The Qwen3-8B x Mistral-Small-24B pair, as in the math domain (plan section 12).
CONDITIONS["c087"] = {"role": "review", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501", "identity": "c054",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c078", "g": "c071",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c088"] = {"role": "review", "model_id": "Qwen/Qwen3-8B", "identity": "c026",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c071", "g": "c078",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
# The stratified pair selection (plan section 15): development reviews, then each review on test1500.
CONDITIONS["c126"] = {"role": "review", "model_id": "Qwen/Qwen2.5-7B-Instruct", "identity": "c031",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c076", "g": "c071",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c127"] = {"role": "review", "model_id": "Qwen/Qwen3-8B", "identity": "c026",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c071", "g": "c076",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c128"] = {"role": "review", "model_id": "mistralai/Ministral-8B-Instruct-2410", "identity": "c032",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c077", "g": "c075",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c129"] = {"role": "review", "model_id": "Qwen/Qwen2.5-3B-Instruct", "identity": "c030",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c075", "g": "c077",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c130"] = {"role": "review", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "identity": "c029",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c074", "g": "c075",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c131"] = {"role": "review", "model_id": "Qwen/Qwen2.5-3B-Instruct", "identity": "c030",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c075", "g": "c074",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c132"] = {"role": "review", "model_id": "Qwen/Qwen2.5-3B-Instruct", "identity": "c030",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c075", "g": "c076",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c133"] = {"role": "review", "model_id": "Qwen/Qwen2.5-7B-Instruct", "identity": "c031",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c076", "g": "c075",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c134"] = {"role": "review", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "identity": "c029",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c074", "g": "c071",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c135"] = {"role": "review", "model_id": "Qwen/Qwen3-8B", "identity": "c026",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c071", "g": "c074",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
STRATIFIED_TEST = {"c136": ("c085", "c081"),
                   "c137": ("c086", "c122"),
                   "c138": ("c126", "c082"),
                   "c139": ("c127", "c122"),
                   "c140": ("c128", "c121"),
                   "c141": ("c129", "c123"),
                   "c142": ("c130", "c121"),
                   "c143": ("c131", "c120"),
                   "c144": ("c132", "c122"),
                   "c145": ("c133", "c121"),
                   "c146": ("c134", "c082"),
                   "c147": ("c135", "c120"),
                   # Self-review on test1500 (2026-10-09, docs/plan/self_review_test1500.md):
                   # the reviewer of c147 / c138 reads its own test solo instead of the other model's.
                   "c388": ("c135", "c082"),
                   "c389": ("c126", "c122")}
for _cid, (_dev, _g) in STRATIFIED_TEST.items():
    CONDITIONS[_cid] = {**{k: v for k, v in CONDITIONS[_dev].items() if k not in ("base", "g")},
                        "base": _dev, "manifest": "test1500", "allowed_sections": TEST_SECTIONS,
                        "allowed_model_keys": set(), "g": _g}
# Prompt-variant sanity check (2026-10-09, docs/plan/prompt_variant_test500.md):
# c147 / c136 with only the review prompt changed to the frozen commit-first prompt.
for _cid, _test in (("c392", "c147"), ("c393", "c136")):
    CONDITIONS[_cid] = {**{k: v for k, v in CONDITIONS[_test].items() if k != "base"},
                        "base": _test, "prompt": mmlu_pro.MC_REVIEW_COMMIT_FIRST_PROMPT,
                        "allowed_sections": TEST_SECTIONS | {"model"},
                        "allowed_model_keys": {"prompt_pass1", "prompt_pass1_provenance"}}
# Stratified pairs with a strong model (plan section 16): development reviews, then each review on test1500.
CONDITIONS["c170"] = {"role": "review", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501", "identity": "c054",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c078", "g": "c076",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c171"] = {"role": "review", "model_id": "Qwen/Qwen2.5-7B-Instruct", "identity": "c031",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c076", "g": "c078",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c172"] = {"role": "review", "model_id": "Qwen/Qwen3-32B", "identity": "c055",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c079", "g": "c075",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c173"] = {"role": "review", "model_id": "Qwen/Qwen2.5-3B-Instruct", "identity": "c030",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c075", "g": "c079",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c174"] = {"role": "review", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501", "identity": "c054",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c078", "g": "c074",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c175"] = {"role": "review", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "identity": "c029",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c074", "g": "c078",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
STRONG_TEST = {"c176": ("c087", "c082"),
                "c177": ("c088", "c089"),
                "c178": ("c170", "c122"),
                "c179": ("c171", "c089"),
                "c180": ("c172", "c121"),
                "c181": ("c173", "c124"),
                "c182": ("c174", "c120"),
                "c183": ("c175", "c089")}
for _cid, (_dev, _g) in STRONG_TEST.items():
    CONDITIONS[_cid] = {**{k: v for k, v in CONDITIONS[_dev].items() if k not in ("base", "g")},
                        "base": _dev, "manifest": "test1500", "allowed_sections": TEST_SECTIONS,
                        "allowed_model_keys": set(), "g": _g}
# The strong/strong cell (plan section 16): development reviews, then each review on test1500.
CONDITIONS["c184"] = {"role": "review", "model_id": "Qwen/Qwen3-32B", "identity": "c055",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c079", "g": "c078",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c185"] = {"role": "review", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501", "identity": "c054",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c078", "g": "c079",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c186"] = {"role": "review", "model_id": "Qwen/Qwen2.5-72B-Instruct", "identity": "c066",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c080", "g": "c078",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
CONDITIONS["c187"] = {"role": "review", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501", "identity": "c054",
                      "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": "c078", "g": "c080",
                      "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
STRONG_STRONG_TEST = {"c188": ("c184", "c089"),
                       "c189": ("c185", "c124"),
                       "c190": ("c186", "c089"),
                       "c191": ("c187", "c125")}
for _cid, (_dev, _g) in STRONG_STRONG_TEST.items():
    CONDITIONS[_cid] = {**{k: v for k, v in CONDITIONS[_dev].items() if k not in ("base", "g")},
                        "base": _dev, "manifest": "test1500", "allowed_sections": TEST_SECTIONS,
                        "allowed_model_keys": set(), "g": _g}
# The originally planned pairs with Qwen2.5-72B (plan section 20, decided 2026-10-02): development
# reviews c270-c273, then each review on test1500 (c274-c277).
_REVIEWERS = {"c080": ("Qwen/Qwen2.5-72B-Instruct", "c066"), "c076": ("Qwen/Qwen2.5-7B-Instruct", "c031"),
              "c074": ("meta-llama/Llama-3.2-3B-Instruct", "c029")}
for _cid, _base, _g in (("c270", "c080", "c076"), ("c271", "c076", "c080"),
                        ("c272", "c080", "c074"), ("c273", "c074", "c080")):
    CONDITIONS[_cid] = {"role": "review", "model_id": _REVIEWERS[_base][0], "identity": _REVIEWERS[_base][1],
                        "prompt": mmlu_pro.MC_REVIEW_PROMPT, "base": _base, "g": _g,
                        "allowed_sections": REVIEW_SECTIONS, "allowed_model_keys": REVIEW_MODEL_KEYS}
ORIGINAL72_TEST = {"c274": ("c270", "c122"),
                   "c275": ("c271", "c125"),
                   "c276": ("c272", "c120"),
                   "c277": ("c273", "c125")}
for _cid, (_dev, _g) in ORIGINAL72_TEST.items():
    CONDITIONS[_cid] = {**{k: v for k, v in CONDITIONS[_dev].items() if k not in ("base", "g")},
                        "base": _dev, "manifest": "test1500", "allowed_sections": TEST_SECTIONS,
                        "allowed_model_keys": set(), "g": _g}
CONDITION_ID = "c070"
ROLE = "solve"


# ------------------------------------------------------------------ configuration

def configure(condition_id: str) -> None:
    global CONDITION_ID, ROLE
    if condition_id not in CONDITIONS:
        raise SystemExit(f"unknown condition {condition_id}")
    CONDITION_ID, ROLE = condition_id, CONDITIONS[condition_id]["role"]
    shared.CONDITION_ID = condition_id
    shared.CONDITION_PATH = ROOT / "conditions" / condition_id / "condition.yaml"
    shared.DATASET_ID = DATASET_ID
    shared.REPLICATE_PLAN_ID = REPLICATE_PLAN_ID
    shared.STAGE = "mmlu-pro-" + ROLE
    shared.SEED_CELL_ID = condition_id
    shared.EXPECTED_ROWS = EXPECTED_ROWS
    shared.configure()


def condition(condition_id: str | None = None) -> dict[str, Any]:
    cid = condition_id or CONDITION_ID
    return resolve_condition(ROOT / "conditions" / cid / "condition.yaml", repository_root=ROOT)


def registered_dataset() -> dict[str, Any]:
    return yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8"))


def require_registered(condition_id: str) -> dict[str, Any]:
    spec = CONDITIONS[condition_id]
    cond = condition(condition_id)
    model = cond["model"]
    if model["prompt_pass1"] != spec["prompt"]:
        raise SystemExit(f"{condition_id} must use the frozen prompt {spec['prompt']}")
    mmlu_pro.require_frozen_prompt(ROOT, model["prompt_pass1"])
    if model["model_id"] != spec["model_id"] or model.get("identity_reference") != spec["identity"]:
        raise SystemExit(f"{condition_id} must run {spec['model_id']} with {spec['identity']}'s identity")
    reference = condition(spec["identity"])["model"]
    for field in IDENTITY_FIELDS:
        if model.get(field) != reference.get(field):
            raise SystemExit(f"{condition_id} model.{field} differs from {spec['identity']}'s")
    decoding = model["pass1_decoding"]
    if float(decoding["temperature"]) != 0.0 or model.get("thinking") is not False:
        raise SystemExit(f"{condition_id} requires temperature 0 and thinking false")
    if int(decoding["max_new_tokens"]) != BUDGET:
        raise SystemExit(f"{condition_id} output budget must be {BUDGET}")
    if float(decoding["repetition_penalty"]) != 1.0 or float(decoding["presence_penalty"]) != 0.0:
        raise SystemExit(f"{condition_id} must send repetition_penalty 1.0 and presence_penalty 0.0")
    registered = registered_dataset()
    manifest_name = spec.get("manifest", "dev1000")
    manifest = registered["manifests"][cond["dataset"]["manifest"]]
    if (cond["dataset"]["id"] != DATASET_ID or cond["dataset"]["manifest"] != manifest_name
            or cond["dataset"]["manifest_rows"] != manifest["rows"]
            or (manifest_name == "dev1000" and manifest["rows"] != EXPECTED_ROWS)
            or cond["dataset"]["manifest_sha256"] != manifest["sha256"]):
        raise SystemExit(f"{condition_id} dataset block disagrees with {DATASET_PATH.name}")
    if cond["replicates"]["plan_id"] != REPLICATE_PLAN_ID \
            or cond["replicates"]["seed_cell_id"] != condition_id:
        raise SystemExit(f"{condition_id} must use {REPLICATE_PLAN_ID} with its own seed cell")
    if "base" in spec:
        parent = condition(spec["base"])
        for key in sorted(set(cond) | set(parent)):
            if key not in spec["allowed_sections"] and cond.get(key) != parent.get(key):
                raise SystemExit(f"{condition_id} {key} differs from {spec['base']}")
        for key in sorted(set(model) | set(parent["model"])):
            if key not in spec["allowed_model_keys"] and model.get(key) != parent["model"].get(key):
                raise SystemExit(f"{condition_id} model.{key} differs from {spec['base']}")
    if spec["role"] == "review":
        source = cond["generator_source"]
        if source["condition_id"] != spec["g"]:
            raise SystemExit(f"{condition_id} must review {spec['g']}'s outputs")
        if "REQUIRED_BEFORE_RUN" in str(source["artifact_path"]) + str(source["pass1_run_id"]):
            raise SystemExit(f"{condition_id} generator_source is not filled in yet")
    return model


# ------------------------------------------------------------------ data

def manifest_rows(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} manifest rows, got {len(rows)}")
    return rows


def load_mmlu_pro_rows(problem_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Join an id-only manifest to the pinned source."""
    from datasets import load_dataset

    wanted = set(problem_ids)
    rows: dict[str, dict[str, Any]] = {}
    for row in load_dataset(SOURCE_REPO, revision=SOURCE_REVISION, split="test"):
        pid = str(row["question_id"])
        if pid not in wanted:
            continue
        if pid in rows:
            raise ValueError(f"duplicate source row {pid}")
        rows[pid] = {"question": row["question"], "options": list(row["options"]),
                     "answer": str(row["answer"]).strip(), "category": str(row["category"]),
                     "origin": mmlu_pro.source_origin(str(row["src"]))}
    missing = wanted.difference(rows)
    if missing:
        raise ValueError(f"manifest ids missing from the pinned source: {sorted(missing)[:5]}")
    return rows


# ------------------------------------------------------------------ model calls

def model_identity(model: Mapping[str, Any]) -> dict[str, Any]:
    return {"model_id": model["model_id"], "revision": model["revision"],
            "server_path": model["server_path"], "backend": model["backend"],
            "prompt_sha256": sha256_file(ROOT / model["prompt_pass1"]),
            "chat_template_sha256": model["chat_template_hash"], "dtype": model["dtype"],
            "quantization": model["quantization"]}


def build_messages(model: Mapping[str, Any], question: str, options: list[str],
                   y0: str | None = None) -> list[dict[str, str]]:
    system, user = split_prompt(ROOT / model["prompt_pass1"])
    return [{"role": "system", "content": system},
            {"role": "user", "content": mmlu_pro.user_message(user, question=question,
                                                              options=options, y0=y0)}]


def call_model(args: argparse.Namespace, model: Mapping[str, Any], messages: list[dict[str, str]],
               seed: int, capture_id: str) -> tuple[str, dict[str, Any]]:
    content, _, usage = api_call(
        api_base=args.api_base, served_model=args.served_model, messages=messages,
        decoding={**model["pass1_decoding"],
                  "output_logprobs": {"enabled": True, "mode": "raw_logprobs",
                                      "return_token_ids": True},
                  "vllm_xargs": {"c014_capture_id": capture_id}},
        seed=seed, thinking=False, chat_template_kwargs=chat_template_kwargs_for(model))
    return content, usage


def stage_smoke(args: argparse.Namespace) -> None:
    model = require_registered(CONDITION_ID)
    y0 = "Water boils at 100 degrees Celsius at sea level.\nAnswer: A" if ROLE == "review" else None
    messages = build_messages(model, "At sea level, water boils at which temperature?",
                              ["100 °C", "50 °C", "0 °C"], y0)
    capture_id = stable_id(args.run_id, "smoke", "same_forward_logit_capture")
    content, usage = call_model(args, model, messages, 1, capture_id)
    trace = usage.get("token_logprobs") or {}
    record = {"run_id": args.run_id, "problem_id": "smoke",
              "outputs": [{"raw_output": content, "token_logprobs": trace}]}
    capture = shared.wait_for_same_forward_capture(Path(args.capture_sidecar), capture_id,
                                                   timeout_seconds=60)
    native = shared.build_same_forward_logit_record(run_id=args.run_id, pass1_record=record,
                                                    capture=capture)
    problems = []
    if not content:
        problems.append("empty content")
    if trace.get("status") != "ok":
        problems.append(f"trace {trace.get('status')}")
    if native["status"] != "ok":
        problems.append(f"native capture {native['status']}")
    print(canonical_json({"stage": "smoke", "condition_id": CONDITION_ID,
                          "native_status": native["status"],
                          "answer": mmlu_pro.parse_answer_letter(content, 3),
                          "finish_reason": usage.get("finish_reason"), "problems": problems}))
    if problems:
        raise SystemExit("smoke failed: " + "; ".join(problems))


def _run_rows(args: argparse.Namespace, *, g_rows: Mapping[str, Mapping[str, Any]] | None) -> None:
    model = require_registered(CONDITION_ID)
    manifest = manifest_rows(Path(args.manifest), int(args.expected_rows))
    ids = [str(r["problem_id"]) for r in manifest]
    by_id = {str(r["problem_id"]): r for r in manifest}
    sources = load_mmlu_pro_rows(ids)
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    scope = "review" if g_rows is not None else "pass1"
    written = 0
    for problem_id in ids:
        if (problem_id,) in done:
            continue
        row, manifest_row = sources[problem_id], by_id[problem_id]
        if len(row["options"]) != int(manifest_row["n_options"]):
            raise RuntimeError(f"{problem_id} option count differs from the manifest")
        y0 = None if g_rows is None else (g_rows[problem_id]["outputs"][0]["raw_output"] or "")
        messages = build_messages(model, row["question"], row["options"], y0)
        seed = derive_call_seed(REPLICATE_PLAN_ID, scope, args.replicate_id, int(args.base_seed),
                                CONDITION_ID, problem_id, 0)
        capture_id = stable_id(args.run_id, problem_id, "same_forward_logit_capture")
        content, usage = call_model(args, model, messages, seed, capture_id)
        trace = usage.pop("token_logprobs", None)
        letter = mmlu_pro.parse_answer_letter(content, len(row["options"]))
        record = {
            "schema_version": "mc-review-record-v1" if g_rows is not None else "mc-record-v1",
            "record_id": stable_id(args.run_id, problem_id, scope),
            "run_id": args.run_id, "condition_id": CONDITION_ID,
            "condition_hash": shared.condition_hash(), "dataset_id": DATASET_ID,
            "replicate_id": args.replicate_id, "stage": shared.STAGE,
            "problem_id": problem_id, "problem_text": row["question"],
            "problem_metadata": {"category": manifest_row["category"], "origin": manifest_row["origin"],
                                 "n_options": int(manifest_row["n_options"])},
            "generator": model_identity(model),
            "outputs": [{"sample_index": 0, "call_seed": seed, "raw_output": content,
                         "extracted_answer": letter,
                         "parse_status": "ok" if letter is not None else "model_invalid",
                         "token_logprobs": trace, "usage": shared.base.compact_usage(usage)}],
            "y0_sample_index": 0, "created_at_utc": utc_now(),
        }
        if g_rows is not None:
            g_row = g_rows[problem_id]
            record.update({"source_g_run_id": g_row["run_id"], "source_g_record_id": g_row["record_id"],
                           "source_g_output_sha256": sha256_bytes(y0.encode("utf-8"))})
        append_jsonl(output, record)
        capture = shared.wait_for_same_forward_capture(Path(args.capture_sidecar), capture_id,
                                                       timeout_seconds=60)
        native = shared.build_same_forward_logit_record(run_id=args.logit_run_id,
                                                        pass1_record=record, capture=capture)
        append_jsonl(Path(args.logit_output), native)
        if native["status"] != "ok":
            raise RuntimeError(f"same-forward logit capture failed for {problem_id}: {native['status']}")
        written += 1
    print(canonical_json({"stage": scope, "written": written, "total": len(ids)}))


def stage_generate(args: argparse.Namespace) -> None:
    if ROLE != "solve":
        raise SystemExit(f"{CONDITION_ID} reviews; use the review command")
    _run_rows(args, g_rows=None)


def stage_review(args: argparse.Namespace) -> None:
    if ROLE != "review":
        raise SystemExit(f"{CONDITION_ID} solves; use the generate command")
    g_rows = {r["problem_id"]: r for r in read_jsonl(Path(args.g_records))}
    source_run = condition()["generator_source"]["pass1_run_id"]
    foreign = sorted({r["run_id"] for r in g_rows.values()} - {source_run})
    if foreign:
        raise RuntimeError(f"reviewed records come from {foreign}, not the registered {source_run}")
    _run_rows(args, g_rows=g_rows)


# ------------------------------------------------------------------ evidence, grading, analysis

def stage_evidence(args: argparse.Namespace) -> None:
    natives = {row["problem_id"]: row for row in read_jsonl(Path(args.logit_traces))}
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for record in read_jsonl(Path(args.records)):
        pid = record["problem_id"]
        if (pid,) in done:
            continue
        native = natives.get(pid)
        if native is None or native.get("status") != "ok":
            raise RuntimeError(f"missing valid native trace for {pid}")
        features, audit = derive_features(
            output=record["outputs"][0], native=native,
            n_options=int(record["problem_metadata"]["n_options"]), question_text=record["problem_text"])
        append_jsonl(output, {
            "schema_version": "mc-evidence-record-v1",
            "record_id": stable_id(args.run_id, pid, "mc-evidence"), "run_id": args.run_id,
            "source_record_run_id": record["run_id"], "source_logit_run_id": native["run_id"],
            "condition_id": CONDITION_ID, "condition_hash": shared.condition_hash(),
            "problem_id": pid, "role": ROLE, "feature_contract": FEATURE_CONTRACT,
            "features": features, "missingness": {k: v is None for k, v in features.items()},
            "audit": audit, "created_at_utc": utc_now()})
        written += 1
    print(canonical_json({"stage": "evidence", "written": written}))


def grade_letter(letter: str | None, gold: str) -> int:
    return int(letter is not None and letter == gold)


def stage_grade(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.records))
    sources = load_mmlu_pro_rows([r["problem_id"] for r in records])
    g_rows = ({r["problem_id"]: r for r in read_jsonl(Path(args.g_records))}
              if ROLE == "review" else {})
    output = Path(args.output)
    output.unlink(missing_ok=True)
    own = "u_review" if ROLE == "review" else "u0"
    for record in records:
        pid, out = record["problem_id"], record["outputs"][0]
        gold, n_options = sources[pid]["answer"], len(sources[pid]["options"])
        official = mmlu_pro.official_parse_without_random(out["raw_output"], n_options)
        append_jsonl(output, {"problem_id": pid, "outcome": own,
                              "u": grade_letter(out["extracted_answer"], gold),
                              "answer": out["extracted_answer"], "source_record_id": record["record_id"]})
        append_jsonl(output, {"problem_id": pid, "outcome": own + "_official_parser",
                              "u": grade_letter(official, gold), "answer": official,
                              "source_record_id": record["record_id"]})
        if ROLE == "review":
            v_letter = out["extracted_answer"]
            g_letter = g_rows[pid]["outputs"][0]["extracted_answer"]
            delivered = v_letter if v_letter is not None else g_letter
            append_jsonl(output, {"problem_id": pid, "outcome": "u_gv", "u": grade_letter(delivered, gold),
                                  "answer": delivered,
                                  "delivered_from": "review" if v_letter is not None else "g_fallback",
                                  "source_record_id": record["record_id"]})
            append_jsonl(output, {"problem_id": pid, "outcome": "u_gv_strict",
                                  "u": grade_letter(v_letter, gold), "answer": v_letter,
                                  "source_record_id": record["record_id"]})
    print(canonical_json({"stage": "grade", "rows": len(records)}))


def repetition_share(text: str, n: int = 8, window: int = 400) -> float:
    """Share of repeated word n-grams in the last ``window`` words; near 1 for a loop."""
    words = text.split()[-window:]
    grams = [tuple(words[i:i + n]) for i in range(max(len(words) - n + 1, 0))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def stage_analyze(args: argparse.Namespace) -> None:
    records = {r["problem_id"]: r for r in read_jsonl(Path(args.records))}
    evidence = {r["problem_id"]: r for r in read_jsonl(Path(args.evidence))}
    grades: dict[str, dict[str, Mapping[str, Any]]] = collections.defaultdict(dict)
    for row in read_jsonl(Path(args.grades)):
        grades[row["outcome"]][row["problem_id"]] = row
    expected = int(args.expected_rows)
    own = "u_review" if ROLE == "review" else "u0"
    if not (len(records) == len(evidence) == len(grades[own]) == expected):
        raise RuntimeError(f"incomplete inputs: {len(records)}, {len(evidence)}, {len(grades[own])}")
    ids = sorted(records)
    total = lambda outcome: sum(int(grades[outcome][p]["u"]) for p in ids)  # noqa: E731
    usage = [records[p]["outputs"][0]["usage"] for p in ids]
    tokens = [int(u.get("output_tokens") or 0) for u in usage]
    truncated = [p for p, u in zip(ids, usage) if u.get("finish_reason") == "length"]
    loops = [p for p in truncated if repetition_share(records[p]["outputs"][0]["raw_output"]) >= 0.5]
    parseable = sum(records[p]["outputs"][0]["extracted_answer"] is not None for p in ids)
    missing = {name: sum(evidence[p]["missingness"][name] for p in ids)
               for name in evidence[ids[0]]["features"]}
    by_category = collections.defaultdict(lambda: [0, 0])
    for p in ids:
        cell = by_category[records[p]["problem_metadata"]["category"]]
        cell[0] += int(grades[own][p]["u"])
        cell[1] += 1
    report: dict[str, Any] = {
        "condition_id": CONDITION_ID, "run_id": args.run_id, "role": ROLE, "rows": expected,
        "correct": total(own), "accuracy": total(own) / expected,
        "official_parser_correct_without_random": total(own + "_official_parser"),
        "answer_parseable": parseable, "truncated_rows": len(truncated),
        "truncated_repetition_loops": len(loops), "truncated_problem_ids": truncated,
        "correct_by_category": {k: {"correct": v[0], "rows": v[1]} for k, v in sorted(by_category.items())},
        "output_tokens": {"sum": sum(tokens), "mean": statistics.mean(tokens),
                          "median": statistics.median(tokens), "max": max(tokens)},
        "scalar_missingness": missing, "feature_contract": FEATURE_CONTRACT,
        "development_only": True,
    }
    lines = [f"# {CONDITION_ID} MMLU-Pro {ROLE} summary", "",
             f"- Rows: {expected}. Registered answer correct: {total(own)}/{expected} = "
             f"{total(own) / expected:.4f}; answer parseable on {parseable} rows.",
             f"- Official MMLU-Pro extraction with its random fallback removed: "
             f"{report['official_parser_correct_without_random']}/{expected} (for comparison only).",
             f"- Output tokens: sum {sum(tokens)}, mean {statistics.mean(tokens):.1f}, "
             f"max {max(tokens)}; truncated at {BUDGET}: {len(truncated)} rows, "
             f"{len(loops)} of them repetition loops (repeated 8-gram share >= 0.5).",
             f"- Scalar missingness: {json.dumps({k: v for k, v in missing.items() if v}, sort_keys=True)}"]
    if ROLE == "review":
        g_grades = {r["problem_id"]: r for r in read_jsonl(Path(args.g_grades)) if r["outcome"] == "u0"}
        g_letters = {r["problem_id"]: r["outputs"][0]["extracted_answer"]
                     for r in read_jsonl(Path(args.g_records))}
        g_correct = sum(int(g_grades[p]["u"]) for p in ids)
        repair = sum(int(grades["u_gv"][p]["u"]) and not int(g_grades[p]["u"]) for p in ids)
        harm = sum(not int(grades["u_gv"][p]["u"]) and int(g_grades[p]["u"]) for p in ids)
        fallback = sum(grades["u_gv"][p]["delivered_from"] == "g_fallback" for p in ids)
        relation = collections.Counter(
            "reviewer_unparseable" if records[p]["outputs"][0]["extracted_answer"] is None
            else "same_as_reviewed" if records[p]["outputs"][0]["extracted_answer"] == g_letters[p]
            else "changed" for p in ids)
        report.update({"gv_primary_correct": total("u_gv"), "gv_strict_correct": total("u_gv_strict"),
                       "reviewed_alone_correct_same_rows": g_correct, "gv_vs_reviewed_repair": repair,
                       "gv_vs_reviewed_harm": harm, "fallback_rows": fallback,
                       "answer_relation_to_reviewed": dict(relation)})
        lines += [f"- GV delivered correct (primary): {total('u_gv')}/{expected}; strict: "
                  f"{total('u_gv_strict')}/{expected}; fallback rows: {fallback}.",
                  f"- Reviewed model alone on the same rows: {g_correct}/{expected}; GV vs it "
                  f"repair/harm {repair}/{harm}.",
                  f"- Reviewer's letter relative to the reviewed model's: {dict(relation)}.",
                  "- The comparison with the reviewer alone and max(G, V) is a separate analysis over the archives."]
    lines += ["", "Evidence was written before grading. Grading is letter equality with the reference answer.", ""]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output.parent / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------ entry point

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-runnable")
    smoke = sub.add_parser("smoke")
    for name in ("--run-id", "--api-base", "--served-model", "--capture-sidecar"):
        smoke.add_argument(name, required=True)
    for command in ("generate", "review"):
        p = sub.add_parser(command)
        for name in ("--run-id", "--api-base", "--served-model", "--manifest", "--output",
                     "--capture-sidecar", "--logit-output", "--logit-run-id"):
            p.add_argument(name, required=True)
        p.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
        p.add_argument("--replicate-id", default=REPLICATE_ID)
        p.add_argument("--base-seed", type=int, default=BASE_SEED)
        if command == "review":
            p.add_argument("--g-records", required=True)
    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("records")
    reconcile.add_argument("traces")
    ev = sub.add_parser("evidence")
    for name in ("--run-id", "--records", "--logit-traces", "--output"):
        ev.add_argument(name, required=True)
    grade = sub.add_parser("grade")
    for name in ("--records", "--output"):
        grade.add_argument(name, required=True)
    grade.add_argument("--g-records")
    analyze = sub.add_parser("analyze")
    for name in ("--run-id", "--records", "--evidence", "--grades", "--output"):
        analyze.add_argument(name, required=True)
    analyze.add_argument("--g-records")
    analyze.add_argument("--g-grades")
    analyze.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    return parser


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] != "--condition-id":
        raise SystemExit(__doc__)
    configure(argv[1])
    args = build_parser().parse_args(argv[2:])
    if args.command == "check-runnable":
        model = require_registered(CONDITION_ID)
        print(canonical_json({"runnable": True, "condition_id": CONDITION_ID, "role": ROLE,
                              "model_id": model["model_id"], "revision": model["revision"],
                              "max_new_tokens": BUDGET, "dataset_id": DATASET_ID}))
    elif args.command == "smoke":
        stage_smoke(args)
    elif args.command == "generate":
        stage_generate(args)
    elif args.command == "review":
        stage_review(args)
    elif args.command == "reconcile":
        print(canonical_json({"stage": "reconcile",
                              **mathfull.reconcile(Path(args.records), Path(args.traces))}))
    elif args.command == "evidence":
        stage_evidence(args)
    elif args.command == "grade":
        if ROLE == "review" and not args.g_records:
            raise SystemExit("review grading needs --g-records")
        stage_grade(args)
    else:
        if ROLE == "review" and not (args.g_records and args.g_grades):
            raise SystemExit("review analysis needs --g-records and --g-grades")
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
