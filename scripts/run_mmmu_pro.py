#!/usr/bin/env python3
"""The MMMU-Pro vision study (docs/plan/vision_mmmu_pro_study.md): VL-G alone, VL-V alone and
VL-GV (V reviewing G) on the 1,000 dev rows. c036/c037/c038 at 4,096 output tokens are the
active conditions; c033/c034/c035 at 8,192 were stopped (plan section 9.3) and stay runnable
only as registered.

Usage: run_mmmu_pro.py --condition-id {c033,c034,c035,c036,c037,c038,c039} <command> [args]
(c039 is the reverse direction: the 4B reviewing c037's 8B outputs.)

Commands:
    check-runnable  the frozen prompt, model provenance, dataset and image-audit hashes,
                    decoding and seed cell are the registered ones, and c034/c035 differ
                    from their base only where registered
    smoke           one call on a generated test image: exact token alignment, a paired
                    same-forward capture, and the served <|image_pad|> count equal to what
                    the registered pixel limits imply
    generate        c033/c034: one solve call per row with same-forward capture; resumable
    review          c035: one review call per row on c033's G output; resumable
    reconcile       keep only rows present and valid in both the record and trace files
    evidence        vision_mc_scalar_v1 for the condition's own output, before grading
    grade           letter equality with the reference answer (and, for c035, both GV
                    delivery readings)
    analyze         reports/analysis.json and reports/summary.md

Every row's images are checked byte for byte against the dataset's image audit, and every
served prompt must hold exactly the <|image_pad|> count smart_resize gives under the
registered pixel limits; either failure stops the job rather than recording a row the
registration does not describe.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
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

from authority_allocation import derive_call_seed, mmmu_pro  # noqa: E402
from authority_allocation.api import api_call  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.prompts import split_prompt  # noqa: E402
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.serialization import (  # noqa: E402
    canonical_json, sha256_bytes, sha256_file, stable_id, utc_now,
)
from authority_allocation.vision_mc_features import (  # noqa: E402
    FEATURE_CONTRACT, derive_features,
)

DATASET_ID = "mmmupro-r563f3e8-std10-s20260926"
DATASET_PATH = ROOT / "datasets" / DATASET_ID / "dataset.yaml"
SOURCE_REPO = "MMMU/MMMU_Pro"
SOURCE_REVISION = "563f3e84bb3b90893083a1f039cfa13077f2302b"
SOURCE_CONFIG = "standard (10 options)"
REPLICATE_PLAN_ID = "r013"
REPLICATE_ID = "p01"
BASE_SEED = 20260926
EXPECTED_ROWS = 1000

CONDITIONS = {
    "c033": {"role": "solve", "model_id": "Qwen/Qwen3-VL-4B-Instruct",
             "prompt": mmmu_pro.VL_SOLVE_PROMPT},
    "c034": {"role": "solve", "model_id": "Qwen/Qwen3-VL-8B-Instruct",
             "prompt": mmmu_pro.VL_SOLVE_PROMPT, "base": "c033",
             "allowed_sections": {"condition_id", "status", "purpose", "gate", "runnable_gate",
                                  "model", "replicates"},
             "allowed_model_keys": {"model_id", "revision", "tokenizer_revision", "provenance",
                                    "server_path", "chat_template_hash"}},
    "c035": {"role": "review", "model_id": "Qwen/Qwen3-VL-8B-Instruct",
             "prompt": mmmu_pro.VL_REVIEW_PROMPT, "base": "c034", "g": "c033",
             "allowed_sections": {"condition_id", "status", "purpose", "gate", "runnable_gate",
                                  "model", "replicates", "protocol", "generator_source",
                                  "delivery", "stages"},
             "allowed_model_keys": {"prompt_pass1", "prompt_pass1_provenance", "prompt_inputs"}},
    "c036": {"role": "solve", "model_id": "Qwen/Qwen3-VL-4B-Instruct",
             "prompt": mmmu_pro.VL_SOLVE_PROMPT, "base": "c033", "budget_only": True,
             "allowed_sections": {"condition_id", "status", "purpose", "gate", "runnable_gate",
                                  "model", "replicates"},
             "allowed_model_keys": {"pass1_decoding"}},
    "c037": {"role": "solve", "model_id": "Qwen/Qwen3-VL-8B-Instruct",
             "prompt": mmmu_pro.VL_SOLVE_PROMPT, "base": "c036",
             "allowed_sections": {"condition_id", "status", "purpose", "gate", "runnable_gate",
                                  "model", "replicates"},
             "allowed_model_keys": {"model_id", "revision", "tokenizer_revision", "provenance",
                                    "server_path", "chat_template_hash"}},
    "c038": {"role": "review", "model_id": "Qwen/Qwen3-VL-8B-Instruct",
             "prompt": mmmu_pro.VL_REVIEW_PROMPT, "base": "c037", "g": "c036",
             "allowed_sections": {"condition_id", "status", "purpose", "gate", "runnable_gate",
                                  "model", "replicates", "protocol", "generator_source",
                                  "delivery", "stages"},
             "allowed_model_keys": {"prompt_pass1", "prompt_pass1_provenance", "prompt_inputs"}},
}
CONDITIONS["c039"] = {"role": "review", "model_id": "Qwen/Qwen3-VL-4B-Instruct",
                     "prompt": mmmu_pro.VL_REVIEW_PROMPT, "base": "c036", "g": "c037",
                     "allowed_sections": CONDITIONS["c038"]["allowed_sections"],
                     "allowed_model_keys": {"prompt_pass1", "prompt_pass1_provenance", "prompt_inputs"}}
BUDGETS = {"c033": 8192, "c034": 8192, "c035": 8192, "c036": 4096, "c037": 4096, "c038": 4096,
           "c039": 4096}
PROVENANCE_FIELDS = ("revision", "tokenizer_revision", "server_path", "chat_template_hash")
CONDITION_ID = "c033"
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
    shared.STAGE = "mmmu-pro-" + ROLE
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
    mmmu_pro.require_frozen_prompt(ROOT, model["prompt_pass1"])
    if model["model_id"] != spec["model_id"]:
        raise SystemExit(f"{condition_id} must run {spec['model_id']}")
    provenance = json.loads((ROOT / model["provenance"]).read_text(encoding="utf-8"))
    if provenance["repo_id"] != model["model_id"]:
        raise SystemExit(f"{condition_id} provenance is for {provenance['repo_id']}")
    for field in PROVENANCE_FIELDS:
        if model.get(field) != provenance[field]:
            raise SystemExit(f"{condition_id} model.{field} disagrees with {model['provenance']}")
    decoding = model["pass1_decoding"]
    if float(decoding["temperature"]) != 0.0 or model.get("thinking") is not False:
        raise SystemExit(f"{condition_id} requires temperature 0 and thinking false")
    if int(decoding["max_new_tokens"]) != BUDGETS[condition_id]:
        raise SystemExit(f"{condition_id} output budget must be {BUDGETS[condition_id]}")
    if float(decoding["repetition_penalty"]) != 1.0 or float(decoding["presence_penalty"]) != 0.0:
        raise SystemExit(f"{condition_id} must send repetition_penalty 1.0 and presence_penalty 0.0")
    registered = registered_dataset()
    manifest = registered["manifests"][cond["dataset"]["manifest"]]
    if (cond["dataset"]["id"] != DATASET_ID or cond["dataset"]["manifest"] != "dev"
            or cond["dataset"]["manifest_rows"] != manifest["rows"]
            or manifest["rows"] != EXPECTED_ROWS
            or cond["dataset"]["manifest_sha256"] != manifest["sha256"]
            or cond["dataset"]["image_audit_sha256"] != registered["image_audit"]["sha256"]):
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
        if spec.get("budget_only"):
            ours, theirs = dict(model["pass1_decoding"]), dict(parent["model"]["pass1_decoding"])
            ours.pop("max_new_tokens"), theirs.pop("max_new_tokens")
            if ours != theirs:
                raise SystemExit(f"{condition_id} decoding differs from {spec['base']} beyond the budget")
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


def image_audit() -> dict[tuple[str, int], dict[str, Any]]:
    """The dataset's image audit, verified against its registered hash."""
    registered = registered_dataset()["image_audit"]
    path = Path(registered["server_path"])
    if sha256_file(path) != registered["sha256"]:
        raise SystemExit(f"{path} does not match the registered image-audit hash")
    return {(row["problem_id"], int(row["image_number"])): row for row in read_jsonl(path)}


def load_mmmu_pro_rows(problem_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Join an id-only manifest to the pinned source, images as original bytes."""
    from datasets import Image as ImageFeature, load_dataset

    wanted = set(problem_ids)
    source = load_dataset(SOURCE_REPO, SOURCE_CONFIG, revision=SOURCE_REVISION, split="test")
    for n in range(1, mmmu_pro.MAX_IMAGES + 1):
        source = source.cast_column(f"image_{n}", ImageFeature(decode=False))
    rows: dict[str, dict[str, Any]] = {}
    for row in source:
        pid = str(row["id"])
        if pid not in wanted:
            continue
        if pid in rows:
            raise ValueError(f"duplicate source row {pid}")
        options = mmmu_pro.parse_options(row["options"])
        images = {}
        for n in range(1, mmmu_pro.MAX_IMAGES + 1):
            if row[f"image_{n}"] and row[f"image_{n}"].get("bytes"):
                images[n] = row[f"image_{n}"]["bytes"]
        rows[pid] = {"question": row["question"], "options": options,
                     "answer": str(row["answer"]).strip(), "subject": str(row["subject"]),
                     "images": images}
    missing = wanted.difference(rows)
    if missing:
        raise ValueError(f"manifest ids missing from the pinned source: {sorted(missing)[:5]}")
    return rows


def visual_input(problem_id: str, numbers: list[int], images: Mapping[int, bytes],
                 audit: Mapping[tuple[str, int], Mapping[str, Any]],
                 limits: Mapping[str, Any]) -> tuple[dict[str, Any], dict[int, tuple[bytes, str]]]:
    """Check each image against the audit and compute the visual tokens it must become."""
    shas, sendable, expected = [], {}, 0
    for n in numbers:
        entry = audit.get((problem_id, n))
        data = images.get(n)
        if entry is None or data is None:
            raise RuntimeError(f"{problem_id} image_{n} is not in the audit or the source")
        digest = sha256_bytes(data)
        if digest != entry["sha256"]:
            raise RuntimeError(f"{problem_id} image_{n} bytes differ from the image audit")
        shas.append(digest)
        sendable[n] = (data, entry["format"])
        expected += mmmu_pro.visual_tokens(int(entry["height"]), int(entry["width"]),
                                           min_pixels=int(limits["min_pixels"]),
                                           max_pixels=int(limits["max_pixels"]))
    return ({"image_numbers": list(numbers), "image_sha256": shas, "num_images": len(numbers),
             "expected_visual_tokens": expected, "num_visual_tokens": None,
             "min_pixels": int(limits["min_pixels"]), "max_pixels": int(limits["max_pixels"])},
            sendable)


# ------------------------------------------------------------------ model calls

def model_identity(model: Mapping[str, Any]) -> dict[str, Any]:
    return {"model_id": model["model_id"], "revision": model["revision"],
            "server_path": model["server_path"], "backend": model["backend"],
            "prompt_sha256": sha256_file(ROOT / model["prompt_pass1"]),
            "chat_template_sha256": model["chat_template_hash"], "dtype": model["dtype"],
            "quantization": model["quantization"]}


def build_messages(model: Mapping[str, Any], question: str, options: list[str],
                   y0: str | None = None) -> tuple[list[dict[str, Any]], list[int]]:
    system, user = split_prompt(ROOT / model["prompt_pass1"])
    parts = mmmu_pro.user_content_parts(user, question=question, options=options, y0=y0)
    return ([{"role": "system", "content": system}, {"role": "user", "content": parts}],
            mmmu_pro.image_numbers_in_order(parts))


def call_model(args: argparse.Namespace, model: Mapping[str, Any], messages: list[dict[str, Any]],
               images: Mapping[int, tuple[bytes, str]], seed: int,
               capture_id: str) -> tuple[str, dict[str, Any], int | None]:
    wire = [messages[0], {"role": "user",
                          "content": mmmu_pro.openai_user_content(messages[1]["content"], images)}]
    content, _, usage = api_call(
        api_base=args.api_base, served_model=args.served_model, messages=wire,
        decoding={**model["pass1_decoding"],
                  "output_logprobs": {"enabled": True, "mode": "raw_logprobs",
                                      "return_token_ids": True, "return_prompt_token_ids": True},
                  "vllm_xargs": {"c014_capture_id": capture_id}},
        seed=seed, thinking=False)
    prompt_ids = usage.pop("prompt_token_ids", None)
    served = mmmu_pro.count_token(prompt_ids, int(model["image_processing"]["image_pad_token_id"]))
    return content, usage, served


def smoke_image() -> tuple[bytes, int, int]:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 448), "white")
    ImageDraw.Draw(image).rectangle((200, 120, 440, 330), fill=(220, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), 448, 640


def stage_smoke(args: argparse.Namespace) -> None:
    model = require_registered(CONDITION_ID)
    data, height, width = smoke_image()
    limits = model["image_processing"]
    y0 = "The square is red.\nAnswer: A" if ROLE == "review" else None
    messages, numbers = build_messages(model, "What color is the square in <image 1>?",
                                       ["red", "blue", "green"], y0)
    capture_id = stable_id(args.run_id, "smoke", "same_forward_logit_capture")
    content, usage, served = call_model(args, model, messages, {1: (data, "PNG")}, 1, capture_id)
    expected = mmmu_pro.visual_tokens(height, width, min_pixels=int(limits["min_pixels"]),
                                      max_pixels=int(limits["max_pixels"]))
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
    if served is None:
        problems.append("no prompt_token_ids in the response")
    elif served != expected:
        problems.append(f"served {served} visual tokens, registered limits imply {expected}")
    if native["status"] != "ok":
        problems.append(f"native capture {native['status']}")
    print(canonical_json({"stage": "smoke", "condition_id": CONDITION_ID, "served_visual_tokens": served,
                          "expected_visual_tokens": expected, "native_status": native["status"],
                          "answer": mmmu_pro.parse_answer_letter(content, 3),
                          "finish_reason": usage.get("finish_reason"), "problems": problems}))
    if problems:
        raise SystemExit("smoke failed: " + "; ".join(problems))


def _record_base(args: argparse.Namespace, problem_id: str, row: Mapping[str, Any],
                 manifest_row: Mapping[str, Any], model: Mapping[str, Any], kind: str) -> dict[str, Any]:
    return {"record_id": stable_id(args.run_id, problem_id, kind),
            "run_id": args.run_id, "condition_id": CONDITION_ID,
            "condition_hash": shared.condition_hash(), "dataset_id": DATASET_ID,
            "replicate_id": args.replicate_id, "stage": shared.STAGE,
            "problem_id": problem_id, "problem_text": row["question"],
            "problem_metadata": {"subject": manifest_row["subject"], "origin": manifest_row["origin"],
                                 "n_options": int(manifest_row["n_options"]),
                                 "n_images": int(manifest_row["n_images"])},
            "generator": model_identity(model)}


def _run_rows(args: argparse.Namespace, *, g_rows: Mapping[str, Mapping[str, Any]] | None) -> None:
    model = require_registered(CONDITION_ID)
    manifest = manifest_rows(Path(args.manifest), int(args.expected_rows))
    ids = [str(r["problem_id"]) for r in manifest]
    by_id = {str(r["problem_id"]): r for r in manifest}
    sources = load_mmmu_pro_rows(ids)
    audit = image_audit()
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
        y0 = None
        if g_rows is not None:
            y0 = g_rows[problem_id]["outputs"][0]["raw_output"] or ""
        messages, numbers = build_messages(model, row["question"], row["options"], y0)
        if len(numbers) != int(manifest_row["n_images"]):
            raise RuntimeError(f"{problem_id} image count differs from the manifest")
        vis, images = visual_input(problem_id, numbers, row["images"], audit, model["image_processing"])
        seed = derive_call_seed(REPLICATE_PLAN_ID, scope, args.replicate_id, int(args.base_seed),
                                CONDITION_ID, problem_id, 0)
        capture_id = stable_id(args.run_id, problem_id, "same_forward_logit_capture")
        content, usage, served = call_model(args, model, messages, images, seed, capture_id)
        if served != vis["expected_visual_tokens"]:
            raise RuntimeError(f"{problem_id}: served {served} visual tokens, registered limits "
                               f"imply {vis['expected_visual_tokens']}")
        vis["num_visual_tokens"] = served
        trace = usage.pop("token_logprobs", None)
        letter = mmmu_pro.parse_answer_letter(content, len(row["options"]))
        record = {
            "schema_version": "vl-review-record-v1" if g_rows is not None else "vl-record-v1",
            **_record_base(args, problem_id, row, manifest_row, model,
                           "review" if g_rows is not None else "pass1"),
            "outputs": [{"sample_index": 0, "call_seed": seed, "raw_output": content,
                         "extracted_answer": letter,
                         "parse_status": "ok" if letter is not None else "model_invalid",
                         "token_logprobs": trace, "usage": shared.base.compact_usage(usage)}],
            "visual_input": vis, "y0_sample_index": 0, "created_at_utc": utc_now(),
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
        raise RuntimeError(f"G records come from {foreign}, not the registered {source_run}")
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
        meta = record["problem_metadata"]
        features, audit = derive_features(
            output=record["outputs"][0], native=native, n_options=int(meta["n_options"]),
            n_images=int(meta["n_images"]), question_text=record["problem_text"],
            num_visual_tokens=record["visual_input"]["num_visual_tokens"])
        append_jsonl(output, {
            "schema_version": "vl-evidence-record-v1",
            "record_id": stable_id(args.run_id, pid, "vision-evidence"), "run_id": args.run_id,
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
    sources = load_mmmu_pro_rows([r["problem_id"] for r in records])
    g_rows = ({r["problem_id"]: r for r in read_jsonl(Path(args.g_records))}
              if ROLE == "review" else {})
    output = Path(args.output)
    output.unlink(missing_ok=True)
    for record in records:
        pid, out = record["problem_id"], record["outputs"][0]
        gold, options = sources[pid]["answer"], sources[pid]["options"]
        own = "u_review" if ROLE == "review" else "u0"
        official = mmmu_pro.official_parse_without_random(out["raw_output"], options)
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
    report: dict[str, Any] = {
        "condition_id": CONDITION_ID, "run_id": args.run_id, "role": ROLE, "rows": expected,
        "correct": total(own), "accuracy": total(own) / expected,
        "official_parser_correct_without_random": total(own + "_official_parser"),
        "answer_parseable": parseable,
        "truncated_rows": len(truncated), "truncated_repetition_loops": len(loops),
        "truncated_problem_ids": truncated,
        "output_tokens": {"sum": sum(tokens), "mean": statistics.mean(tokens),
                          "median": statistics.median(tokens), "max": max(tokens)},
        "visual_tokens": {"sum": sum(records[p]["visual_input"]["num_visual_tokens"] for p in ids)},
        "scalar_missingness": missing, "feature_contract": FEATURE_CONTRACT,
        "development_only": True,
    }
    lines = [f"# {CONDITION_ID} MMMU-Pro {ROLE} summary", "",
             f"- Rows: {expected}. Registered answer correct: {total(own)}/{expected} = "
             f"{total(own) / expected:.4f}; answer parseable on {parseable} rows.",
             f"- Official MMMU-Pro parser with its random fallback removed: "
             f"{report['official_parser_correct_without_random']}/{expected} (for comparison only).",
             f"- Output tokens: sum {sum(tokens)}, mean {statistics.mean(tokens):.1f}, "
             f"max {max(tokens)}; truncated at {BUDGETS[CONDITION_ID]}: {len(truncated)} rows, "
             f"{len(loops)} of them repetition loops (repeated 8-gram share >= 0.5).",
             f"- Scalar missingness: {json.dumps({k: v for k, v in missing.items() if v}, sort_keys=True)}"]
    if ROLE == "review":
        g_grades = {r["problem_id"]: r for r in read_jsonl(Path(args.g_grades)) if r["outcome"] == "u0"}
        g_correct = sum(int(g_grades[p]["u"]) for p in ids)
        repair = sum(int(grades["u_gv"][p]["u"]) and not int(g_grades[p]["u"]) for p in ids)
        harm = sum(not int(grades["u_gv"][p]["u"]) and int(g_grades[p]["u"]) for p in ids)
        fallback = sum(grades["u_gv"][p]["delivered_from"] == "g_fallback" for p in ids)
        g_letters = {p: read_g_letter(args, p) for p in ids}
        relation = collections.Counter(
            "v_unparseable" if records[p]["outputs"][0]["extracted_answer"] is None
            else "same_as_g" if records[p]["outputs"][0]["extracted_answer"] == g_letters[p]
            else "changed" for p in ids)
        report.update({"gv_primary_correct": total("u_gv"), "gv_strict_correct": total("u_gv_strict"),
                       "g_alone_correct_same_rows": g_correct, "gv_vs_g_repair": repair,
                       "gv_vs_g_harm": harm, "g_fallback_rows": fallback,
                       "answer_relation_to_g": dict(relation)})
        lines += [f"- GV delivered correct (primary, G when V is unparseable): {total('u_gv')}/{expected}; "
                  f"strict (V unparseable is wrong): {total('u_gv_strict')}/{expected}; "
                  f"fallback rows: {fallback}.",
                  f"- G alone on the same rows: {g_correct}/{expected}; GV vs G repair/harm {repair}/{harm}.",
                  f"- Reviewer's letter relative to G's: {dict(relation)}.",
                  "- The comparison with VL-V alone (c034) and max(G, V) is a separate analysis over the archives."]
    lines += ["", "Evidence was written before grading. Grading is letter equality with the reference answer.", ""]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output.parent / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


_G_LETTERS: dict[str, str | None] = {}


def read_g_letter(args: argparse.Namespace, problem_id: str) -> str | None:
    if not _G_LETTERS:
        for row in read_jsonl(Path(args.g_records)):
            _G_LETTERS[row["problem_id"]] = row["outputs"][0]["extracted_answer"]
    return _G_LETTERS[problem_id]


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
                              "max_new_tokens": BUDGETS[CONDITION_ID], "dataset_id": DATASET_ID}))
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
