#!/usr/bin/env python3
"""The GV system on mathfull dev1000: Qwen3-8B reviews Qwen3-4B's solutions.

c024 reviews c022's G with the plain reviewer prompt v1. c027 reviews c025's G with the
frozen review prompt v2 (docs/plan/gv_vs_max_g_v_study.md section 4.7). c028 swaps the
models under the same prompt: Qwen3-4B (c025's identity) reviews c026's Qwen3-8B outputs.

Usage: run_mathfull_review.py [--condition-id {c024,c027,c028,c042,c043,c044-c053,c058-c061,c064,c065,c090,c091,c094,c095,c150-c167,c210-c253,c256-c258,c260-c269,c386,c387,c390,c391}] <command> [args]
(the condition defaults to c024, the only one before c027 existed)

Commands:
    check-runnable  the condition differs from its base only where registered (c024 from
                    c023; c027 from c024: prompt, generator source, evidence schema, seeds;
                    c028 from c027: reviewer identity, which must equal c025's, generator
                    source, evidence schema, seeds),
                    the prompt takes the problem and G's output, and a frozen prompt still
                    has its registered hash
    smoke           one reviewer call on a toy problem, with an exact content trace
    review          one reviewer call per problem with same-forward logit capture; resumable
    reconcile       as in run_mathfull_solo.py, on review.jsonl and its native trace
    evidence        c017's seven scalars on the reviewer's own output
    grade           u_review (reviewer's answer) and u_gv (delivered: the reviewer's answer,
                    or G's output if the reviewer's yields no answer)
    analyze         accuracy, fallbacks and answer changes, over every row except the
                    registered ungradeable-gold problems; any other grader_error stops it
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c002_dev30 as base  # noqa: E402
import run_c014_router as shared  # noqa: E402
import run_c017_math_expression_scalars as c017  # noqa: E402
import run_mathfull_solo as solo  # noqa: E402

from authority_allocation import derive_call_seed  # noqa: E402
from authority_allocation.api import api_call, chat_template_kwargs_for  # noqa: E402
from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
from authority_allocation.prompts import PROMPT_VARIABLE, render, split_prompt  # noqa: E402
from authority_allocation.records import append_jsonl, completed_keys, read_jsonl  # noqa: E402
from authority_allocation.serialization import canonical_json, sha256_bytes, stable_id, utc_now  # noqa: E402

CONDITION_ID = "c024"
# base: the condition it overlays; g: the solo condition whose outputs are reviewed; v: the
# solo run of the reviewer's own model; model_from: where a swapped reviewer's identity comes
# from (default base).
REVIEW_CONDITIONS = {
    "c024": {"base": "c023", "g": "c022", "v": "c023"},
    "c027": {"base": "c024", "g": "c025", "v": "c026", "frozen_prompt": True},
    "c028": {"base": "c027", "g": "c026", "v": "c025", "frozen_prompt": True, "model_from": "c025"},
    # The confirmatory test runs (docs/plan/test_mathfull_v2.md): c027/c028 on the sealed test
    # manifest, reviewing the test-set solo runs c040/c041.
    "c042": {"base": "c027", "g": "c040", "v": "c041", "frozen_prompt": True, "test": True},
    "c043": {"base": "c028", "g": "c041", "v": "c040", "frozen_prompt": True, "test": True},
    # The Llama-3.2-3B / Qwen2.5-3B test (docs/plan/test_llama_qwen25.md): c044/c045 on the
    # first 1,500 test rows, reviewing the test-set solo runs c056/c057.
    "c058": {"base": "c044", "g": "c056", "v": "c057", "frozen_prompt": True, "test": True},
    "c059": {"base": "c045", "g": "c057", "v": "c056", "frozen_prompt": True, "test": True},
    # The Qwen3-4B / Qwen2.5-7B pair (2026-09-27): both directions on dev1000, and
    # both on the first 1,500 test rows (docs/plan/test_qwen3_4b_qwen25_7b.md).
    "c060": {"base": "c027", "g": "c025", "v": "c031", "frozen_prompt": True, "model_from": "c031"},
    "c061": {"base": "c028", "g": "c031", "v": "c025", "frozen_prompt": True},
    "c064": {"base": "c060", "g": "c062", "v": "c063", "frozen_prompt": True, "test": True},
    "c065": {"base": "c061", "g": "c063", "v": "c062", "frozen_prompt": True, "test": True},
    # The Qwen3-8B / Mistral-Small-24B pair (2026-09-28): both directions on
    # dev1000, and both on the first 1,500 test rows (docs/plan/test_qwen3_8b_mistral24b.md).
    "c090": {"base": "c027", "g": "c026", "v": "c054", "frozen_prompt": True, "model_from": "c054"},
    "c091": {"base": "c027", "g": "c054", "v": "c026", "frozen_prompt": True},
    "c094": {"base": "c090", "g": "c092", "v": "c093", "frozen_prompt": True, "test": True},
    "c095": {"base": "c091", "g": "c093", "v": "c092", "frozen_prompt": True, "test": True},
    # The Ministral-8B / Qwen2.5-3B pair, the weak-weak stratum (2026-09-28): both
    # directions on dev1000 and on the first 1,500 test rows (docs/plan/test_pair_baselines.md).
    "c150": {"base": "c027", "g": "c032", "v": "c030", "frozen_prompt": True, "model_from": "c030"},
    "c151": {"base": "c027", "g": "c030", "v": "c032", "frozen_prompt": True, "model_from": "c032"},
    "c152": {"base": "c150", "g": "c096", "v": "c057", "frozen_prompt": True, "test": True},
    "c153": {"base": "c151", "g": "c057", "v": "c096", "frozen_prompt": True, "test": True},
    # The Qwen2.5-7B / Qwen3-8B pair, the middle-middle stratum (2026-09-28).
    "c154": {"base": "c027", "g": "c031", "v": "c026", "frozen_prompt": True},
    "c155": {"base": "c027", "g": "c026", "v": "c031", "frozen_prompt": True, "model_from": "c031"},
    "c156": {"base": "c154", "g": "c063", "v": "c092", "frozen_prompt": True, "test": True},
    "c157": {"base": "c155", "g": "c092", "v": "c063", "frozen_prompt": True, "test": True},
    # Strong-strong, strong-middle, strong-weak and middle-weak pairs on dev1000
    # (2026-09-29; test conditions are registered separately once the solo tests are checked).
    "c158": {"base": "c027", "g": "c055", "v": "c054", "frozen_prompt": True, "model_from": "c054"},
    "c159": {"base": "c027", "g": "c054", "v": "c055", "frozen_prompt": True, "model_from": "c055"},
    "c160": {"base": "c027", "g": "c031", "v": "c055", "frozen_prompt": True, "model_from": "c055"},
    "c161": {"base": "c027", "g": "c055", "v": "c031", "frozen_prompt": True, "model_from": "c031"},
    "c162": {"base": "c027", "g": "c030", "v": "c054", "frozen_prompt": True, "model_from": "c054"},
    "c163": {"base": "c027", "g": "c054", "v": "c030", "frozen_prompt": True, "model_from": "c030"},
    "c164": {"base": "c027", "g": "c032", "v": "c055", "frozen_prompt": True, "model_from": "c055"},
    "c165": {"base": "c027", "g": "c055", "v": "c032", "frozen_prompt": True, "model_from": "c032"},
    "c166": {"base": "c027", "g": "c030", "v": "c025", "frozen_prompt": True, "model_from": "c025"},
    "c167": {"base": "c027", "g": "c025", "v": "c030", "frozen_prompt": True, "model_from": "c030"},
    # The stratified pairs on the re-split development set testpool_dev1000 (decided
    # 2026-09-30, docs/plan/testpool_dev1000.md section 8): dev_pool allows only the change to that
    # 1,000-row development manifest; the reviewed outputs are the testpool_dev1000 solos c200-c208.
    "c210": {"base": "c027", "g": "c205", "v": "c200", "frozen_prompt": True, "dev_pool": True, "model_from": "c025"},
    "c211": {"base": "c027", "g": "c200", "v": "c205", "frozen_prompt": True, "dev_pool": True, "model_from": "c054"},
    "c212": {"base": "c027", "g": "c202", "v": "c206", "frozen_prompt": True, "dev_pool": True, "model_from": "c055"},
    "c213": {"base": "c027", "g": "c206", "v": "c202", "frozen_prompt": True, "dev_pool": True, "model_from": "c030"},
    "c214": {"base": "c027", "g": "c204", "v": "c201", "frozen_prompt": True, "dev_pool": True},
    "c215": {"base": "c027", "g": "c201", "v": "c204", "frozen_prompt": True, "dev_pool": True, "model_from": "c032"},
    "c216": {"base": "c027", "g": "c207", "v": "c201", "frozen_prompt": True, "dev_pool": True},
    "c217": {"base": "c027", "g": "c201", "v": "c207", "frozen_prompt": True, "dev_pool": True, "model_from": "c067"},
    "c218": {"base": "c027", "g": "c205", "v": "c203", "frozen_prompt": True, "dev_pool": True, "model_from": "c031"},
    "c219": {"base": "c027", "g": "c203", "v": "c205", "frozen_prompt": True, "dev_pool": True, "model_from": "c054"},
    "c220": {"base": "c027", "g": "c202", "v": "c203", "frozen_prompt": True, "dev_pool": True, "model_from": "c031"},
    "c221": {"base": "c027", "g": "c203", "v": "c202", "frozen_prompt": True, "dev_pool": True, "model_from": "c030"},
    "c222": {"base": "c027", "g": "c204", "v": "c202", "frozen_prompt": True, "dev_pool": True, "model_from": "c030"},
    "c223": {"base": "c027", "g": "c202", "v": "c204", "frozen_prompt": True, "dev_pool": True, "model_from": "c032"},
    "c224": {"base": "c027", "g": "c208", "v": "c203", "frozen_prompt": True, "dev_pool": True, "model_from": "c031"},
    "c225": {"base": "c027", "g": "c203", "v": "c208", "frozen_prompt": True, "dev_pool": True, "model_from": "c029"},
    "c226": {"base": "c027", "g": "c208", "v": "c207", "frozen_prompt": True, "dev_pool": True, "model_from": "c067"},
    "c227": {"base": "c027", "g": "c207", "v": "c208", "frozen_prompt": True, "dev_pool": True, "model_from": "c029"},
    "c228": {"base": "c027", "g": "c208", "v": "c204", "frozen_prompt": True, "dev_pool": True, "model_from": "c032"},
    "c229": {"base": "c027", "g": "c204", "v": "c208", "frozen_prompt": True, "dev_pool": True, "model_from": "c029"},
    # Their runs on the first 1,500 test rows, reviewing the test-set solos.
    "c230": {"base": "c210", "g": "c093", "v": "c062", "frozen_prompt": True, "test": True},
    "c231": {"base": "c211", "g": "c062", "v": "c093", "frozen_prompt": True, "test": True},
    "c232": {"base": "c212", "g": "c057", "v": "c097", "frozen_prompt": True, "test": True},
    "c233": {"base": "c213", "g": "c097", "v": "c057", "frozen_prompt": True, "test": True},
    "c234": {"base": "c214", "g": "c096", "v": "c092", "frozen_prompt": True, "test": True},
    "c235": {"base": "c215", "g": "c092", "v": "c096", "frozen_prompt": True, "test": True},
    "c236": {"base": "c216", "g": "c068", "v": "c092", "frozen_prompt": True, "test": True},
    "c237": {"base": "c217", "g": "c092", "v": "c068", "frozen_prompt": True, "test": True},
    "c238": {"base": "c218", "g": "c093", "v": "c063", "frozen_prompt": True, "test": True},
    "c239": {"base": "c219", "g": "c063", "v": "c093", "frozen_prompt": True, "test": True},
    "c240": {"base": "c220", "g": "c057", "v": "c063", "frozen_prompt": True, "test": True},
    "c241": {"base": "c221", "g": "c063", "v": "c057", "frozen_prompt": True, "test": True},
    "c242": {"base": "c224", "g": "c056", "v": "c063", "frozen_prompt": True, "test": True},
    "c243": {"base": "c225", "g": "c063", "v": "c056", "frozen_prompt": True, "test": True},
    "c244": {"base": "c226", "g": "c056", "v": "c068", "frozen_prompt": True, "test": True},
    "c245": {"base": "c227", "g": "c068", "v": "c056", "frozen_prompt": True, "test": True},
    "c246": {"base": "c228", "g": "c056", "v": "c096", "frozen_prompt": True, "test": True},
    "c247": {"base": "c229", "g": "c096", "v": "c056", "frozen_prompt": True, "test": True},
    # The two strong-strong pairs, each substituted for a pair containing Qwen2.5-72B (decided
    # 2026-10-01, docs/plan/testpool_dev1000.md section 8); Qwen3-4B / Qwen3-8B is tested by c042/c043.
    "c248": {"base": "c027", "g": "c200", "v": "c201", "frozen_prompt": True, "dev_pool": True},
    "c249": {"base": "c027", "g": "c201", "v": "c200", "frozen_prompt": True, "dev_pool": True, "model_from": "c025"},
    "c250": {"base": "c027", "g": "c201", "v": "c206", "frozen_prompt": True, "dev_pool": True, "model_from": "c055"},
    "c251": {"base": "c027", "g": "c206", "v": "c201", "frozen_prompt": True, "dev_pool": True},
    "c252": {"base": "c250", "g": "c092", "v": "c097", "frozen_prompt": True, "test": True},
    "c253": {"base": "c251", "g": "c097", "v": "c092", "frozen_prompt": True, "test": True},
    # The OlympiadBench pair (docs/plan/olympiadbench_extension.md): c214/c215 on that dataset's
    # manifest, reviewing its solos c254 (Ministral-8B) and c255 (Qwen3-8B).
    "c256": {"base": "c214", "g": "c254", "v": "c255", "frozen_prompt": True, "extension": True},
    "c257": {"base": "c215", "g": "c255", "v": "c254", "frozen_prompt": True, "extension": True},
    "c260": {"base": "c248", "g": "c259", "v": "c255", "frozen_prompt": True, "extension": True},
    "c261": {"base": "c249", "g": "c255", "v": "c259", "frozen_prompt": True, "extension": True},
    # The two original strong-strong pairs, with Qwen2.5-72B (decided 2026-10-02, after the
    # substitutes c248-c253; docs/plan/testpool_dev1000.md section 8). 72B reviews run on two GPUs.
    "c262": {"base": "c027", "g": "c209", "v": "c200", "frozen_prompt": True, "dev_pool": True, "model_from": "c025"},
    "c263": {"base": "c027", "g": "c200", "v": "c209", "frozen_prompt": True, "dev_pool": True, "model_from": "c066"},
    "c264": {"base": "c027", "g": "c209", "v": "c206", "frozen_prompt": True, "dev_pool": True, "model_from": "c055"},
    "c265": {"base": "c027", "g": "c206", "v": "c209", "frozen_prompt": True, "dev_pool": True, "model_from": "c066"},
    "c266": {"base": "c262", "g": "c098", "v": "c062", "frozen_prompt": True, "test": True},
    "c267": {"base": "c263", "g": "c062", "v": "c098", "frozen_prompt": True, "test": True},
    "c268": {"base": "c264", "g": "c098", "v": "c097", "frozen_prompt": True, "test": True},
    "c269": {"base": "c265", "g": "c097", "v": "c098", "frozen_prompt": True, "test": True},
    # Self-review on test1500 (2026-10-09, docs/plan/self_review_test1500.md): the reviewer
    # of c236 / c230 reads its own test solo instead of the other model's.
    "c386": {"base": "c216", "g": "c092", "v": "c092", "frozen_prompt": True, "test": True},
    "c387": {"base": "c210", "g": "c062", "v": "c062", "frozen_prompt": True, "test": True},
    # Prompt-variant sanity check (2026-10-09, docs/plan/prompt_variant_test500.md):
    # c236 / c253 with only the review prompt changed to the frozen commit-first prompt.
    "c390": {"base": "c236", "g": "c068", "v": "c092", "frozen_prompt": True, "test": True,
             "prompt": "conditions/c390/prompts/v_commit_first_v1.txt"},
    "c391": {"base": "c253", "g": "c097", "v": "c092", "frozen_prompt": True, "test": True,
             "prompt": "conditions/c390/prompts/v_commit_first_v1.txt"},
    # Replaces c244, which stops deterministically at its 64th row (see REPAIR_RULE): c244 with
    # only the special-token bytes repair added (decided 2026-10-01).
    "c258": {"base": "c244", "g": "c056", "v": "c068", "frozen_prompt": True, "test": True,
             "special_token_repair": True},
    # Two new pairs, each in both directions (2026-09-26): Llama-3.2-3B with
    # Qwen2.5-3B, and Llama-3.2-3B with Ministral-8B. The reviewer is the other solo model.
    "c044": {"base": "c027", "g": "c029", "v": "c030", "frozen_prompt": True, "model_from": "c030"},
    "c045": {"base": "c027", "g": "c030", "v": "c029", "frozen_prompt": True, "model_from": "c029"},
    "c046": {"base": "c027", "g": "c029", "v": "c032", "frozen_prompt": True, "model_from": "c032"},
    "c047": {"base": "c027", "g": "c032", "v": "c029", "frozen_prompt": True, "model_from": "c029"},
    "c048": {"base": "c027", "g": "c029", "v": "c026", "frozen_prompt": True, "model_from": "c026"},
    "c049": {"base": "c027", "g": "c026", "v": "c029", "frozen_prompt": True, "model_from": "c029"},
    # Self-review: the reviewer reviews its own solo outputs.
    "c050": {"base": "c027", "g": "c032", "v": "c032", "frozen_prompt": True, "model_from": "c032"},
    "c051": {"base": "c027", "g": "c026", "v": "c026", "frozen_prompt": True, "model_from": "c026"},
    "c052": {"base": "c027", "g": "c032", "v": "c026", "frozen_prompt": True, "model_from": "c026"},
    "c053": {"base": "c027", "g": "c026", "v": "c032", "frozen_prompt": True, "model_from": "c032"},
}
STAGE = "gv-review"
ALLOWED_SECTIONS = {"condition_id", "status", "purpose", "gate", "runnable_gate", "protocol",
                    "generator_source", "model", "delivery", "replicates", "stages"}
ALLOWED_MODEL_KEYS = {"prompt_pass1", "prompt_pass1_provenance", "prompt_inputs"}
# c258's registered repair (decided 2026-10-01): c244 stopped at its 64th row because
# Llama-3.1-8B emitted <|start_header_id|> mid-answer; vLLM drops special tokens from the returned
# text but returns their bytes in the token trace, so the two never align.
REPAIR_KEY, REPAIR_RULE = "special_token_bytes_repair", "drop_special_token_bytes_v1"
REPAIRS_FILE = "token_byte_repairs.jsonl"


def configure(condition_id: str | None = None) -> None:
    global CONDITION_ID
    if condition_id is not None:
        if condition_id not in REVIEW_CONDITIONS:
            raise SystemExit(f"unknown review condition {condition_id}")
        CONDITION_ID = condition_id
    solo.configure(CONDITION_ID)
    c017.STAGE = STAGE
    c017.configure_shared()


def condition() -> dict[str, Any]:
    return resolve_condition(ROOT / "conditions" / CONDITION_ID / "condition.yaml", repository_root=ROOT)


def require_registered_differences() -> dict[str, Any]:
    spec = REVIEW_CONDITIONS[CONDITION_ID]
    cond = condition()
    parent = resolve_condition(ROOT / "conditions" / spec["base"] / "condition.yaml", repository_root=ROOT)
    for key in sorted(set(cond) | set(parent)):
        if key == "dataset" and (spec.get("test") or spec.get("dev_pool") or spec.get("extension")):
            continue
        if key not in ALLOWED_SECTIONS and cond.get(key) != parent.get(key):
            raise SystemExit(f"{CONDITION_ID} {key} differs from {spec['base']}")
    if spec.get("test"):
        if cond["dataset"]["manifest"] not in solo.TEST_MANIFESTS:
            raise SystemExit(f"{CONDITION_ID} is a test condition and must run a sealed test manifest")
        solo.manifest_info(ROOT / "conditions" / CONDITION_ID / "condition.yaml")
        changed = {k for k in set(cond["dataset"]) | set(parent["dataset"])
                   if cond["dataset"].get(k) != parent["dataset"].get(k)}
        if not changed <= {"manifest", "manifest_rows", "manifest_sha256", "development_only"}:
            raise SystemExit(f"{CONDITION_ID} dataset differs from {spec['base']} beyond the manifest")
    if spec.get("dev_pool"):
        info = solo.manifest_info(ROOT / "conditions" / CONDITION_ID / "condition.yaml")
        if cond["dataset"]["manifest"] in solo.TEST_MANIFESTS or info["rows"] != solo.EXPECTED_ROWS \
                or cond["dataset"].get("development_only") is not True:
            raise SystemExit(f"{CONDITION_ID} must run a {solo.EXPECTED_ROWS}-row development manifest")
        changed = {k for k in set(cond["dataset"]) | set(parent["dataset"])
                   if cond["dataset"].get(k) != parent["dataset"].get(k)}
        if not changed <= {"manifest", "manifest_rows", "manifest_sha256", "development_only"}:
            raise SystemExit(f"{CONDITION_ID} dataset differs from {spec['base']} beyond the manifest")
    if spec.get("extension"):
        # A review on the second math dataset: the base's review on the OlympiadBench manifest,
        # reviewing that dataset's solo outputs (docs/plan/olympiadbench_extension.md).
        solo.manifest_info(ROOT / "conditions" / CONDITION_ID / "condition.yaml")
        if cond["dataset"]["id"] != solo.OLYMPIAD_ID or cond["dataset"]["manifest"] not in solo.EXTENSION_MANIFESTS:
            raise SystemExit(f"{CONDITION_ID} must run the registered OlympiadBench manifest")
    source = spec.get("model_from", spec["base"])
    reference = parent if source == spec["base"] else resolve_condition(
        ROOT / "conditions" / source / "condition.yaml", repository_root=ROOT)
    capture = dict(cond["model"].get("pass1_logit_capture") or {})
    if spec.get("special_token_repair"):
        if capture.pop(REPAIR_KEY, None) != REPAIR_RULE or capture != reference["model"].get("pass1_logit_capture"):
            raise SystemExit(f"{CONDITION_ID} must add only {REPAIR_KEY}: {REPAIR_RULE} to the capture block")
    elif REPAIR_KEY in capture:
        raise SystemExit(f"{CONDITION_ID} does not register {REPAIR_KEY}")
    for key in sorted(set(cond["model"]) | set(reference["model"])):
        if key in ALLOWED_MODEL_KEYS or (key == "pass1_logit_capture" and spec.get("special_token_repair")):
            continue
        if cond["model"].get(key) != reference["model"].get(key):
            raise SystemExit(f"{CONDITION_ID} model.{key} differs from {source}")
    _, user = split_prompt(ROOT / cond["model"]["prompt_pass1"])
    if set(PROMPT_VARIABLE.findall(user)) != {"problem", "y0"}:
        raise SystemExit("reviewer prompt must take exactly the problem and G's output")
    if spec.get("frozen_prompt"):
        expected = spec.get("prompt", solo.REVIEW_PROMPT_V2)
        if cond["model"]["prompt_pass1"] != expected:
            raise SystemExit(f"{CONDITION_ID} must use the frozen review prompt {expected}")
        solo.require_frozen_prompt(cond["model"]["prompt_pass1"])
    if cond["generator_source"]["condition_id"] != spec["g"]:
        raise SystemExit(f"{CONDITION_ID} must review {spec['g']}'s G outputs")
    if cond["replicates"]["seed_cell_id"] != CONDITION_ID:
        raise SystemExit(f"{CONDITION_ID} must use its own seed cell")
    return cond["model"]


def messages(model: dict, problem: str, y0: str) -> list[dict[str, str]]:
    system, user = split_prompt(ROOT / model["prompt_pass1"])
    return [{"role": "system", "content": system},
            {"role": "user", "content": render(user, problem=problem, y0=y0)}]


def stage_smoke(args: argparse.Namespace) -> None:
    model = shared.require_frozen_model()
    decoding = {**model["pass1_decoding"], "output_logprobs": {
        "enabled": True, "mode": "raw_logprobs", "return_token_ids": True}}
    content, _, usage = api_call(
        api_base=args.api_base, served_model=args.served_model,
        messages=messages(model, "What is $2+2$?", "We add: $2+2=4$. \\boxed{4}"),
        decoding=decoding, seed=1, thinking=False,
        chat_template_kwargs=chat_template_kwargs_for(model))
    trace = usage.get("token_logprobs") or {}
    if not content or trace.get("status") != "ok":
        raise RuntimeError(f"reviewer smoke failed: content={bool(content)} trace={trace.get('status')}")
    print(canonical_json({"stage": "smoke", "status": "passed", "tokens": len(trace.get("tokens") or [])}))


def special_token_ids(model: dict[str, Any]) -> set[int]:
    """Ids of the served snapshot's special added tokens, the ones vLLM leaves out of the text."""
    data = json.loads((Path(model["server_path"]) / "tokenizer.json").read_text(encoding="utf-8"))
    return {int(t["id"]) for t in data.get("added_tokens", []) if t.get("special")}


def repair_special_token_bytes(trace: dict[str, Any] | None, content: str,
                               special: set[int]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Empty the bytes of special tokens, as the detokenizer that wrote the text does, then locate
    the content with the unchanged rule of generated_token_logprobs.build_generated_token_trace.

    Only a trace that failed to align and holds a special token with bytes is touched; token ids,
    and so the same-forward capture check, are unchanged. A repaired trace that still does not
    align keeps status text_mismatch and the run stops as before."""
    if not trace or trace.get("status") == "ok":
        return trace, []
    tokens = [dict(t) for t in trace.get("tokens") or []]
    if not any(t.get("token_id") in special and t["bytes"] for t in tokens):
        return trace, []
    repaired, reconstructed = [], bytearray()
    for t in tokens:
        if t.get("token_id") in special and t["bytes"]:
            repaired.append({"index": t["index"], "token_id": t["token_id"], "bytes_hex": bytes(t["bytes"]).hex()})
            t["bytes"] = []
        t["start_byte"] = len(reconstructed)
        reconstructed.extend(bytes(t["bytes"]))
        t["end_byte"] = len(reconstructed)
    raw, content_bytes = bytes(reconstructed), content.encode("utf-8")
    start = None
    if content_bytes and raw.endswith(content_bytes):
        start = len(raw) - len(content_bytes)
    elif content_bytes and raw.count(content_bytes) == 1:
        start = raw.index(content_bytes)
    fixed = {**trace, "tokens": tokens, "status": "ok" if start is not None else "text_mismatch",
             "reconstructed_text_sha256": sha256_bytes(raw), "content_start_byte": start,
             "content_end_byte": None if start is None else start + len(content_bytes)}
    return fixed, repaired


def stage_review(args: argparse.Namespace) -> None:
    model = shared.require_frozen_model()
    special = special_token_ids(model) if REVIEW_CONDITIONS[CONDITION_ID].get("special_token_repair") else None
    ids = shared.manifest_ids(Path(args.manifest), int(args.expected_rows))
    sources = base.load_source_rows(ids)
    g_rows = {r["problem_id"]: r for r in read_jsonl(Path(args.g_records))}
    missing = [p for p in ids if p not in g_rows]
    if missing:
        raise RuntimeError(f"G records missing for {missing[:5]}")
    source_run = condition()["generator_source"]["pass1_run_id"]
    foreign = sorted({r["run_id"] for r in g_rows.values()} - {source_run})
    if foreign:
        raise RuntimeError(f"G records come from {foreign}, not the registered {source_run}")
    template_kwargs = chat_template_kwargs_for(model)
    output = Path(args.output)
    done = completed_keys(output, ("problem_id",))
    written = 0
    for problem_id in ids:
        if (problem_id,) in done:
            continue
        g_row = g_rows[problem_id]
        y0 = g_row["outputs"][0]["raw_output"] or ""
        seed = derive_call_seed(c017.REPLICATE_PLAN_ID, "review", args.replicate_id,
                                int(args.base_seed), c017.SEED_CELL_ID, problem_id, 0)
        capture_id = stable_id(args.run_id, problem_id, "same_forward_logit_capture")
        content, _, usage = api_call(
            api_base=args.api_base, served_model=args.served_model,
            messages=messages(model, sources[problem_id]["problem"], y0),
            decoding={**model["pass1_decoding"],
                      "output_logprobs": {"enabled": True, "mode": "raw_logprobs",
                                          "return_token_ids": True},
                      "vllm_xargs": {"c014_capture_id": capture_id}},
            seed=seed, thinking=False, chat_template_kwargs=template_kwargs)
        trace = usage.pop("token_logprobs", None)
        if special is not None:
            trace, repaired = repair_special_token_bytes(trace, content or "", special)
            if repaired:
                append_jsonl(output.parent / REPAIRS_FILE, {
                    "problem_id": problem_id, "trace_status_after": trace["status"], "repaired_tokens": repaired})
        extracted, parse_status = shared.extract_answer(content)
        record = {
            "schema_version": "gv-review-record-v1",
            "record_id": stable_id(args.run_id, problem_id, "review"),
            **shared._identity(args.run_id), "replicate_id": args.replicate_id, "stage": STAGE,
            "problem_id": problem_id, "problem_text": sources[problem_id]["problem"],
            "generator": shared.model_identity(ROOT / model["prompt_pass1"]),
            "outputs": [{"sample_index": 0, "call_seed": seed, "raw_output": content,
                         "extracted_answer": extracted, "parse_status": parse_status,
                         "token_logprobs": trace, "usage": base.compact_usage(usage)}],
            "y0_sample_index": 0, "created_at_utc": utc_now(),
            "source_g_run_id": g_row["run_id"], "source_g_record_id": g_row["record_id"],
            "source_g_output_sha256": sha256_bytes(y0.encode("utf-8")),
        }
        append_jsonl(output, record)
        capture = shared.wait_for_same_forward_capture(Path(args.capture_sidecar), capture_id,
                                                       timeout_seconds=30)
        logit_record = shared.build_same_forward_logit_record(
            run_id=args.logit_run_id, pass1_record=record, capture=capture)
        append_jsonl(Path(args.logit_output), logit_record)
        if logit_record["status"] != "ok":
            raise RuntimeError(f"same-forward capture failed for {problem_id}: {logit_record['status']}")
        written += 1
    print(canonical_json({"stage": "review", "written": written, "total": len(ids)}))


def stage_grade(args: argparse.Namespace) -> None:
    reviews = list(read_jsonl(Path(args.review_records)))
    g_rows = {r["problem_id"]: r for r in read_jsonl(Path(args.g_records))}
    sources = base.load_source_rows([r["problem_id"] for r in reviews])
    output = Path(args.output)
    for row in reviews:
        pid = row["problem_id"]
        gold = sources[pid]["gold"]
        v_text = base.answer_text(row["outputs"][0])
        u, status = base.grade_prediction(gold, v_text)
        append_jsonl(output, {"problem_id": pid, "outcome": "u_review", "u": u, "grader_status": status,
                              "source_record_id": row["record_id"]})
        delivered_from = "review" if v_text is not None else "g_fallback"
        text = v_text if v_text is not None else base.answer_text(g_rows[pid]["outputs"][0])
        u, status = base.grade_prediction(gold, text)
        append_jsonl(output, {"problem_id": pid, "outcome": "u_gv", "u": u, "grader_status": status,
                              "delivered_from": delivered_from, "source_record_id": row["record_id"]})
    print(canonical_json({"stage": "grade", "rows": len(reviews)}))


def stage_analyze(args: argparse.Namespace) -> None:
    spec = REVIEW_CONDITIONS[CONDITION_ID]
    listed = solo.ungradeable_ids()
    reviews = {r["problem_id"]: r for r in read_jsonl(Path(args.review_records))}
    g_rows = {r["problem_id"]: r for r in read_jsonl(Path(args.g_records))}
    g_grades = {r["problem_id"]: r["u"] for r in read_jsonl(Path(args.g_grades)) if r["outcome"] == "u0"}
    grades = collections.defaultdict(dict)
    for r in read_jsonl(Path(args.grades)):
        grades[r["problem_id"]][r["outcome"]] = r
    if len(reviews) != int(args.expected_rows) or len(grades) != int(args.expected_rows):
        raise RuntimeError(f"incomplete inputs: {len(reviews)} reviews, {len(grades)} graded rows")
    errors = sorted(p for p, g in grades.items() for r in g.values()
                    if r["u"] is None and p not in listed)
    if errors:
        raise SystemExit(f"grader_error on problems not registered as ungradeable: {errors}")
    ids = [p for p in reviews if p not in listed]
    excluded = sorted(set(reviews) & listed)
    gv = sum(grades[p]["u_gv"]["u"] for p in ids)
    rev = sum(grades[p]["u_review"]["u"] for p in ids)
    g = sum(int(g_grades[p]) for p in ids)
    fallback = sum(grades[p]["u_gv"]["delivered_from"] == "g_fallback" for p in ids)
    changed = collections.Counter()
    for p in ids:
        gt, vt = base.answer_text(g_rows[p]["outputs"][0]), base.answer_text(reviews[p]["outputs"][0])
        try:
            eq = None if gt is None or vt is None else base.math_equivalent(gt, vt)[0]
        except (Exception, base.MathVerifyTimeout):   # math_verify's timeout is a BaseException
            eq = None
        changed["kept_g_answer" if eq else "changed_answer" if eq is False else "not_comparable"] += 1
    repair = sum(g_grades[p] == 0 and grades[p]["u_gv"]["u"] == 1 for p in ids)
    harm = sum(g_grades[p] == 1 and grades[p]["u_gv"]["u"] == 0 for p in ids)
    tokens = [int(reviews[p]["outputs"][0]["usage"].get("output_tokens") or 0) for p in ids]
    report = {"condition_id": CONDITION_ID, "run_id": args.run_id, "manifest_rows": int(args.expected_rows),
              "rows": len(ids), "excluded_ungradeable_gold": excluded,
              "gv_correct": gv, "gv_accuracy": gv / len(ids), "review_correct": rev,
              "g_correct": g, "repair_vs_g": repair, "harm_vs_g": harm,
              "g_fallback_rows": fallback, "answer_relation": dict(changed),
              "review_output_tokens": {"sum": sum(tokens), "mean": sum(tokens) / len(tokens)},
              "grader_status_gv": dict(collections.Counter(grades[p]["u_gv"]["grader_status"] for p in ids)),
              "manifest": condition()["dataset"]["manifest"],
              "development_only": condition()["dataset"]["manifest"]
              not in solo.TEST_MANIFESTS | solo.EXTENSION_MANIFESTS}
    if spec.get("special_token_repair"):
        path = Path(args.review_records).parent / REPAIRS_FILE
        rows = {r["problem_id"]: r for r in read_jsonl(path)} if path.exists() else {}
        rows = {p: r for p, r in rows.items() if p in reviews}
        report["special_token_bytes_repaired"] = {
            "rows": len(rows), "tokens": sum(len(r["repaired_tokens"]) for r in rows.values()),
            "problem_ids": sorted(rows)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = "\n".join([
        f"# {CONDITION_ID} GV reviewer summary", "",
        f"- Manifest rows {report['manifest_rows']}; excluded as ungradeable gold: {excluded}; "
        f"figures below are over {len(ids)} rows.",
        f"- GV delivered correct: {gv}/{len(ids)} = {gv / len(ids):.4f}",
        f"- Reviewer's own answer correct: {rev}/{len(ids)}",
        f"- G alone ({spec['g']}) on the same rows: {g}/{len(ids)}; GV vs G repair/harm {repair}/{harm}",
        f"- Rows delivered from G because the reviewer gave no answer: {fallback}",
        f"- Reviewer's final answer relative to G's: {dict(changed)}",
        f"- Reviewer output tokens: {sum(tokens)} (mean {sum(tokens) / len(tokens):.1f})", "",
        f"Evidence was written before grading. The comparison with V alone ({spec['v']}) and with",
        "max(G, V) is a separate analysis over the three archives.", ""])
    if "special_token_bytes_repaired" in report:
        fixed = report["special_token_bytes_repaired"]
        summary += (f"\nSpecial-token bytes emptied before alignment ({REPAIR_RULE}): {fixed['rows']} rows, "
                    f"{fixed['tokens']} tokens {fixed['problem_ids']}.\n")
    (out.parent / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def main() -> int:
    argv = sys.argv[1:]
    condition_id = None
    if argv[:1] == ["--condition-id"] and len(argv) >= 2:
        condition_id, argv = argv[1], argv[2:]
    configure(condition_id)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-runnable")
    p = sub.add_parser("smoke")
    p.add_argument("--run-id", required=True); p.add_argument("--api-base", required=True)
    p.add_argument("--served-model", required=True)
    p = sub.add_parser("review")
    for name in ("--run-id", "--api-base", "--served-model", "--manifest", "--g-records", "--output",
                 "--capture-sidecar", "--logit-output", "--logit-run-id"):
        p.add_argument(name, required=True)
    p.add_argument("--expected-rows", type=int, default=solo.EXPECTED_ROWS)
    p.add_argument("--replicate-id", default="p01")
    p.add_argument("--base-seed", type=int, default=solo.BASE_SEED)
    p = sub.add_parser("reconcile")
    p.add_argument("review"); p.add_argument("trace")
    sub.add_parser("manifest-info")
    p = sub.add_parser("shard-manifest")
    p.add_argument("full"); p.add_argument("index", type=int); p.add_argument("count", type=int)
    p.add_argument("out")
    p = sub.add_parser("merge-shards")
    p.add_argument("full"); p.add_argument("shard_root"); p.add_argument("count", type=int)
    p.add_argument("out_dir"); p.add_argument("names", nargs="+")
    p = sub.add_parser("evidence")
    for name in ("--run-id", "--pass1-records", "--logit-traces", "--output"):
        p.add_argument(name, required=True)
    p = sub.add_parser("grade")
    for name in ("--review-records", "--g-records", "--output"):
        p.add_argument(name, required=True)
    p = sub.add_parser("analyze")
    for name in ("--run-id", "--review-records", "--g-records", "--g-grades", "--grades", "--output"):
        p.add_argument(name, required=True)
    p.add_argument("--expected-rows", type=int, default=solo.EXPECTED_ROWS)
    args = parser.parse_args(argv)
    if args.command == "check-runnable":
        model = require_registered_differences()
        print(canonical_json({"runnable": True, "condition_id": CONDITION_ID,
                              "model_id": model["model_id"], "prompt": model["prompt_pass1"]}))
    elif args.command == "smoke":
        stage_smoke(args)
    elif args.command == "review":
        stage_review(args)
    elif args.command == "reconcile":
        print(canonical_json({"stage": "reconcile", **solo.reconcile(Path(args.review), Path(args.trace))}))
    elif args.command == "manifest-info":
        info = solo.manifest_info(ROOT / "conditions" / CONDITION_ID / "condition.yaml")
        print(info["path"], info["sha256"], info["rows"])
    elif args.command == "shard-manifest":
        print(solo.write_shard_manifest(Path(args.full), args.index, args.count, Path(args.out)))
    elif args.command == "merge-shards":
        print(canonical_json({"stage": "merge-shards", **solo.merge_shards(
            Path(args.full), Path(args.shard_root), args.count, Path(args.out_dir), args.names)}))
    elif args.command == "evidence":
        c017.stage_evidence(args)
    elif args.command == "grade":
        stage_grade(args)
    else:
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
