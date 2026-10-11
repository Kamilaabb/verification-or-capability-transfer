#!/usr/bin/env python3
"""c019: Qwen3-8B solves math dev300 alone with c017's G prompt and seven scalars.

c017's runner is reused unchanged; only its condition, replicate plan and seed cell are
pointed at c019. check-runnable additionally proves that c019 differs from c017 in the
model alone and that the model is c018's verifier snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_c017_math_expression_scalars as c017  # noqa: E402

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402

CONDITION_ID = "c019"
CONDITION_PATH = ROOT / "conditions/c019/condition.yaml"
REPLICATE_PLAN_ID = "r011"
SEED_CELL_ID = "c019"
BASE_SEED = 20260924

# Fields that may differ from c017's model block; everything else must be identical.
MODEL_IDENTITY_FIELDS = {
    "model_id", "revision", "tokenizer_revision", "provenance", "identity_reused_from",
    "identity_reuse_reason", "server_path", "prompt_pass1_provenance", "preflight_job",
    "preflight_reused_from", "preflight_policy",
}


def configure() -> None:
    c017.CONDITION_ID = CONDITION_ID
    c017.CONDITION_PATH = CONDITION_PATH
    c017.REPLICATE_PLAN_ID = REPLICATE_PLAN_ID
    c017.SEED_CELL_ID = SEED_CELL_ID


def require_only_model_changed() -> dict[str, Any]:
    c019 = resolve_condition(CONDITION_PATH, repository_root=ROOT)
    base = resolve_condition(ROOT / "conditions/c017/condition.yaml", repository_root=ROOT)
    c018 = resolve_condition(ROOT / "conditions/c018/condition.yaml", repository_root=ROOT)
    model, base_model, verifier = c019["model"], base["model"], c018["verifier"]
    for key in sorted(set(model) | set(base_model)):
        if key not in MODEL_IDENTITY_FIELDS and model.get(key) != base_model.get(key):
            raise SystemExit(f"c019 model.{key} differs from c017; only the model may change")
    for key in ("model_id", "revision", "server_path", "chat_template_hash"):
        if model[key] != verifier[key]:
            raise SystemExit(f"c019 model.{key} is not c018's verifier snapshot")
    if model["tokenizer_revision"] != model["revision"]:
        raise SystemExit("c019 tokenizer revision must equal the model revision")
    for section in ("features", "dataset", "stages"):
        if c019[section] != base[section]:
            raise SystemExit(f"c019 {section} differs from c017")
    if c019["replicates"]["seed_cell_id"] != SEED_CELL_ID:
        raise SystemExit("c019 seed cell must be c019")
    return model


def main() -> int:
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == "check-runnable":
        require_only_model_changed()
    if len(sys.argv) > 1 and sys.argv[1] == "generate" and "--base-seed" not in sys.argv:
        sys.argv += ["--base-seed", str(BASE_SEED)]
    return c017.main()


if __name__ == "__main__":
    raise SystemExit(main())
