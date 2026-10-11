#!/usr/bin/env python3
"""c021: c019's solo Qwen3-8B pass 1 with c018's 4096-token budget.

c017's runner is reused unchanged. c019's replicate plan and seed cell are kept, so call
seeds are identical to c019's; under greedy decoding every c019 output that stopped before
3072 tokens is expected to reproduce exactly. check-runnable proves that c021 differs from
c019 in the pass-1 budget alone and that the budget equals c018's verifier budget.
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
from authority_allocation.serialization import canonical_json  # noqa: E402

CONDITION_ID = "c021"
CONDITION_PATH = ROOT / "conditions/c021/condition.yaml"
REPLICATE_PLAN_ID = "r011"
SEED_CELL_ID = "c019"
BASE_SEED = 20260924
# Keys that may differ from c019 anywhere in the resolved condition.
ALLOWED = {"condition_id", "status", "purpose", "gate", "runnable_gate", "comparison"}


def configure() -> None:
    c017.CONDITION_ID = CONDITION_ID
    c017.CONDITION_PATH = CONDITION_PATH
    c017.REPLICATE_PLAN_ID = REPLICATE_PLAN_ID
    c017.SEED_CELL_ID = SEED_CELL_ID


def require_only_budget_changed() -> dict[str, Any]:
    c021 = resolve_condition(CONDITION_PATH, repository_root=ROOT)
    c019 = resolve_condition(ROOT / "conditions/c019/condition.yaml", repository_root=ROOT)
    c018 = resolve_condition(ROOT / "conditions/c018/condition.yaml", repository_root=ROOT)
    for key in sorted(set(c021) | set(c019)):
        if key in ALLOWED or key in ("model", "protocol"):
            continue
        if c021.get(key) != c019.get(key):
            raise SystemExit(f"c021 {key} differs from c019")
    protocol, base_protocol = dict(c021["protocol"]), dict(c019["protocol"])
    schemas, base_schemas = dict(protocol.pop("record_schemas")), dict(base_protocol.pop("record_schemas"))
    schemas.pop("scalar_evidence"); base_schemas.pop("scalar_evidence")
    if protocol != base_protocol or schemas != base_schemas:
        raise SystemExit("c021 protocol differs from c019 beyond its evidence schema id")
    model, base = c021["model"], c019["model"]
    for key in sorted(set(model) | set(base)):
        if key in ("pass1_decoding", "pass1_budget_matched_to"):
            continue
        if model.get(key) != base.get(key):
            raise SystemExit(f"c021 model.{key} differs from c019")
    decoding, base_decoding = dict(model["pass1_decoding"]), dict(base["pass1_decoding"])
    budget = decoding.pop("max_new_tokens")
    base_decoding.pop("max_new_tokens")
    if decoding != base_decoding:
        raise SystemExit("c021 pass1 decoding differs from c019 beyond max_new_tokens")
    if int(budget) != int(c018["verifier"]["decoding"]["max_new_tokens"]):
        raise SystemExit("c021 pass1 budget must equal c018's verifier budget")
    if model.get("thinking") is not False or float(decoding["temperature"]) != 0.0:
        raise SystemExit("c021 requires thinking=false and temperature 0")
    if c021["features"]["count"] != 7:
        raise SystemExit("c021 must keep the seven registered scalars")
    return model


def main() -> int:
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == "check-runnable":
        model = require_only_budget_changed()
        print(canonical_json({"runnable": True, "condition_id": CONDITION_ID,
                              "revision": model["revision"], "features": 7,
                              "max_new_tokens": model["pass1_decoding"]["max_new_tokens"]}))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "generate" and "--base-seed" not in sys.argv:
        sys.argv += ["--base-seed", str(BASE_SEED)]
    return c017.main()


if __name__ == "__main__":
    raise SystemExit(main())
