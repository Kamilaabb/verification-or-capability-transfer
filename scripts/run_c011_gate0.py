#!/usr/bin/env python3
"""Complete-prompt Gate 0: Seed-Coder G against the frozen Qwen3 verifier."""

from __future__ import annotations

import run_c005_gate0 as runner


def configure_c011() -> None:
    runner.CONDITION_ID = "c011"
    runner.CONDITION_PATH = runner.ROOT / "conditions/c011/condition.yaml"
    runner.DATASET_ID = "bigcodebench-b74c0d0-complete-s20260917"
    runner.PROBLEM_FIELD = "complete_prompt"
    runner.REPLICATE_PLAN_ID = "r004"
    runner.STAGE = "gate0"
    runner.EXPECTED_ROWS = 300
    runner.G_RECORD_SCHEMA_VERSION = "g-record-single-v3"
    runner.V_RECORD_SCHEMA_VERSION = "v-record-v3"


def main() -> int:
    configure_c011()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
