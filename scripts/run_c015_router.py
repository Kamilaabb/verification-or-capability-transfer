#!/usr/bin/env python3
"""c015: c014 with a two-independent-check pass-2 prompt and reused pass 1."""

from __future__ import annotations

from pathlib import Path

import run_c014_router as runner


def configure_c015() -> None:
    runner.CONDITION_ID = "c015"
    runner.CONDITION_PATH = Path(__file__).resolve().parents[1] / "conditions/c015/condition.yaml"
    runner.SEED_CELL_ID = "c014"  # exact pairing with the completed c014 pass-2 calls


def main() -> int:
    configure_c015()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
