#!/usr/bin/env python3
"""Re-measure Gate 0 with a structured-outputs backend that can carry a program."""

from __future__ import annotations

import run_c005_gate0 as runner


def main() -> int:
    runner.configure_c006_guidance_backend()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
