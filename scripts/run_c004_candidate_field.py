#!/usr/bin/env python3
"""Rebuild c004 evidence from c003 traces and run its downstream stages."""

from __future__ import annotations

import run_c002_dev30 as runner


def main() -> int:
    runner.configure_c004_candidate_field()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
