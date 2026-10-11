#!/usr/bin/env python3
"""Run c003 on the frozen 300-item development manifest.

The implementation reuses the battle-tested staged c002 runner while selecting
the c003 protocol, condition, feature, schema, and logprob-capture profile.
"""

from __future__ import annotations

import run_c002_dev30 as runner


def main() -> int:
    runner.configure_c003_development300()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
