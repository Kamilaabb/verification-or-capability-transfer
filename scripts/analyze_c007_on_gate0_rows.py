#!/usr/bin/env python3
"""Compare c007 and c013 cap outcomes on the identical 300 Gate-0 rows.

c013's `-3` net for M1 and c007's `+24` are not directly comparable: three registered
things differ between them at once (generator family, problem field, row count). This
script removes the third one for free.

Both datasets reuse the same frozen membership and order, and
`datasets/bigcodebench-b74c0d0-complete-s20260917/dataset.yaml` records that gate0 is
the first 300 development rows, so c007's 832-row counterfactual archive already
contains c013's 300 rows. Restricting it to them is arithmetic over two archived JSONL
files: no inference, no GPU, no manifest read, and no new fit. The row set is taken
from c013's own archive rather than from a manifest so that the comparison needs no
server data at all.

What remains after it is a two-way confound (generator family and problem field), which
this script cannot break and does not claim to.

Usage on the server:

    python scripts/analyze_c007_on_gate0_rows.py \
        --c007-counterfactual <c007 artifact>/counterfactual_records.jsonl \
        --c013-counterfactual <c013 artifact>/counterfactual_records.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.decision_density import (  # noqa: E402
    cap_correct,
    contrast_report,
    decision_relevant_rows,
    load_cap_outcomes,
    pattern_counts,
    restrict,
    two_sample_repair_share_z,
)
from authority_allocation.records import read_jsonl  # noqa: E402
from authority_allocation.serialization import canonical_json  # noqa: E402

EXPECTED_GATE0_ROWS = 300
EXPECTED_DEVELOPMENT_ROWS = 832


def summarize(outcomes, label: str) -> dict:
    rows = len(outcomes)
    return {
        "label": label,
        "rows": rows,
        "correct": {cap: cap_correct(outcomes, cap) for cap in ("z0", "z1", "z2")},
        "accuracy": {cap: cap_correct(outcomes, cap) / rows for cap in ("z0", "z1", "z2")},
        "oracle_correct": rows - pattern_counts(outcomes)["000"],
        "decision_relevant_rows": decision_relevant_rows(outcomes),
        "decision_relevant_share": decision_relevant_rows(outcomes) / rows,
        "patterns": pattern_counts(outcomes),
        "contrasts": [
            contrast_report(outcomes, "z0", "z1"),
            contrast_report(outcomes, "z1", "z2"),
            contrast_report(outcomes, "z0", "z2"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c007-counterfactual", required=True)
    parser.add_argument("--c013-counterfactual", required=True)
    parser.add_argument(
        "--allow-unexpected-row-counts",
        action="store_true",
        help="Skip the 832/300 archive size checks. Only for archives this script was "
             "not written against; the default guards against pointing it at the wrong run.",
    )
    args = parser.parse_args()

    c007 = load_cap_outcomes(read_jsonl(Path(args.c007_counterfactual)))
    c013 = load_cap_outcomes(read_jsonl(Path(args.c013_counterfactual)))

    if not args.allow_unexpected_row_counts:
        if len(c007) != EXPECTED_DEVELOPMENT_ROWS:
            raise SystemExit(f"expected {EXPECTED_DEVELOPMENT_ROWS} c007 rows, got {len(c007)}")
        if len(c013) != EXPECTED_GATE0_ROWS:
            raise SystemExit(f"expected {EXPECTED_GATE0_ROWS} c013 rows, got {len(c013)}")

    gate0_ids = list(c013)
    c007_gate0 = restrict(c007, gate0_ids)

    c007_full = summarize(c007, "c007 full development (832)")
    c007_restricted = summarize(c007_gate0, "c007 restricted to gate0 rows")
    c013_summary = summarize(c013, "c013 gate0 rows")

    a = c007_restricted["contrasts"][0]
    b = c013_summary["contrasts"][0]
    z, p = two_sample_repair_share_z(a["repair"], a["harm"], b["repair"], b["harm"])

    print(canonical_json({
        "row_set": {
            "source": "c013 counterfactual archive",
            "rows": len(gate0_ids),
            "c007_rows_held_out_of_the_comparison": len(c007) - len(gate0_ids),
        },
        "summaries": [c007_full, c007_restricted, c013_summary],
        "paired_row_set_m1_vs_m0": {
            "c007_repair_share": a["repair"] / a["discordant"] if a["discordant"] else None,
            "c013_repair_share": b["repair"] / b["discordant"] if b["discordant"] else None,
            "two_proportion_z": z,
            "two_proportion_p": p,
            "remaining_confound": "generator family and problem field still differ",
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
