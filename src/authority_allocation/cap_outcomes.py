"""Multi-output supervision targets derived from frozen counterfactuals."""

from __future__ import annotations

from typing import Mapping


CAP_NAMES = ("M0", "M1", "M2")


def build_cap_outcome_target(
    z0: int,
    z1: int,
    z2: int,
) -> dict[str, object]:
    """Return the lossless three-cap target and derived Oracle diagnostics.

    ``cap_outcomes`` is the supervision target.  ``oracle_actions`` and
    ``minimal_oracle_action`` are descriptive derivatives; the latter must not
    be used as a single multiclass training label because it collapses ties.
    """

    values = (z0, z1, z2)
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError("cap outcomes must be fully graded binary values")

    cap_outcomes = dict(zip(CAP_NAMES, values, strict=True))
    oracle_actions = [cap for cap in CAP_NAMES if cap_outcomes[cap] == 1]
    return {
        "cap_outcomes": cap_outcomes,
        "oracle_correct": int(bool(oracle_actions)),
        "oracle_actions": oracle_actions,
        "minimal_oracle_action": oracle_actions[0] if oracle_actions else None,
        "oracle_class": (
            "unfixable" if not oracle_actions else "_and_".join(oracle_actions)
        ),
    }


def cap_outcome_target_from_record(record: Mapping[str, object]) -> dict[str, object]:
    """Build the target from a counterfactual record containing z0/z1/z2."""

    values = (record["z0"], record["z1"], record["z2"])
    if any(type(value) is not int for value in values):
        raise ValueError("cap outcomes must be fully graded binary values")
    return build_cap_outcome_target(*values)  # type: ignore[arg-type]
