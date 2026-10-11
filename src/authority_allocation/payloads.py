"""Parsing and semantic validation of structured model payloads.

Kept apart from the grammar backend on purpose: the backend decides what the model
may emit, these decide what the project will accept.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def parse_json_object(text: str) -> dict[str, Any] | None:
    # strict=False: the grammar backend permits raw tabs inside string values, which
    # strict JSON forbids. Rejecting those discarded 16 semantically valid c003 reports.
    decoder = json.JSONDecoder(strict=False)
    stripped = text.strip()
    try:
        value = json.loads(stripped, strict=False)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_v_report(value: Mapping[str, Any] | None) -> bool:
    if value is None or set(value) != {"verdict", "critique", "candidate_answer", "requested_action"}:
        return False
    return (
        value["verdict"] in {"correct", "incorrect", "uncertain"}
        and isinstance(value["critique"], str)
        and bool(value["critique"].strip())
        and isinstance(value["candidate_answer"], str)
        and bool(value["candidate_answer"].strip())
        and value["requested_action"] in {"KEEP", "CRITIQUE_REVISE", "OVERRIDE"}
    )


def validate_controller(value: Mapping[str, Any] | None, menu: str) -> bool:
    allowed = {"M0", "M2"} if menu == "binary" else {"M0", "M1", "M2"}
    return value is not None and set(value) == {"authority_cap"} and value["authority_cap"] in allowed
