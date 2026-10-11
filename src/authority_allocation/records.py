"""Append-only JSONL record IO with resume support.

Records are fsynced per row so an interrupted stage can be resumed by key rather
than rerun, which matters when a stage costs GPU hours.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from authority_allocation.serialization import canonical_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: Path, fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[Any, ...]] = set()
    for row in read_jsonl(path):
        key = tuple(row.get(field) for field in fields)
        if key in keys:
            raise ValueError(f"duplicate key {key!r} in {path}")
        keys.add(key)
    return keys
