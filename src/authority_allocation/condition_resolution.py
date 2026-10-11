"""Resolve condition ``extends + overrides`` chains.

Mappings deep-merge, lists replace as a whole, and explicit null remains a value.  The
resolved object contains no composition directives and is what runners hash and archive.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = {key: deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            merged[key] = (
                deep_merge(merged[key], value) if key in merged else deepcopy(value)
            )
        return merged
    return deepcopy(override)


def resolve_condition(path: Path, *, repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()

    def load(current: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
        current = current.resolve()
        if current in chain:
            cycle = " -> ".join(str(item) for item in (*chain, current))
            raise ValueError(f"condition inheritance cycle: {cycle}")
        data = yaml.safe_load(current.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"condition must be a mapping: {current}")
        if "extends" not in data:
            if "overrides" in data:
                raise ValueError(f"condition has overrides without extends: {current}")
            return data
        if set(data) - {"schema_version", "condition_id", "status", "extends", "overrides"}:
            raise ValueError(
                f"overlay carries fields outside overrides: {current}"
            )
        base_path = (repository_root / str(data["extends"])).resolve()
        try:
            base_path.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError("condition base escapes repository root") from exc
        overrides = data.get("overrides")
        if not isinstance(overrides, dict):
            raise ValueError(f"condition overlay overrides must be a mapping: {current}")
        resolved = deep_merge(load(base_path, (*chain, current)), overrides)
        if resolved.get("condition_id") != data.get("condition_id"):
            raise ValueError("overlay condition_id must equal resolved condition_id")
        return resolved

    return load(path, ())


__all__ = ["deep_merge", "resolve_condition"]
