#!/usr/bin/env python3
"""Check archived records against the schemas their own condition registered.

The gap this closes: runners write each registered schema's sha256 into the run
manifest, which records which schema a run claims, and then nothing ever checks the
records against it. g_record_single_v2 pins condition_id to c003, the dataset to
math500 and the stage to pilot300, and both code-domain conditions registered it
unchanged for bigcodebench rows at stage gate0. Two full runs passed with a registered
schema that describes a different experiment.

Records are matched to schemas by their own schema_version, not by position, so a file
that mixes generations validates correctly: c007 reuses 300 rows written under
g-record-single-v2 alongside 532 written under v3.

Dependency-free on purpose. The repository already hand-rolls validate_v_report rather
than taking a JSON Schema library, and adding a package to the run environment
mid-experiment is the kind of change that has already cost this project a job. The
subset implemented here is the subset these schemas use; anything unrecognised is
reported rather than ignored, so the checker cannot quietly pass what it cannot read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

KNOWN_KEYWORDS = {
    "$schema", "$id", "$defs", "$ref", "title", "description", "type", "const", "enum",
    "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
    "minimum", "maximum", "minLength", "pattern", "format", "allOf", "anyOf",
    "if", "then", "uniqueItems",
}

TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
    "null": type(None),
}


class SchemaStore:
    """Loads schema files and resolves $ref, both file-relative and in-document."""

    def __init__(self) -> None:
        self._by_path: dict[Path, dict] = {}

    def load(self, path: Path) -> dict:
        path = path.resolve()
        if path not in self._by_path:
            self._by_path[path] = json.loads(path.read_text(encoding="utf-8"))
        return self._by_path[path]

    def resolve(self, ref: str, base: Path, root_doc: dict) -> tuple[dict, Path, dict]:
        if ref.startswith("#/"):
            node: Any = root_doc
            for part in ref[2:].split("/"):
                node = node[part]
            return node, base, root_doc
        target = (base.parent / ref).resolve()
        doc = self.load(target)
        return doc, target, doc


def type_matches(value: Any, expected: Any) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        python_type = TYPES.get(name)
        if python_type is None:
            return False
        # JSON has no separate boolean-as-integer; a bool must not satisfy "integer".
        if name in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, python_type):
            return True
    return False


def validate(value: Any, schema: dict, store: SchemaStore, base: Path, root_doc: dict,
             where: str, errors: list[str]) -> None:
    unknown = set(schema) - KNOWN_KEYWORDS
    if unknown:
        errors.append(f"{where}: schema uses keywords this checker does not implement: "
                      f"{sorted(unknown)}")
        return

    if "$ref" in schema:
        target, new_base, new_root = store.resolve(schema["$ref"], base, root_doc)
        validate(value, target, store, new_base, new_root, where, errors)
        return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: expected const {schema['const']!r}, found {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {value!r} is not one of {schema['enum']}")
    if "type" in schema and not type_matches(value, schema["type"]):
        errors.append(f"{where}: expected type {schema['type']}, "
                      f"found {type(value).__name__}")
        return

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{where}: shorter than minLength {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{where}: does not match {schema['pattern']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{where}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{where}: above maximum {schema['maximum']}")

    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{where}: missing required field {name!r}")
        properties = schema.get("properties", {})
        extra = set(value) - set(properties)
        if schema.get("additionalProperties") is False and extra:
            errors.append(f"{where}: unregistered fields {sorted(extra)}")
        elif isinstance(schema.get("additionalProperties"), dict):
            for name in sorted(extra):
                validate(value[name], schema["additionalProperties"], store, base,
                         root_doc, f"{where}.{name}", errors)
        for name, subschema in properties.items():
            if name in value:
                validate(value[name], subschema, store, base, root_doc,
                         f"{where}.{name}", errors)

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{where}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{where}: more than maxItems {schema['maxItems']}")
        if schema.get("uniqueItems") and len(value) != len({json.dumps(v, sort_keys=True) for v in value}):
            errors.append(f"{where}: items are not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                validate(item, schema["items"], store, base, root_doc,
                         f"{where}[{index}]", errors)

    for index, subschema in enumerate(schema.get("allOf", [])):
        validate(value, subschema, store, base, root_doc, f"{where}", errors)

    if "anyOf" in schema:
        branches = []
        for subschema in schema["anyOf"]:
            branch: list[str] = []
            validate(value, subschema, store, base, root_doc, where, branch)
            if not branch:
                break
            branches.append(branch)
        else:
            errors.append(f"{where}: matched none of the anyOf branches "
                          f"({len(schema['anyOf'])} tried)")

    if "if" in schema:
        probe: list[str] = []
        validate(value, schema["if"], store, base, root_doc, where, probe)
        if not probe and "then" in schema:
            validate(value, schema["then"], store, base, root_doc, where, errors)


def schema_version_of(schema: dict) -> str | None:
    node = schema.get("properties", {}).get("schema_version", {})
    return node.get("const")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a run's records against the schemas it registered")
    parser.add_argument("--bundle-dir", required=True,
                        help="a run work directory or archive, holding condition.resolved.yaml")
    parser.add_argument("--max-report", type=int, default=5,
                        help="how many failures to print per record file")
    args = parser.parse_args()

    bundle = Path(args.bundle_dir)
    condition_path = bundle / "condition.resolved.yaml"
    if not condition_path.is_file():
        print(f"No condition.resolved.yaml in {bundle}", file=sys.stderr)
        return 2

    # The condition is read for its registered schema paths only; a two-line scan avoids
    # taking a YAML dependency for a nested block of plain path strings.
    registered: list[Path] = []
    in_block = False
    for line in condition_path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s{2}record_schemas:\s*$", line):
            in_block = True
            continue
        if in_block:
            match = re.match(r"^\s{4}\w+:\s*(\S+\.json)\s*$", line)
            if match:
                registered.append(ROOT / match.group(1))
                continue
            if line.strip() and not line.startswith("    "):
                in_block = False
    if not registered:
        print(f"No record_schemas found in {condition_path}", file=sys.stderr)
        return 2

    store = SchemaStore()
    by_version: dict[str, tuple[dict, Path]] = {}
    for path in registered:
        if not path.is_file():
            print(f"Registered schema missing: {path}", file=sys.stderr)
            return 2
        doc = store.load(path)
        version = schema_version_of(doc)
        if version is not None:
            by_version[version] = (doc, path.resolve())

    print(f"registered schema versions: {sorted(by_version)}")

    total_rows = 0
    total_bad = 0
    unversioned = 0
    unmatched: dict[str, int] = {}
    for records in sorted((bundle / "records").glob("*.jsonl")):
        rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines() if line.strip()]
        bad = 0
        shown = 0
        versions: dict[str, int] = {}
        for index, row in enumerate(rows, start=1):
            version = row.get("schema_version")
            versions[str(version)] = versions.get(str(version), 0) + 1
            if version is None:
                # Intermediate working files such as grade_requests carry no
                # schema_version and are not records; nothing claims a shape for them.
                unversioned += 1
                continue
            entry = by_version.get(version)
            if entry is None:
                unmatched[version] = unmatched.get(version, 0) + 1
                continue
            doc, path = entry
            errors: list[str] = []
            validate(row, doc, store, path, doc, f"{records.name}:{index}", errors)
            if errors:
                bad += 1
                if shown < args.max_report:
                    shown += 1
                    for message in errors[:4]:
                        print(f"  FAIL {message}")
        total_rows += len(rows)
        total_bad += bad
        status = "ok" if bad == 0 else f"{bad} INVALID"
        print(f"{records.name}: {len(rows)} rows, versions {versions} -> {status}")

    if unversioned:
        print(f"rows with no schema_version (working files, not records): {unversioned}")
    if unmatched:
        print("SCHEMA NOT REGISTERED BY THIS CONDITION: " + str(unmatched))
    print(f"TOTAL: {total_rows} rows, {total_bad} invalid, "
          f"{sum(unmatched.values())} carrying a version the condition does not register")
    return 0 if total_bad == 0 and not unmatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
