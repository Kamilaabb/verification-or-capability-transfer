from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import scripts.validate_records as checker


def run(value, schema: dict, base: Path | None = None) -> list[str]:
    errors: list[str] = []
    checker.validate(
        value, schema, checker.SchemaStore(), base or Path(__file__), schema,
        "row", errors,
    )
    return errors


class UnknownKeywordTests(unittest.TestCase):
    """The one property the checker cannot get wrong.

    A validator that ignores what it does not understand reports success on rules it
    never applied, which is worse than having no validator: it converts an unchecked
    constraint into a passing check.
    """

    def test_an_unimplemented_keyword_is_reported_not_skipped(self):
        errors = run(5, {"type": "integer", "multipleOf": 2})
        self.assertEqual(len(errors), 1)
        self.assertIn("does not implement", errors[0])

    def test_every_keyword_used_by_the_repository_schemas_is_implemented(self):
        root = Path(__file__).resolve().parents[1]
        used = set()

        def walk(node):
            if isinstance(node, dict):
                used.update(node.keys())
                for child in node.values():
                    walk(child)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        files = sorted((root / "protocols").rglob("*.json"))
        self.assertTrue(files, "no schemas found")
        for path in files:
            walk(json.loads(path.read_text(encoding="utf-8")))

        # Property names are not keywords; only look at the vocabulary we recognise as
        # JSON Schema, and assert none of it falls outside what the checker implements.
        vocabulary = {
            "$schema", "$id", "$defs", "$ref", "title", "description", "type", "const",
            "enum", "properties", "required", "additionalProperties", "items",
            "minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength",
            "pattern", "format", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
            "uniqueItems", "multipleOf", "exclusiveMinimum", "exclusiveMaximum",
            "propertyNames", "patternProperties", "dependentRequired", "contains",
        }
        self.assertEqual((used & vocabulary) - checker.KNOWN_KEYWORDS, set())


class ConstAndIdentityTests(unittest.TestCase):
    def test_a_pinned_const_rejects_another_condition(self):
        """The c006 case: a schema pinned to c003 against a record from c005."""

        schema = {"type": "object", "properties": {"condition_id": {"const": "c003"}}}
        self.assertEqual(run({"condition_id": "c003"}, schema), [])
        errors = run({"condition_id": "c005"}, schema)
        self.assertEqual(len(errors), 1)
        self.assertIn("expected const 'c003'", errors[0])

    def test_an_unpinned_identity_accepts_both(self):
        schema = {"type": "object",
                  "properties": {"condition_id": {"type": "string", "minLength": 1}}}
        self.assertEqual(run({"condition_id": "c003"}, schema), [])
        self.assertEqual(run({"condition_id": "c007"}, schema), [])
        self.assertEqual(len(run({"condition_id": ""}, schema)), 1)


class StructureTests(unittest.TestCase):
    def test_unregistered_fields_are_caught(self):
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"a": {"type": "string"}}}
        self.assertEqual(run({"a": "x"}, schema), [])
        errors = run({"a": "x", "b": 1}, schema)
        self.assertEqual(len(errors), 1)
        self.assertIn("unregistered fields ['b']", errors[0])

    def test_missing_required_fields_are_caught(self):
        schema = {"type": "object", "required": ["a", "b"],
                  "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
        errors = run({"a": "x"}, schema)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing required field 'b'", errors[0])

    def test_a_bool_does_not_satisfy_integer(self):
        self.assertEqual(len(run(True, {"type": "integer"})), 1)
        self.assertEqual(run(1, {"type": "integer"}), [])

    def test_nullable_with_a_pattern_accepts_null(self):
        """evidence_sha256 in counterfactual_record_v2 is exactly this shape."""

        schema = {"type": ["string", "null"], "pattern": "^[a-f0-9]{64}$"}
        self.assertEqual(run(None, schema), [])
        self.assertEqual(run("a" * 64, schema), [])
        self.assertEqual(len(run("nope", schema)), 1)


class ConditionalTests(unittest.TestCase):
    def test_if_then_fires_only_when_the_condition_holds(self):
        """rewrite_record_v2: deferred_to_grader forbids a fallback action."""

        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"},
                           "failure_fallback_action": {"type": ["string", "null"]}},
            "allOf": [{
                "if": {"properties": {"status": {"const": "deferred_to_grader"}},
                       "required": ["status"]},
                "then": {"properties": {"failure_fallback_action": {"type": "null"}}},
            }],
        }
        self.assertEqual(
            run({"status": "deferred_to_grader", "failure_fallback_action": None}, schema),
            [],
        )
        self.assertEqual(
            run({"status": "call_error", "failure_fallback_action": "KEEP"}, schema),
            [],
        )
        errors = run(
            {"status": "deferred_to_grader", "failure_fallback_action": "KEEP"}, schema)
        self.assertEqual(len(errors), 1)


class RepositorySchemaTests(unittest.TestCase):
    """The schemas c007 registers must accept the records c007 writes."""

    SCHEMAS = Path(__file__).resolve().parents[1] / "protocols/authority_v6/schemas"

    def check(self, filename: str, record: dict) -> list[str]:
        path = self.SCHEMAS / filename
        schema = json.loads(path.read_text(encoding="utf-8"))
        errors: list[str] = []
        checker.validate(record, schema, checker.SchemaStore(), path, schema,
                         "row", errors)
        return errors

    def grade_record(self, **overrides) -> dict:
        row = {
            "schema_version": "code-grade-record-v1",
            "record_id": "r", "run_id": "run", "condition_id": "c007",
            "condition_hash": "a" * 64, "stage": "gate1", "problem_id": "p",
            "outcome": "u1", "u": 1, "status": "ok", "detail": "pass",
            "seconds": 0.5, "created_at_utc": "2026-09-19T00:00:00Z",
        }
        row.update(overrides)
        return row

    def test_the_grade_schema_accepts_a_real_shaped_row(self):
        self.assertEqual(self.check("code_grade_record_v1.json", self.grade_record()), [])

    def test_the_grade_schema_nulls_u_on_a_grader_error(self):
        self.assertEqual(
            self.check("code_grade_record_v1.json",
                       self.grade_record(status="grader_error", u=None)),
            [],
        )
        errors = self.check("code_grade_record_v1.json",
                            self.grade_record(status="grader_error", u=0))
        self.assertTrue(errors)

    def test_the_grade_schema_accepts_u1_which_gate_0_never_wrote(self):
        for outcome in ("u0", "u1", "u2"):
            self.assertEqual(
                self.check("code_grade_record_v1.json",
                           self.grade_record(outcome=outcome)), [],
                f"outcome {outcome} rejected")


class RouterSchemaTests(unittest.TestCase):
    SCHEMA = (Path(__file__).resolve().parents[1]
              / "protocols/router_v1/schemas/self_check_record_v1.json")

    def test_self_check_accepts_registered_null_pass2_logprobs(self):
        row = {
            "schema_version": "self-check-record-v1",
            "record_id": "r", "run_id": "run", "source_pass1_run_id": "run",
            "condition_id": "c014", "condition_hash": "a" * 64,
            "dataset_id": "math500-pilot-v1", "replicate_id": "r007",
            "stage": "pass2", "cell_id": "pilot300", "problem_id": "p",
            "call_seed": 7, "raw_output": None, "verdict": None,
            "extracted_answer": None, "answer_changed": None,
            "parse_status": "model_invalid", "token_logprobs": None,
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "latency_ms": 1.0, "finish_reason": "stop", "retries": 0},
            "created_at_utc": "2026-09-21T00:00:00Z",
        }
        schema = json.loads(self.SCHEMA.read_text(encoding="utf-8"))
        errors: list[str] = []
        checker.validate(row, schema, checker.SchemaStore(), self.SCHEMA, schema,
                         "row", errors)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
