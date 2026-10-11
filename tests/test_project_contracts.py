from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "protocols" / "authority_v5" / "schemas"
V6_SCHEMAS = ROOT / "protocols" / "authority_v6" / "schemas"
V6_V3_SCHEMAS = ROOT / "protocols" / "authority_v6" / "schemas_scalar_single_v3"

FEATURES = {
    "g_support_y0_valid",
    "g_mode_share_valid",
    "g_answer_entropy_norm",
    "g_valid_fraction",
    "g_y0_is_mode",
    "v_verdict_correct",
    "v_verdict_incorrect",
    "v_verdict_uncertain",
    "v_request_keep",
    "v_request_revise",
    "v_request_override",
    "gv_final_answer_equivalent",
    "gv_support_y2_valid",
    "gv_y2_is_g_mode",
}

SINGLE_FEATURES = {
    "g_answer_extractable",
    "v_verdict_correct",
    "v_verdict_incorrect",
    "v_verdict_uncertain",
    "v_request_keep",
    "v_request_revise",
    "v_request_override",
    "v_candidate_extractable",
    "gv_final_answer_equivalent",
}

SINGLE_V2_FEATURES = SINGLE_FEATURES | {
    "g_output_mean_logprob",
    "g_final_answer_mean_logprob",
    "v_verdict_mean_logprob",
    "v_request_mean_logprob",
    "v_candidate_final_answer_mean_logprob",
    "v_critique_token_length_log",
    "gv_answer_type_match",
}

SINGLE_V3_FEATURES = (
    SINGLE_V2_FEATURES - {"v_candidate_final_answer_mean_logprob"}
) | {"v_candidate_answer_mean_logprob"}


class ProjectContractTests(unittest.TestCase):
    def load_schema(self, name: str) -> dict:
        return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))

    def test_all_schemas_are_valid_json(self) -> None:
        files = list(SCHEMAS.glob("*.json"))
        self.assertEqual(len(files), 14)
        for path in files:
            with self.subTest(path=path.name):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_controller_evidence_is_an_exact_closed_feature_set(self) -> None:
        schema = self.load_schema("controller_evidence_record_v1.json")
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["features"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["missingness"]["additionalProperties"])
        self.assertEqual(set(schema["$defs"]["features"]["properties"]), FEATURES)
        self.assertEqual(set(schema["$defs"]["missingness"]["properties"]), FEATURES)

    def test_controller_decision_contains_no_outcome(self) -> None:
        properties = self.load_schema("controller_decision_record_v1.json")["properties"]
        forbidden = {"final_correct", "u0", "u1", "u2", "z0", "z1", "z2", "gold", "evidence"}
        self.assertFalse(forbidden.intersection(properties))

    def test_single_sample_evidence_is_exact_and_closed(self) -> None:
        schema = self.load_schema("controller_evidence_record_single_v1.json")
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["features"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["missingness"]["additionalProperties"])
        self.assertEqual(set(schema["$defs"]["features"]["properties"]), SINGLE_FEATURES)
        self.assertEqual(set(schema["$defs"]["missingness"]["properties"]), SINGLE_FEATURES)

    def test_single_sample_g_schema_has_exactly_one_output(self) -> None:
        outputs = self.load_schema("g_record_single_v1.json")["properties"]["outputs"]
        self.assertEqual(outputs["minItems"], 1)
        self.assertEqual(outputs["maxItems"], 1)

    def test_v2_evidence_is_exact_closed_and_sixteen_dimensional(self) -> None:
        schema = json.loads(
            (V6_SCHEMAS / "controller_evidence_record_single_v2.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["features"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["missingness"]["additionalProperties"])
        self.assertEqual(
            set(schema["$defs"]["features"]["properties"]), SINGLE_V2_FEATURES
        )
        self.assertEqual(
            set(schema["$defs"]["missingness"]["properties"]), SINGLE_V2_FEATURES
        )

    def test_training_target_preserves_three_cap_outcomes(self) -> None:
        schema = json.loads(
            (V6_SCHEMAS / "controller_training_target_v1.json").read_text(
                encoding="utf-8"
            )
        )
        outcomes = schema["properties"]["cap_outcomes"]
        self.assertFalse(outcomes["additionalProperties"])
        self.assertEqual(set(outcomes["required"]), {"M0", "M1", "M2"})
        self.assertEqual(set(outcomes["properties"]), {"M0", "M1", "M2"})

    def test_v3_replaces_only_the_candidate_logprob_field(self) -> None:
        schema = json.loads(
            (V6_V3_SCHEMAS / "controller_evidence_record_single_v3.json").read_text(
                encoding="utf-8"
            )
        )
        features = schema["$defs"]["features"]
        missingness = schema["$defs"]["missingness"]
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(features["additionalProperties"])
        self.assertFalse(missingness["additionalProperties"])
        self.assertEqual(set(features["properties"]), SINGLE_V3_FEATURES)
        self.assertEqual(set(missingness["properties"]), SINGLE_V3_FEATURES)
        self.assertNotIn("v_candidate_final_answer_mean_logprob", features["properties"])

    def test_controller_input_contains_no_record_identifiers(self) -> None:
        schema = self.load_schema("controller_input_payload_v1.json")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), {"features", "missingness"})

    def test_binary_and_ternary_controller_menus_are_distinct(self) -> None:
        binary = self.load_schema("controller_binary_payload_v1.json")
        ternary = self.load_schema("controller_ternary_payload_v1.json")
        self.assertEqual(binary["properties"]["authority_cap"]["enum"], ["M0", "M2"])
        self.assertEqual(ternary["properties"]["authority_cap"]["enum"], ["M0", "M1", "M2"])

    def test_condition_freezes_y0_and_exposes_thinking_modes(self) -> None:
        text = (ROOT / "conditions" / "c002" / "condition.yaml").read_text(encoding="utf-8")
        self.assertIn("y0_selection: sample_index_0_unconditionally", text)
        self.assertNotIn("first_complete_output", text)
        self.assertIn("samples: 1", text)
        self.assertIn("thinking: false", text)
        self.assertGreaterEqual(text.count("thinking: true"), 2)

    def test_dataset_is_server_only(self) -> None:
        text = (ROOT / "datasets" / "math500-r6e4ed1-l3to5-s20260910" / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("mode: server_only", text)
        self.assertIn("local_files_required: false", text)

    def test_development300_reuses_manifest_but_allows_tuning(self) -> None:
        old_text = (ROOT / "datasets" / "math500-r6e4ed1-l3to5-s20260910" / "dataset.yaml").read_text(encoding="utf-8")
        new_text = (ROOT / "datasets" / "math500-r6e4ed1-l3to5-dev300-s20260910" / "dataset.yaml").read_text(encoding="utf-8")
        manifest_hash = "82b81514679262eac8c3e970a774ab083d769df1c38b0f43a470ef7927c90971"
        self.assertIn(manifest_hash, old_text)
        self.assertIn(manifest_hash, new_text)
        self.assertIn("development_may_tune_design: true", new_text)
        self.assertIn("may_support_confirmatory_claims: false", new_text)

    def test_base_seed_is_part_of_call_seed_derivation(self) -> None:
        text = (ROOT / "replicates" / "r001.yaml").read_text(encoding="utf-8")
        ordered_fields = text.split("ordered_fields:", 1)[1].split(
            "menu_excluded_so_c2_c3_are_paired:", 1
        )[0]
        self.assertIn("- base_seed", ordered_fields)

    def test_grader_exceptions_cannot_be_scored_as_wrong(self) -> None:
        text = (ROOT / "protocols" / "authority_v5" / "grader_math_verify_v1.yaml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(text.count("raise_on_error: true"), 2)
        self.assertIn("grader_exception: null", text)
        self.assertIn("grader_timeout: null", text)

    def test_h2_reports_each_fixed_baseline(self) -> None:
        text = (ROOT / "protocols" / "authority_v5" / "REPORT_TEMPLATE.md").read_text(
            encoding="utf-8"
        )
        for baseline in ("M0", "M1", "M2"):
            self.assertIn(f"C3 − {baseline}", text)


if __name__ == "__main__":
    unittest.main()
