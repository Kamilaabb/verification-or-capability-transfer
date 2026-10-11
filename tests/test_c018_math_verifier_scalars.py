from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "run_c018", ROOT / "scripts/run_c018_math_verifier_scalars.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

from authority_allocation import verifier_scalar_features as runner_features  # noqa: E402


class C018ContractTests(unittest.TestCase):
    def test_condition_freezes_requested_model_and_decoding(self) -> None:
        condition = yaml.safe_load(
            (ROOT / "conditions/c018/condition.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(condition["generator_source"]["condition_id"], "c017")
        self.assertEqual(condition["verifier"]["model_id"], "Qwen/Qwen3-8B")
        self.assertFalse(condition["verifier"]["thinking"])
        self.assertEqual(condition["verifier"]["decoding"]["temperature"], 0.0)
        self.assertIsNone(condition["verifier"]["decoding"]["constrained_schema"])
        self.assertEqual(condition["features"]["count"], 29)
        self.assertEqual(condition["replicates"]["plan_id"], "r010")

    def test_evidence_schema_matches_feature_contract(self) -> None:
        schema = json.loads((
            ROOT / "protocols/math_verifier_scalar_v1/schemas/evidence_record_v1.json"
        ).read_text(encoding="utf-8"))
        properties = schema["$defs"]["features"]["properties"]
        missingness = schema["$defs"]["missingness"]["properties"]
        self.assertEqual(set(properties), set(runner.FEATURE_NAMES))
        self.assertEqual(set(missingness), set(runner.FEATURE_NAMES))
        self.assertEqual(len(properties), 29)
        self.assertEqual(
            schema["properties"]["location_match_status"]["enum"],
            list(runner_features.LOCATION_STATUSES),
        )

    def test_feature_contract_lists_the_implemented_features_in_order(self) -> None:
        contract = yaml.safe_load((
            ROOT / "protocols/math_verifier_scalar_v1/features.yaml"
        ).read_text(encoding="utf-8"))
        names = [feature["name"] for feature in contract["features"]]
        self.assertEqual(names, list(runner.FEATURE_NAMES))
        self.assertEqual(contract["feature_count"], len(names))

    def test_v_usage_matches_record_schema(self) -> None:
        # Job 65117997 produced 300 V rows that all failed validation because the
        # usage block dropped reasoning_tokens.
        schema = json.loads((
            ROOT / "protocols/math_verifier_scalar_v1/schemas/v_record_v1.json"
        ).read_text(encoding="utf-8"))
        usage_schema = schema["$defs"]["usage"]
        usage = runner.v_usage({
            "input_tokens": 1, "output_tokens": 2, "reasoning_tokens": None,
            "latency_ms": 3.0, "finish_reason": "stop",
        })
        self.assertEqual(set(usage), set(usage_schema["required"]))
        self.assertEqual(set(usage), set(usage_schema["properties"]))

    def test_runnable_gate(self) -> None:
        model = runner.require_runnable()
        self.assertEqual(model["revision"], "b968826d9c46dd6066d109eabc6255188de91218")


if __name__ == "__main__":
    unittest.main()
