from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import scripts.run_c002_dev30 as runner_module


ROOT = Path(__file__).resolve().parents[1]


class C004CandidateFieldRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = importlib.reload(runner_module)
        self.runner.configure_c004_candidate_field()

    def tearDown(self) -> None:
        importlib.reload(runner_module)

    def test_profile_reuses_c003_upstream_with_v3_evidence(self) -> None:
        self.assertEqual(self.runner.CONDITION_ID, "c004")
        self.assertEqual(self.runner.PROTOCOL_ID, "authority_v6")
        self.assertEqual(self.runner.REPLICATE_PLAN_ID, "r003")
        self.assertEqual(self.runner.FEATURE_SET_ID, "scalar_single_v3")
        self.assertEqual(
            self.runner.EVIDENCE_SCHEMA_VERSION,
            "controller-evidence-record-single-v3",
        )
        self.assertEqual(self.runner.RUN_REPLICATE_IDS, ("v01", "c01"))

    def test_condition_registers_exact_c003_source_run(self) -> None:
        text = (ROOT / "conditions" / "c004" / "condition.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "20260915T092801Z__c003__pilot300__r003__analysis__r003", text
        )
        self.assertIn("rerun_from: controller_evidence", text)
        self.assertIn("v_candidate_answer_mean_logprob", (
            ROOT / "protocols" / "authority_v6" / "features_scalar_single_v3.yaml"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
