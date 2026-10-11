from __future__ import annotations

import unittest

import scripts.run_c011_gate0 as c011


class ConfigurationTests(unittest.TestCase):
    def test_configures_complete_prompt_gate0(self):
        c011.configure_c011()
        self.assertEqual(c011.runner.CONDITION_ID, "c011")
        self.assertEqual(c011.runner.DATASET_ID, "bigcodebench-b74c0d0-complete-s20260917")
        self.assertEqual(c011.runner.PROBLEM_FIELD, "complete_prompt")
        self.assertEqual(c011.runner.REPLICATE_PLAN_ID, "r004")
        self.assertEqual(c011.runner.STAGE, "gate0")
        self.assertEqual(c011.runner.EXPECTED_ROWS, 300)
        self.assertEqual(c011.runner.G_RECORD_SCHEMA_VERSION, "g-record-single-v3")
        self.assertEqual(c011.runner.V_RECORD_SCHEMA_VERSION, "v-record-v3")

    def test_resolved_condition_registers_the_same_problem_field(self):
        c011.configure_c011()
        condition = c011.runner.load_condition()
        self.assertEqual(condition["condition_id"], "c011")
        self.assertEqual(condition["dataset"]["prompt_field"], "complete_prompt")
        self.assertEqual(condition["generator"]["problem_field"], "complete_prompt")


if __name__ == "__main__":
    unittest.main()
