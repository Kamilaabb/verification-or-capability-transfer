from __future__ import annotations

import unittest

import scripts.run_c009_gate0 as c009


class ConfigurationTests(unittest.TestCase):
    def test_configures_cross_family_gate0(self):
        c009.configure_c009()
        self.assertEqual(c009.runner.CONDITION_ID, "c009")
        self.assertEqual(c009.runner.REPLICATE_PLAN_ID, "r004")
        self.assertEqual(c009.runner.STAGE, "gate0")
        self.assertEqual(c009.runner.EXPECTED_ROWS, 300)
        self.assertEqual(c009.runner.G_RECORD_SCHEMA_VERSION, "g-record-single-v3")
        self.assertEqual(c009.runner.V_RECORD_SCHEMA_VERSION, "v-record-v3")


if __name__ == "__main__":
    unittest.main()
