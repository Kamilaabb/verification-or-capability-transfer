from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402
import run_c014_router as runner  # noqa: E402
import run_c015_router as c015  # noqa: E402


class C015ConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = runner.CONDITION_ID, runner.CONDITION_PATH, runner.SEED_CELL_ID

    def tearDown(self) -> None:
        runner.CONDITION_ID, runner.CONDITION_PATH, runner.SEED_CELL_ID = self.identity
        runner.configure()

    def test_only_pass2_prompt_changes_from_c014_behavior(self):
        c014_condition = resolve_condition(
            ROOT / "conditions/c014/condition.yaml", repository_root=ROOT)
        c015_condition = resolve_condition(
            ROOT / "conditions/c015/condition.yaml", repository_root=ROOT)
        self.assertEqual(c015_condition["condition_id"], "c015")
        self.assertEqual(c015_condition["model"]["prompt_pass1"],
                         c014_condition["model"]["prompt_pass1"])
        self.assertEqual(c015_condition["model"]["pass1_decoding"],
                         c014_condition["model"]["pass1_decoding"])
        self.assertEqual(c015_condition["model"]["pass2_decoding"],
                         c014_condition["model"]["pass2_decoding"])
        self.assertNotEqual(c015_condition["model"]["prompt_pass2"],
                            c014_condition["model"]["prompt_pass2"])

    def test_prompt_requires_two_executed_independent_checks(self):
        c015.configure_c015()
        message = runner.pass2_messages("problem", "candidate")[0]["content"]
        self.assertIn("at least two independent checks", message)
        self.assertIn("Actually carry out", message)
        self.assertIn("does not count as two checks", message)
        self.assertEqual(runner.SEED_CELL_ID, "c014")
        self.assertEqual(runner.CONDITION_ID, "c015")


if __name__ == "__main__":
    unittest.main()
