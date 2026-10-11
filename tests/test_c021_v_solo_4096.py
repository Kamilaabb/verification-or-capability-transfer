from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "run_c021", ROOT / "scripts/run_c021_v_solo_4096.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402


class C021ContractTests(unittest.TestCase):
    def test_only_the_budget_differs_from_c019(self) -> None:
        model = runner.require_only_budget_changed()
        c018 = resolve_condition(ROOT / "conditions/c018/condition.yaml", repository_root=ROOT)
        self.assertEqual(model["pass1_decoding"]["max_new_tokens"],
                         c018["verifier"]["decoding"]["max_new_tokens"])
        self.assertEqual(model["model_id"], "Qwen/Qwen3-8B")

    def test_seeds_are_c019s(self) -> None:
        c021 = resolve_condition(ROOT / "conditions/c021/condition.yaml", repository_root=ROOT)
        self.assertEqual(c021["replicates"]["plan_id"], "r011")
        self.assertEqual(c021["replicates"]["seed_cell_id"], "c019")
        runner.configure()
        self.assertEqual(runner.c017.CONDITION_ID, "c021")
        self.assertEqual(runner.c017.SEED_CELL_ID, "c019")


if __name__ == "__main__":
    unittest.main()
