from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "run_c019", ROOT / "scripts/run_c019_v_solo_math.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402


class C019ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.c019 = resolve_condition(ROOT / "conditions/c019/condition.yaml", repository_root=ROOT)
        self.c017 = resolve_condition(ROOT / "conditions/c017/condition.yaml", repository_root=ROOT)
        self.c018 = resolve_condition(ROOT / "conditions/c018/condition.yaml", repository_root=ROOT)

    def test_only_the_model_differs_from_c017(self) -> None:
        model = runner.require_only_model_changed()
        self.assertEqual(model["model_id"], "Qwen/Qwen3-8B")
        self.assertEqual(model["revision"], self.c018["verifier"]["revision"])
        self.assertEqual(model["prompt_pass1"], self.c017["model"]["prompt_pass1"])
        self.assertEqual(model["pass1_decoding"], self.c017["model"]["pass1_decoding"])
        self.assertFalse(model["thinking"])

    def test_v_is_never_shown_g(self) -> None:
        # Pass 1 renders only the problem; the G prompt has no slot for another answer.
        prompt = (ROOT / self.c019["model"]["prompt_pass1"]).read_text(encoding="utf-8")
        self.assertIn("{{problem}}", prompt)
        self.assertNotIn("{{y0}}", prompt)
        self.assertEqual(self.c019["stages"]["run"],
                         ["pass1_with_same_forward_logit_capture", "scalar_evidence",
                          "grade", "analyze"])

    def test_seeds_and_evidence_schema_are_c019(self) -> None:
        self.assertEqual(self.c019["replicates"]["plan_id"], "r011")
        self.assertEqual(self.c019["replicates"]["seed_cell_id"], "c019")
        schema = json.loads((ROOT / self.c019["protocol"]["record_schemas"]["scalar_evidence"])
                            .read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["condition_id"], {"const": "c019"})

    def test_runner_points_c017_code_at_c019(self) -> None:
        runner.configure()
        self.assertEqual(runner.c017.CONDITION_ID, "c019")
        self.assertEqual(runner.c017.REPLICATE_PLAN_ID, "r011")
        self.assertEqual(runner.c017.SEED_CELL_ID, "c019")


if __name__ == "__main__":
    unittest.main()
