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
    "run_c020", ROOT / "scripts/run_c020_c018_evidence_rederive.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

SCHEMAS = ROOT / "protocols/math_verifier_scalar_v1/schemas"


def load(condition: str) -> dict:
    return yaml.safe_load((ROOT / f"conditions/{condition}/condition.yaml").read_text(encoding="utf-8"))


class C020ContractTests(unittest.TestCase):
    def test_c020_reuses_c018_identity_and_calls_no_model(self) -> None:
        c018, c020 = load("c018"), load("c020")
        self.assertEqual(c020["condition_id"], "c020")
        for section in ("protocol", "dataset", "generator_source", "verifier", "features",
                        "replicates", "analysis", "hash_policy", "change_policy"):
            if section == "protocol":
                schemas = dict(c020[section]["record_schemas"])
                self.assertEqual(schemas.pop("scalar_evidence"),
                                 "protocols/math_verifier_scalar_v1/schemas/evidence_record_c020_v1.json")
                expected = dict(c018[section]["record_schemas"])
                expected.pop("scalar_evidence")
                self.assertEqual(schemas, expected)
                continue
            self.assertEqual(c020[section], c018[section], section)
        self.assertIn("verify", c020["stages"]["skip"])
        self.assertEqual(c020["rederivation"]["expected_changed_features"],
                         ["gv_final_answer_equivalent"])
        self.assertEqual(c020["verifier_source"]["condition_id"], "c018")

    def test_c020_evidence_schema_differs_from_c018_only_in_identity(self) -> None:
        base = json.loads((SCHEMAS / "evidence_record_v1.json").read_text(encoding="utf-8"))
        c020 = json.loads((SCHEMAS / "evidence_record_c020_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(c020["properties"]["condition_id"], {"const": "c020"})
        base["$id"], base["properties"]["condition_id"] = c020["$id"], {"const": "c020"}
        self.assertEqual(c020, base)

    def test_wrapper_points_the_c018_runner_at_c020_and_refuses_model_stages(self) -> None:
        # configure() rebinds module globals shared across the test process; restore them.
        saved = (runner.c018.CONDITION_ID, runner.c018.CONDITION_PATH, list(sys.argv))
        try:
            runner.configure()
            self.assertEqual(runner.c018.CONDITION_ID, "c020")
            self.assertEqual(runner.c018.CONDITION_PATH, runner.CONDITION_PATH)
            for command in ("smoke", "verify"):
                sys.argv = ["run_c020", command]
                with self.assertRaises(SystemExit):
                    runner.main()
        finally:
            runner.c018.CONDITION_ID, runner.c018.CONDITION_PATH, sys.argv[:] = saved


class CompareTests(unittest.TestCase):
    ROW = {
        "record_id": "a", "run_id": "r", "condition_id": "c018", "condition_hash": "h",
        "created_at_utc": "t", "problem_id": "p1", "location_match_status": "exact",
        "features": {"gv_final_answer_equivalent": None, "v_verdict_correct": 1},
        "missingness": {"gv_final_answer_equivalent": True, "v_verdict_correct": False},
    }

    def with_changes(self, **features) -> dict:
        row = json.loads(json.dumps(self.ROW))
        row.update({"record_id": "b", "run_id": "r2", "condition_id": "c020",
                    "condition_hash": "h2", "created_at_utc": "t2"})
        for name, value in features.items():
            row["features"][name] = value
            row["missingness"][name] = value is None
        return row

    def test_run_identity_and_the_fixed_feature_may_differ(self) -> None:
        new = self.with_changes(gv_final_answer_equivalent=True)
        self.assertEqual(runner.unexpected_differences(new, self.ROW), [])

    def test_any_other_difference_is_reported(self) -> None:
        changed = self.with_changes(v_verdict_correct=0)
        self.assertEqual(runner.unexpected_differences(changed, self.ROW),
                         ["features.v_verdict_correct"])
        dropped = self.with_changes(v_verdict_correct=None)
        self.assertEqual(runner.unexpected_differences(dropped, self.ROW),
                         ["features.v_verdict_correct", "missingness.v_verdict_correct"])
        moved = dict(self.ROW, location_match_status="not_found")
        self.assertEqual(runner.unexpected_differences(moved, self.ROW), ["location_match_status"])


if __name__ == "__main__":
    unittest.main()
