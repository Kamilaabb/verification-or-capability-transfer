from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.condition_resolution import deep_merge, resolve_condition


class ConditionResolutionTests(unittest.TestCase):
    def test_deep_merge_replaces_lists_and_preserves_null(self) -> None:
        self.assertEqual(
            deep_merge(
                {"mapping": {"left": 1, "right": 2}, "items": [1, 2], "value": 3},
                {"mapping": {"right": 4}, "items": [9], "value": None},
            ),
            {"mapping": {"left": 1, "right": 4}, "items": [9], "value": None},
        )

    def test_c008_resolves_to_full_condition(self) -> None:
        resolved = resolve_condition(
            ROOT / "conditions/c008/condition.yaml", repository_root=ROOT
        )
        self.assertEqual(resolved["condition_id"], "c008")
        self.assertEqual(
            resolved["generator"]["model_id"], "Qwen/Qwen2.5-Coder-7B-Instruct"
        )
        self.assertEqual(
            resolved["controller"]["algorithm"],
            "three_independent_LogisticRegression_heads",
        )
        self.assertNotIn("extends", resolved)
        self.assertNotIn("overrides", resolved)

    def test_cycle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text(
                "schema_version: condition-overlay-v1\ncondition_id: a\nstatus: x\nextends: b.yaml\noverrides: {condition_id: a}\n",
                encoding="utf-8",
            )
            (root / "b.yaml").write_text(
                "schema_version: condition-overlay-v1\ncondition_id: b\nstatus: x\nextends: a.yaml\noverrides: {condition_id: b}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cycle"):
                resolve_condition(root / "a.yaml", repository_root=root)


if __name__ == "__main__":
    unittest.main()
