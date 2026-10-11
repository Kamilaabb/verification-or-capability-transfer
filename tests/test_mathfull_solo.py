from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location("run_mathfull_solo", ROOT / "scripts/run_mathfull_solo.py")
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class MathfullSoloContractTests(unittest.TestCase):
    def test_both_conditions_differ_only_where_registered(self) -> None:
        g = runner.require_registered_differences("c022")
        v = runner.require_registered_differences("c023")
        self.assertEqual(g["model_id"], "Qwen/Qwen3-4B")
        self.assertEqual(v["model_id"], "Qwen/Qwen3-8B")
        self.assertEqual(g["prompt_pass1"], v["prompt_pass1"])
        self.assertEqual(g["pass1_decoding"], v["pass1_decoding"])
        self.assertEqual(g["pass1_decoding"]["max_new_tokens"], 4096)

    def test_prompt_v2_conditions_differ_from_c022_c023_only_in_the_prompt(self) -> None:
        g, v = runner.require_registered_differences("c025"), runner.require_registered_differences("c026")
        old_g, old_v = runner.require_registered_differences("c022"), runner.require_registered_differences("c023")
        self.assertEqual(g["prompt_pass1"], v["prompt_pass1"])
        self.assertNotEqual(g["prompt_pass1"], old_g["prompt_pass1"])
        for new, old in ((g, old_g), (v, old_v)):
            changed = {k for k in set(new) | set(old) if new.get(k) != old.get(k)}
            self.assertEqual(changed, {"prompt_pass1", "prompt_pass1_provenance"})

    def test_c029_swaps_in_llama_and_pins_the_template_date(self) -> None:
        llama = runner.require_registered_differences("c029")
        qwen = runner.require_registered_differences("c025")
        self.assertEqual(llama["model_id"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(llama["prompt_pass1"], qwen["prompt_pass1"])
        self.assertEqual(llama["pass1_decoding"], qwen["pass1_decoding"])
        self.assertEqual(llama["chat_template_kwargs"], {"date_string": "26 Jul 2024"})
        changed = {k for k in set(llama) | set(qwen) if llama.get(k) != qwen.get(k)}
        self.assertLessEqual(changed, runner.MODEL_IDENTITY_KEYS)

    def test_c067_swaps_in_llama31_8b_and_pins_the_template_date(self) -> None:
        llama = runner.require_registered_differences("c067")
        qwen = runner.require_registered_differences("c025")
        self.assertEqual(llama["model_id"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(llama["prompt_pass1"], qwen["prompt_pass1"])
        self.assertEqual(llama["pass1_decoding"], qwen["pass1_decoding"])
        self.assertEqual(llama["chat_template_kwargs"], {"date_string": "26 Jul 2024"})
        changed = {k for k in set(llama) | set(qwen) if llama.get(k) != qwen.get(k)}
        self.assertLessEqual(changed, runner.MODEL_IDENTITY_KEYS)

    def test_c030_swaps_in_qwen25_3b_with_the_registered_penalty(self) -> None:
        small = runner.require_registered_differences("c030")
        qwen = runner.require_registered_differences("c025")
        self.assertEqual(small["model_id"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(small["prompt_pass1"], qwen["prompt_pass1"])
        self.assertEqual(small["pass1_decoding"], qwen["pass1_decoding"])
        self.assertEqual(small["pass1_decoding"]["repetition_penalty"], 1.0)
        self.assertNotIn("chat_template_kwargs", small)
        changed = {k for k in set(small) | set(qwen) if small.get(k) != qwen.get(k)}
        self.assertLessEqual(changed, runner.MODEL_IDENTITY_KEYS)

    def test_configure_points_c017_code_at_mathfull(self) -> None:
        runner.configure("c023")
        self.assertEqual(runner.c017.CONDITION_ID, "c023")
        self.assertEqual(runner.c017.DATASET_ID, "mathfull-r465bcdb-l3to5-s20260924")
        self.assertEqual(runner.c017.EXPECTED_ROWS, 1000)
        self.assertIs(runner.base.load_source_rows, runner.load_mathfull_rows)


class TestConditionTests(unittest.TestCase):
    def test_test_conditions_change_only_the_manifest(self) -> None:
        for test_id, dev_id, rows, name in (("c040", "c025", 3298, "test"), ("c041", "c026", 3298, "test"),
                                            ("c056", "c029", 1500, "test1500"), ("c057", "c030", 1500, "test1500"),
                                            ("c062", "c025", 1500, "test1500"), ("c063", "c031", 1500, "test1500"),
                                            ("c092", "c026", 1500, "test1500"), ("c093", "c054", 1500, "test1500"),
                                            ("c096", "c032", 1500, "test1500"), ("c097", "c055", 1500, "test1500"),
                                            ("c098", "c066", 1500, "test1500"), ("c068", "c067", 1500, "test1500")):
            test_model = runner.require_registered_differences(test_id)
            dev_model = runner.require_registered_differences(dev_id)
            self.assertEqual(test_model, dev_model)
            info = runner.manifest_info(ROOT / f"conditions/{test_id}/condition.yaml")
            self.assertEqual(info["rows"], rows)
            self.assertTrue(info["path"].endswith(f"/{name}.jsonl"))

    def test_testpool_dev_conditions_change_only_the_manifest(self) -> None:
        for new_id, dev_id in (("c200", "c025"), ("c201", "c026"), ("c202", "c030"), ("c203", "c031"), ("c204", "c032"), ("c205", "c054"), ("c206", "c055"), ("c207", "c067"), ("c208", "c029"), ("c209", "c066")):
            self.assertEqual(runner.require_registered_differences(new_id),
                             runner.require_registered_differences(dev_id))
            self.assertNotIn("test", runner.CONDITIONS[new_id])
            info = runner.manifest_info(ROOT / f"conditions/{new_id}/condition.yaml")
            self.assertEqual(info["rows"], 1000)
            self.assertTrue(info["path"].endswith("/testpool_dev1000.jsonl"))

    def test_olympiad_conditions_change_only_the_dataset(self) -> None:
        for new_id, dev_id in (("c254", "c204"), ("c255", "c201"), ("c259", "c200")):
            self.assertEqual(runner.require_registered_differences(new_id),
                             runner.require_registered_differences(dev_id))
            info = runner.manifest_info(ROOT / f"conditions/{new_id}/condition.yaml")
            self.assertEqual(info["rows"], 571)
            self.assertTrue(info["path"].endswith(f"/{runner.OLYMPIAD_ID}/eval.jsonl"))
            runner.configure(new_id)
            try:
                self.assertEqual(runner.DATASET_ID, runner.OLYMPIAD_ID)
                self.assertIs(runner.base.load_source_rows, runner.load_olympiad_rows)
            finally:
                runner.configure("c025")
            self.assertEqual(runner.DATASET_ID, runner.MATHFULL_ID)

    def test_olympiad_gold_strips_only_outer_dollars(self) -> None:
        self.assertEqual(runner.olympiad_gold([r"$\frac{1}{2}$"]), r"\frac{1}{2}")
        self.assertEqual(runner.olympiad_gold([" 12 "]), "12")
        self.assertEqual(runner.olympiad_gold(["(1,2)"]), "(1,2)")
        with self.assertRaises(ValueError):
            runner.olympiad_gold(["1", "2"])

    def test_a_dev_condition_can_never_point_at_test(self) -> None:
        self.assertNotIn("test", runner.CONDITIONS["c025"])
        self.assertTrue(runner.CONDITIONS["c040"]["test"])


class ShardTests(unittest.TestCase):
    def manifest(self, tmp: Path, n: int) -> Path:
        path = tmp / "full.jsonl"
        path.write_text("".join(json.dumps({"problem_id": f"p{i}"}) + "\n" for i in range(n)))
        return path

    def test_bounds_cover_every_row_once_in_order(self) -> None:
        for rows, count in ((3298, 4), (10, 3), (5, 5), (7, 1)):
            blocks = [runner.shard_bounds(rows, i, count) for i in range(count)]
            self.assertEqual(blocks[0][0], 0)
            self.assertEqual(blocks[-1][1], rows)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(blocks, blocks[1:])))
            self.assertLessEqual(max(e - s for s, e in blocks) - min(e - s for s, e in blocks), 1)
        with self.assertRaises(SystemExit):
            runner.shard_bounds(10, 3, 3)

    def write_shards(self, tmp: Path, full: Path, count: int, drop: str | None = None,
                     bad_status: str | None = None) -> Path:
        root = tmp / "shards"
        for i in range(count):
            (root / str(i) / "records").mkdir(parents=True)
            shard = root / str(i) / "manifest.jsonl"
            runner.write_shard_manifest(full, i, count, shard)
            ids = [json.loads(l)["problem_id"] for l in shard.read_text().splitlines()]
            ids = [p for p in reversed(ids) if p != drop]           # out of order on purpose
            (root / str(i) / "records" / "pass1.jsonl").write_text(
                "".join(json.dumps({"problem_id": p, "x": 1}) + "\n" for p in ids))
            (root / str(i) / "records" / "native_logit_trace.jsonl").write_text(
                "".join(json.dumps({"problem_id": p, "status": "bad" if p == bad_status else "ok"}) + "\n"
                        for p in ids))
        return root

    def test_merge_restores_manifest_order_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            full = self.manifest(tmp, 11)
            root = self.write_shards(tmp, full, 3)
            runner.merge_shards(full, root, 3, tmp / "out", ["pass1.jsonl", "native_logit_trace.jsonl"])
            merged = [json.loads(l) for l in (tmp / "out" / "pass1.jsonl").read_text().splitlines()]
            self.assertEqual([r["problem_id"] for r in merged], [f"p{i}" for i in range(11)])
            first = (tmp / "out" / "pass1.jsonl").read_text().splitlines()[0]
            self.assertEqual(first, json.dumps({"problem_id": "p0", "x": 1}))

    def test_merge_refuses_a_missing_row_or_a_failed_trace(self) -> None:
        for kwargs in ({"drop": "p4"}, {"bad_status": "p7"}):
            with tempfile.TemporaryDirectory() as t:
                tmp = Path(t)
                full = self.manifest(tmp, 11)
                root = self.write_shards(tmp, full, 3, **kwargs)
                with self.assertRaises(SystemExit):
                    runner.merge_shards(full, root, 3, tmp / "out", ["pass1.jsonl", "native_logit_trace.jsonl"])


class ReconcileTests(unittest.TestCase):
    def write(self, path: Path, rows: list[dict], torn: str = "") -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows) + torn, encoding="utf-8")

    def test_keeps_only_rows_complete_in_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pass1, trace = Path(tmp) / "pass1.jsonl", Path(tmp) / "trace.jsonl"
            self.write(pass1, [{"problem_id": "a"}, {"problem_id": "b"}, {"problem_id": "c"}])
            # b's trace failed, c's trace was never written, and the file ends mid-line.
            self.write(trace, [{"problem_id": "a", "status": "ok"},
                               {"problem_id": "b", "status": "token_mismatch"}],
                       torn='{"problem_id": "c", "sta')
            report = runner.reconcile(pass1, trace)
            self.assertEqual(report["kept"], 1)
            self.assertEqual([json.loads(l)["problem_id"] for l in pass1.read_text().splitlines()], ["a"])
            self.assertEqual([json.loads(l)["problem_id"] for l in trace.read_text().splitlines()], ["a"])

    def test_missing_files_are_a_fresh_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = runner.reconcile(Path(tmp) / "p.jsonl", Path(tmp) / "t.jsonl")
            self.assertEqual(report["kept"], 0)


class UngradeableAnalysisTests(unittest.TestCase):
    LISTED = "train/prealgebra/281.json"

    def build(self, tmp: Path, grades: dict[str, tuple]) -> list[str]:
        ids = list(grades)
        feats = {"g_answer_parseable": 1, "g_math_expression_mean_max_logit": 1.0}
        rows = {
            "pass1": [{"problem_id": p, "outputs": [{"usage": {"output_tokens": 5},
                                                     "parse_status": "ok"}]} for p in ids],
            "evidence": [{"problem_id": p, "features": feats, "math_spans": [],
                          "math_token_count": 1, "output_token_count": 5} for p in ids],
            "grades": [{"problem_id": p, "u": u, "grader_status": s} for p, (u, s) in grades.items()],
        }
        for name, data in rows.items():
            (tmp / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
        return ["--run-id", "r", "--pass1-records", str(tmp / "pass1.jsonl"),
                "--evidence", str(tmp / "evidence.jsonl"), "--grades", str(tmp / "grades.jsonl"),
                "--expected-rows", str(len(ids)), "--output", str(tmp / "out" / "analysis.json")]

    def test_the_registered_gold_is_excluded_and_invalid_output_counts_wrong(self) -> None:
        self.assertIn(self.LISTED, runner.ungradeable_ids())
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            argv = self.build(tmp, {self.LISTED: (None, "grader_error"), "a": (1, "ok"),
                                    "b": (0, "model_invalid")})
            runner.analyze(argv)
            report = json.loads((tmp / "out" / "analysis.json").read_text())
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["correct"], 1)
            self.assertEqual(report["excluded_ungradeable_gold"], [self.LISTED])
            self.assertIn("excluded as ungradeable gold", (tmp / "out" / "summary.md").read_text())

    def test_an_unlisted_grader_error_stops_the_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            argv = self.build(Path(t), {"a": (1, "ok"), "b": (None, "grader_error")})
            with self.assertRaises(SystemExit):
                runner.analyze(argv)


if __name__ == "__main__":
    unittest.main()
