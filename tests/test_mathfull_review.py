from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location("run_mathfull_review", ROOT / "scripts/run_mathfull_review.py")
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class ReviewContractTests(unittest.TestCase):
    def test_c024_differs_from_c023_only_where_registered(self) -> None:
        model = runner.require_registered_differences()
        self.assertEqual(model["model_id"], "Qwen/Qwen3-8B")
        self.assertEqual(model["pass1_decoding"]["max_new_tokens"], 4096)
        self.assertFalse(model["thinking"])

    def test_c027_differs_from_c024_only_in_prompt_and_g_source(self) -> None:
        runner.configure("c027")
        try:
            model = runner.require_registered_differences()
            cond = runner.condition()
        finally:
            runner.configure("c024")
        self.assertEqual(model["prompt_pass1"], "conditions/c027/prompts/v_reviewer_v2.txt")
        self.assertEqual(cond["generator_source"]["condition_id"], "c025")
        self.assertTrue(cond["generator_source"]["pass1_run_id"].startswith(
            Path(cond["generator_source"]["artifact_path"]).name + "__pass1__"))
        old = runner.condition()
        for section in ("delivery", "dataset", "features"):
            self.assertEqual(cond[section], old[section])
        changed = {k for k in set(cond["model"]) | set(old["model"])
                   if cond["model"].get(k) != old["model"].get(k)}
        self.assertEqual(changed, {"prompt_pass1", "prompt_pass1_provenance"})

    def test_c028_swaps_the_models_under_the_same_frozen_prompt(self) -> None:
        runner.configure("c028")
        try:
            model = runner.require_registered_differences()
            cond = runner.condition()
        finally:
            runner.configure("c024")
        c025 = runner.resolve_condition(ROOT / "conditions/c025/condition.yaml", repository_root=ROOT)
        runner.configure("c027")
        try:
            c027 = runner.condition()
        finally:
            runner.configure("c024")
        self.assertEqual(model["model_id"], "Qwen/Qwen3-4B")
        self.assertEqual(model["revision"], c025["model"]["revision"])
        self.assertEqual(model["prompt_pass1"], c027["model"]["prompt_pass1"])
        self.assertEqual(model["pass1_decoding"], c027["model"]["pass1_decoding"])
        self.assertFalse(model["thinking"])
        self.assertEqual(cond["generator_source"]["condition_id"], "c026")
        self.assertEqual(cond["generator_source"]["model_id"], "Qwen/Qwen3-8B")
        self.assertEqual(cond["delivery"], c027["delivery"])

    def test_test_reviews_change_only_the_manifest_and_g_source(self) -> None:
        for test_id, dev_id, g, name in (("c042", "c027", "c040", "test"), ("c043", "c028", "c041", "test"),
                                         ("c058", "c044", "c056", "test1500"), ("c059", "c045", "c057", "test1500"),
                                         ("c064", "c060", "c062", "test1500"), ("c065", "c061", "c063", "test1500"),
                                         ("c094", "c090", "c092", "test1500"), ("c095", "c091", "c093", "test1500"),
                                         ("c152", "c150", "c096", "test1500"), ("c153", "c151", "c057", "test1500"),
                                         ("c156", "c154", "c063", "test1500"), ("c157", "c155", "c092", "test1500"),
                                         ("c230", "c210", "c093", "test1500"),
                                         ("c231", "c211", "c062", "test1500"),
                                         ("c232", "c212", "c057", "test1500"),
                                         ("c233", "c213", "c097", "test1500"),
                                         ("c234", "c214", "c096", "test1500"),
                                         ("c235", "c215", "c092", "test1500"),
                                         ("c236", "c216", "c068", "test1500"),
                                         ("c237", "c217", "c092", "test1500"),
                                         ("c238", "c218", "c093", "test1500"),
                                         ("c239", "c219", "c063", "test1500"),
                                         ("c240", "c220", "c057", "test1500"),
                                         ("c241", "c221", "c063", "test1500"),
                                         ("c242", "c224", "c056", "test1500"),
                                         ("c243", "c225", "c063", "test1500"),
                                         ("c244", "c226", "c056", "test1500"),
                                         ("c245", "c227", "c068", "test1500"),
                                         ("c246", "c228", "c056", "test1500"),
                                         ("c247", "c229", "c096", "test1500"),
                                         ("c252", "c250", "c092", "test1500"),
                                         ("c253", "c251", "c097", "test1500"),
                                         ("c266", "c262", "c098", "test1500"),
                                         ("c267", "c263", "c062", "test1500"),
                                         ("c268", "c264", "c098", "test1500"),
                                         ("c269", "c265", "c097", "test1500"),
                                         ("c386", "c216", "c092", "test1500"),
                                         ("c387", "c210", "c062", "test1500")):
            runner.configure(dev_id)
            try:
                dev_model, dev = runner.require_registered_differences(), runner.condition()
                runner.configure(test_id)
                test_model, test = runner.require_registered_differences(), runner.condition()
            finally:
                runner.configure("c024")
            self.assertEqual(test_model, dev_model)
            self.assertEqual(test["delivery"], dev["delivery"])
            self.assertEqual(test["dataset"]["manifest"], name)
            self.assertEqual(test["generator_source"]["condition_id"], g)
            self.assertIn(f"__{g}__mathfull-{name}__", test["generator_source"]["artifact_path"])

    def test_prompt_variant_reviews_change_only_the_prompt(self) -> None:
        for cid, base_id in (("c390", "c236"), ("c391", "c253")):
            runner.configure(base_id)
            try:
                base_model, base_cond = runner.require_registered_differences(), runner.condition()
                runner.configure(cid)
                model, cond = runner.require_registered_differences(), runner.condition()
            finally:
                runner.configure("c024")
            self.assertEqual(model["prompt_pass1"], "conditions/c390/prompts/v_commit_first_v1.txt")
            self.assertEqual({k: v for k, v in model.items() if not k.startswith("prompt_pass1")},
                             {k: v for k, v in base_model.items() if not k.startswith("prompt_pass1")})
            for key in ("dataset", "generator_source", "delivery", "stages"):
                self.assertEqual(cond[key], base_cond[key])

    def test_testpool_dev_reviews_change_only_the_manifest_and_g_source(self) -> None:
        for dev_id, g in (("c210", "c205"), ("c211", "c200"), ("c212", "c202"), ("c213", "c206"), ("c214", "c204"), ("c215", "c201"), ("c216", "c207"), ("c217", "c201"), ("c218", "c205"), ("c219", "c203"), ("c220", "c202"), ("c221", "c203"), ("c222", "c204"), ("c223", "c202"), ("c224", "c208"), ("c225", "c203"), ("c226", "c208"), ("c227", "c207"), ("c228", "c208"), ("c229", "c204"), ("c248", "c200"), ("c249", "c201"), ("c250", "c201"), ("c251", "c206"), ("c262", "c209"), ("c263", "c200"), ("c264", "c209"), ("c265", "c206")):
            runner.configure(dev_id)
            try:
                runner.require_registered_differences()
                cond = runner.condition()
            finally:
                runner.configure("c024")
            self.assertTrue(runner.REVIEW_CONDITIONS[dev_id]["dev_pool"])
            self.assertNotIn("test", runner.REVIEW_CONDITIONS[dev_id])
            self.assertEqual(cond["dataset"]["manifest"], "testpool_dev1000")
            self.assertTrue(cond["dataset"]["development_only"])
            self.assertEqual(cond["generator_source"]["condition_id"], g)
            self.assertIn(f"__{g}__mathfull-testpool_dev1000__", cond["generator_source"]["artifact_path"])

    def test_olympiad_reviews_change_only_the_dataset_and_g_source(self) -> None:
        for ext_id, base_id, g in (("c256", "c214", "c254"), ("c257", "c215", "c255"), ("c260", "c248", "c259"), ("c261", "c249", "c255")):
            runner.configure(base_id)
            try:
                base_model, base = runner.require_registered_differences(), runner.condition()
                runner.configure(ext_id)
                ext_model, ext = runner.require_registered_differences(), runner.condition()
                self.assertEqual(runner.solo.DATASET_ID, runner.solo.OLYMPIAD_ID)
            finally:
                runner.configure("c024")
            self.assertTrue(runner.REVIEW_CONDITIONS[ext_id]["extension"])
            self.assertEqual(ext_model, base_model)
            self.assertEqual(ext["delivery"], base["delivery"])
            self.assertEqual(ext["dataset"]["manifest"], "eval")
            self.assertFalse(ext["dataset"]["development_only"])
            self.assertEqual(ext["generator_source"]["condition_id"], g)
            self.assertIn(f"__{g}__olympiadbench-eval__", ext["generator_source"]["artifact_path"])

    def test_c258_adds_only_the_special_token_repair_to_c244(self) -> None:
        runner.configure("c244")
        try:
            base_model, base = runner.require_registered_differences(), runner.condition()
            runner.configure("c258")
            model, cond = runner.require_registered_differences(), runner.condition()
        finally:
            runner.configure("c024")
        capture = dict(cond["model"]["pass1_logit_capture"])
        self.assertEqual(capture.pop(runner.REPAIR_KEY), runner.REPAIR_RULE)
        self.assertEqual(capture, base["model"]["pass1_logit_capture"])
        for key in ("dataset", "generator_source", "delivery"):
            self.assertEqual(cond[key], base[key])
        self.assertNotIn(runner.REPAIR_KEY, base["model"]["pass1_logit_capture"])

    @staticmethod
    def _trace(pieces: list[tuple[int, bytes]], content: str, status: str) -> dict:
        tokens, at = [], 0
        for i, (tid, b) in enumerate(pieces):
            tokens.append({"index": i, "token_id": tid, "token": None, "bytes": list(b),
                           "start_byte": at, "end_byte": at + len(b), "logprob": -0.1})
            at += len(b)
        return {"status": status, "logprobs_mode": "raw_logprobs", "tokens": tokens,
                "reconstructed_text_sha256": None, "content_start_byte": None, "content_end_byte": None}

    def test_special_token_repair_aligns_only_by_emptying_special_bytes(self) -> None:
        special = {900, 901}
        pieces = [(1, b"x = "), (900, b"<|start_header_id|>"), (2, b"as"), (3, b"sistant 1"), (901, b"<|eot_id|>")]
        content = "x = assistant 1"
        fixed, repaired = runner.repair_special_token_bytes(self._trace(pieces, content, "text_mismatch"), content, special)
        self.assertEqual(fixed["status"], "ok")
        self.assertEqual([r["token_id"] for r in repaired], [900, 901])
        self.assertEqual([t["token_id"] for t in fixed["tokens"]], [1, 900, 2, 3, 901])
        self.assertEqual(fixed["tokens"][1]["bytes"], [])
        self.assertEqual((fixed["content_start_byte"], fixed["content_end_byte"]), (0, len(content.encode())))

    def test_special_token_repair_leaves_aligned_or_other_mismatches_alone(self) -> None:
        special = {900}
        ok = self._trace([(1, b"x"), (900, b"<|eot_id|>")], "x", "ok")
        self.assertEqual(runner.repair_special_token_bytes(ok, "x", special), (ok, []))
        no_special = self._trace([(1, b"x"), (2, b"y")], "z", "text_mismatch")
        self.assertEqual(runner.repair_special_token_bytes(no_special, "z", special), (no_special, []))
        still, repaired = runner.repair_special_token_bytes(
            self._trace([(1, b"x"), (900, b"<|s|>"), (2, b"y")], "q", "text_mismatch"), "q", special)
        self.assertEqual(still["status"], "text_mismatch")
        self.assertEqual(len(repaired), 1)

    def test_prompt_carries_problem_and_g_output_and_asks_for_one_boxed_answer(self) -> None:
        model = runner.condition()["model"]
        msgs = runner.messages(model, "PROBLEM-TEXT", "G-SOLUTION-TEXT")
        self.assertIn("reviewer", msgs[0]["content"])
        self.assertIn("\\boxed{...}", msgs[0]["content"])
        self.assertIn("PROBLEM-TEXT", msgs[1]["content"])
        self.assertIn("G-SOLUTION-TEXT", msgs[1]["content"])


class GradeDeliveryTests(unittest.TestCase):
    def test_gv_falls_back_to_g_only_when_the_reviewer_gives_no_answer(self) -> None:
        def out(text, answer):
            return {"raw_output": text, "extracted_answer": answer}
        reviews = [{"problem_id": "a", "record_id": "ra", "outputs": [out("... \\boxed{5}", "5")]},
                   {"problem_id": "b", "record_id": "rb", "outputs": [out("no answer here", None)]}]
        g = [{"problem_id": "a", "outputs": [out("G says 4", "4")]},
             {"problem_id": "b", "outputs": [out("G says \\boxed{7}", "7")]}]
        saved = (runner.base.load_source_rows, runner.base.grade_prediction)
        runner.base.load_source_rows = lambda ids: {p: {"gold": "7", "problem": "x"} for p in ids}
        runner.base.grade_prediction = lambda gold, text: (0, "model_invalid") if text is None \
            else (int("7" in text), "ok")
        try:
            with tempfile.TemporaryDirectory() as t:
                tmp = Path(t)
                for name, rows in (("review", reviews), ("g", g)):
                    (tmp / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                runner.stage_grade(argparse.Namespace(review_records=str(tmp / "review.jsonl"),
                                                      g_records=str(tmp / "g.jsonl"),
                                                      output=str(tmp / "grades.jsonl")))
                rows = [json.loads(l) for l in (tmp / "grades.jsonl").read_text().splitlines()]
        finally:
            runner.base.load_source_rows, runner.base.grade_prediction = saved
        gv = {r["problem_id"]: r for r in rows if r["outcome"] == "u_gv"}
        self.assertEqual(gv["a"]["delivered_from"], "review")
        self.assertEqual(gv["b"]["delivered_from"], "g_fallback")
        self.assertEqual(gv["b"]["u"], 1)


if __name__ == "__main__":
    unittest.main()
