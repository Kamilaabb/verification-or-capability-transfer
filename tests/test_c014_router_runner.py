"""CPU selftest for the c014 runner: the stages that need no model and no network.

pass1, native raw-logit capture and pass2 all call a served model, so they are exercised by the GPU
preflight instead. Everything downstream of them is pure, and this file runs it on
synthetic records, validating each emitted row against the schema the condition actually
registers rather than against a hand-written copy of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_c014_router as runner  # noqa: E402
from validate_records import SchemaStore, validate  # noqa: E402


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_schema(case, rows, schema_name):
    schema_path = ROOT / "protocols/router_v1/schemas" / schema_name
    store = SchemaStore()
    schema = store.load(schema_path)
    for row in rows:
        errors: list[str] = []
        validate(row, schema, store, schema_path, schema, row.get("problem_id", "?"), errors)
        case.assertEqual(errors, [], f"{schema_name}: {errors}")


def pass1_row(pid, answer, tokens=120):
    return {"run_id": "run1", "problem_id": pid, "problem_text": f"problem {pid}",
            "outputs": [{"sample_index": 0, "extracted_answer": answer,
                         "usage": {"output_tokens": tokens}}]}


def pass2_row(pid, answer, status="ok", verdict="incorrect", tokens=60):
    return {"run_id": "run1", "problem_id": pid, "extracted_answer": answer,
            "verdict": verdict, "parse_status": status, "answer_changed": True,
            "usage": {"output_tokens": tokens}}


def grade_rows(spec):
    rows = []
    for pid, (u0, u_self) in spec.items():
        rows.append({"problem_id": pid, "outcome": "u0", "value": u0, "grader_status": "ok"})
        if u_self is not None:
            rows.append({"problem_id": pid, "outcome": "u_self", "value": u_self,
                         "grader_status": "ok"})
    return rows


class CounterfactualStageTests(unittest.TestCase):
    def build(self, spec, statuses=None):
        statuses = statuses or {}
        self.dir = Path(tempfile.mkdtemp())
        write(self.dir / "p1.jsonl", [pass1_row(p, "1") for p in spec])
        write(self.dir / "p2.jsonl",
              [pass2_row(p, "2", status=statuses.get(p, "ok")) for p in spec])
        write(self.dir / "g.jsonl", grade_rows(spec))
        with patch.object(runner, "condition_hash", return_value="a" * 64):
            runner.stage_counterfactual(argparse.Namespace(
                run_id="run1", replicate_id="p01", base_seed=1,
                pass1_records=str(self.dir / "p1.jsonl"),
                pass2_records=str(self.dir / "p2.jsonl"),
                grades=str(self.dir / "g.jsonl"),
                output=str(self.dir / "cf.jsonl")))
        return read(self.dir / "cf.jsonl")

    def test_emits_rows_valid_against_the_registered_schema(self):
        rows = self.build({"a": (0, 1), "b": (1, 0), "c": (1, 1), "d": (0, 0)})
        self.assertEqual(len(rows), 4)
        check_schema(self, rows, "router_counterfactual_v1.json")

    def test_both_arms_and_both_token_costs_are_carried_per_row(self):
        rows = {r["problem_id"]: r for r in self.build({"a": (0, 1), "b": (1, 0)})}
        self.assertEqual((rows["a"]["a0"], rows["a"]["a1"]), (0, 1))   # repair
        self.assertEqual((rows["b"]["a0"], rows["b"]["a1"]), (1, 0))   # harm
        for row in rows.values():
            self.assertEqual(row["tokens_pass1"], 120)
            self.assertEqual(row["tokens_pass2"], 60)

    def test_a_failed_self_check_falls_back_to_y0_rather_than_scoring_zero(self):
        rows = {r["problem_id"]: r for r in
                self.build({"a": (1, None)}, statuses={"a": "call_error"})}
        self.assertIsNone(rows["a"]["u_self"])
        self.assertEqual(rows["a"]["a1"], 1)
        check_schema(self, list(rows.values()), "router_counterfactual_v1.json")


class AnalyzeStageTests(unittest.TestCase):
    def analyze(self, spec, scores=None, varying_cost=False):
        directory = Path(tempfile.mkdtemp())
        rows = []
        for i, (pid, (u0, u_self)) in enumerate(spec.items()):
            arms = runner.deployed_outcomes(u0=u0, u_self=u_self, self_check_status="ok")
            rows.append({"problem_id": pid, "u0": u0, "u_self": u_self, **arms,
                         "tokens_pass1": 100,
                         "tokens_pass2": (10 * (i + 1)) if varying_cost else 50})
        write(directory / "cf.jsonl", rows)
        score_path = None
        if scores:
            score_path = directory / "s.jsonl"
            write(score_path, [{"problem_id": p, "score": s} for p, s in scores.items()])
        with patch.object(runner, "condition_hash", return_value="a" * 64):
            runner.stage_analyze(argparse.Namespace(
                run_id="run1", counterfactuals=str(directory / "cf.jsonl"),
                logit_traces=None, router_scores=str(score_path) if score_path else None,
                bootstrap_seed=1, bootstrap_repetitions=200,
                output=str(directory / "report.json")))
        return json.loads((directory / "report.json").read_text(encoding="utf-8"))

    def balanced(self, n=20):
        spec = {}
        for i in range(n):
            spec[f"r{i}"] = (0, 1)
            spec[f"h{i}"] = (1, 0)
        return spec

    def test_gate0_reports_min_r_h_not_h_as_the_headroom(self):
        # 3 repairs, 8 harms: the best fixed arm is A0 and the headroom is 3, not 8.
        spec = {f"r{i}": (0, 1) for i in range(3)}
        spec.update({f"h{i}": (1, 0) for i in range(8)})
        report = self.analyze(spec)
        self.assertEqual(report["pools"]["headroom_over_best_fixed"], 3)
        self.assertEqual(report["best_fixed_arm"]["name"], "A0_never_self_check")

    def test_a_tied_average_effect_still_passes_the_only_stop_criterion(self):
        report = self.analyze(self.balanced(20))
        self.assertEqual(report["a1_minus_a0_descriptive"]["net"], 0)
        self.assertEqual(report["best_fixed_arm"]["name"], "tie")
        self.assertTrue(report["gate_0"]["G1_decision_relevant_at_least_20"]["passed"])
        self.assertTrue(report["gate_0"]["superiority_testing_permitted"]["permitted"])

    def test_small_pools_stop_and_forbid_superiority_testing(self):
        report = self.analyze({"r0": (0, 1), "h0": (1, 0), "c0": (1, 1)})
        self.assertFalse(report["gate_0"]["G1_decision_relevant_at_least_20"]["passed"])
        self.assertFalse(report["gate_0"]["superiority_testing_permitted"]["permitted"])

    def test_bootstrap_covers_accuracy_and_tokens(self):
        report = self.analyze(self.balanced(20), varying_cost=True)
        boot = report["bootstrap_a1_vs_a0"]
        self.assertLessEqual(boot["accuracy_ci"][0], boot["accuracy_ci"][1])
        self.assertGreater(boot["token_difference"], 0)
        # Per-row cost varies, so the saving a fresh draw of problems would show varies
        # too. This is exactly the sampling uncertainty a token total is NOT free of.
        self.assertLess(boot["token_ci"][0], boot["token_ci"][1])

    def test_constant_per_row_cost_gives_a_degenerate_token_interval(self):
        # With an identical cost on every row, always-call spends n * cost under any
        # resample, so the interval is a point. That is correct, and it is the only
        # regime where "a token count has no sampling error" holds literally.
        report = self.analyze(self.balanced(20), varying_cost=False)
        boot = report["bootstrap_a1_vs_a0"]
        self.assertEqual(boot["token_ci"][0], boot["token_ci"][1])
        self.assertEqual(boot["token_ci"][0], boot["token_difference"])

    def test_frontier_is_reported_entire_with_no_champion_threshold(self):
        spec = self.balanced(10)
        scores = {p: (1.0 if p.startswith("r") else 0.0) for p in spec}
        report = self.analyze(spec, scores=scores)
        frontier = report["oof_frontier"]
        self.assertEqual(len(frontier), len(spec) + 1)          # every k, k=0..n
        self.assertEqual(frontier[0]["call_fraction"], 0.0)
        self.assertEqual(frontier[-1]["call_fraction"], 1.0)
        self.assertIn("oof_frontier_is_exploratory", report)
        # No single operating point is promoted anywhere in the report.
        self.assertNotIn("best_threshold", json.dumps(report))
        self.assertNotIn("selected_operating_point", json.dumps(report))

    def test_frontier_tokens_rise_monotonically_with_the_call_fraction(self):
        spec = self.balanced(10)
        scores = {p: (1.0 if p.startswith("r") else 0.0) for p in spec}
        frontier = self.analyze(spec, scores=scores)["oof_frontier"]
        tokens = [point["output_tokens"] for point in frontier]
        self.assertEqual(tokens, sorted(tokens))


class RunnableGateTests(unittest.TestCase):
    def test_pass2_frozen_prompt_is_one_user_message_without_rewriting(self):
        messages = runner.pass2_messages("What is 1+1?", "The answer is 2.")
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("What is 1+1?", messages[0]["content"])
        self.assertIn("The answer is 2.", messages[0]["content"])
        self.assertNotIn("{{problem}}", messages[0]["content"])

    def test_same_forward_capture_keeps_all_registered_per_token_statistics(self):
        source_trace = {
            "status": "ok",
            "tokens": [{"index": 0, "token_id": 17, "logprob": -0.25}],
        }
        pass1 = {
            "run_id": "pass1-run", "problem_id": "p", "problem_text": "2+2",
            "outputs": [{"raw_output": "4", "token_logprobs": source_trace}],
        }
        top_ids = [17] + list(range(100, 119))
        top_logits = [8.0 - index for index in range(20)]
        capture = {
            "capture_id": "capture",
            "output_token_ids": [17],
            "positions": [{
                "index": 0, "sampled_token_id": 17, "sampled_raw_logit": 8.0,
                "max_raw_logit": 8.0, "logsumexp": 8.25, "entropy": 0.5,
                "top1_top2_margin": 1.0, "top20_token_ids": top_ids,
                "top20_raw_logits": top_logits,
            }],
        }
        with patch.object(runner, "condition_hash", return_value="a" * 64):
            row = runner.build_same_forward_logit_record(
                run_id="raw-run", pass1_record=pass1, capture=capture)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["capture_id"], "capture")
        position = row["positions"][0]
        self.assertEqual(position["sampled_logprob"], -0.25)
        self.assertEqual(position["sampled_raw_logit"], 8.0)
        self.assertEqual(position["logsumexp"], 8.25)
        self.assertEqual(len(position["top20_token_ids"]), 20)
        self.assertNotIn("full_vocab_logits", json.dumps(row))
        check_schema(self, [row], "native_logit_trace_record_v1.json")

    def test_same_forward_capture_rejects_a_logprob_identity_mismatch(self):
        pass1 = {"run_id": "p1", "problem_id": "p",
                 "outputs": [{"token_logprobs": {"status": "ok", "tokens": [
                     {"index": 0, "token_id": 1, "logprob": -0.2}]}}]}
        capture = {"output_token_ids": [1], "positions": [{
            "index": 0, "sampled_token_id": 1, "sampled_raw_logit": 4.0,
            "max_raw_logit": 4.0, "logsumexp": 5.0, "entropy": 1.0,
            "top1_top2_margin": 1.0, "top20_token_ids": list(range(1, 21)),
            "top20_raw_logits": [4.0 - i for i in range(20)]}]}
        with patch.object(runner, "condition_hash", return_value="a" * 64):
            row = runner.build_same_forward_logit_record(
                run_id="raw", pass1_record=pass1, capture=capture)
        self.assertEqual(row["status"], "identity_mismatch")

    def test_condition_hash_covers_router_record_schema_paths(self):
        digest = runner.condition_hash()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, runner.condition_hash())

    def test_condition_hash_ignores_long_behavioral_prose_on_linux(self):
        frozen = runner.condition()
        frozen["gate"]["scope_limit"] = "x" * 4096
        with patch.object(runner, "condition", return_value=frozen):
            self.assertRegex(runner.condition_hash(), r"^[0-9a-f]{64}$")

    def test_pass1_identity_comes_from_the_shared_c014_model(self):
        identity = runner.model_identity(ROOT / "conditions/c002/prompts/g_initial_v1.txt")
        self.assertEqual(identity["model_id"], "Qwen/Qwen3-4B")
        self.assertEqual(identity["revision"], runner.model_block()["revision"])
        self.assertEqual(identity["chat_template_sha256"],
                         runner.model_block()["chat_template_hash"])
        self.assertRegex(identity["prompt_sha256"], r"^[0-9a-f]{64}$")

    def test_an_unpinned_model_blocks_before_any_gpu_work(self):
        unfrozen = {"revision": None, "server_path": None, "chat_template_hash": None}
        with patch.object(runner, "model_block", return_value=unfrozen), \
                self.assertRaises(SystemExit) as caught:
            runner.require_frozen_model()
        self.assertIn("revision", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
