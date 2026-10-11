"""End-to-end check of scripts/run_bigcodebench_review.py: c100 solves on a simulated server, then
c110 reviews those records; grading uses a simulated container result file, including a reviewer
output with no program so the primary delivery falls back to the reviewed program. Every written
record is validated against the schemas c110 registers. No data, model, container or GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_bigcodebench_review as review  # noqa: E402
import run_bigcodebench_solo as solo  # noqa: E402
from test_run_bigcodebench_solo import ROWS, FakeServer, fake_rows  # noqa: E402

from authority_allocation import bigcodebench_study  # noqa: E402
from authority_allocation.prompts import PROMPT_VARIABLE, split_prompt  # noqa: E402

# Reviewed (c100) grades: 1 pass, 2 fail, 3 no program. Reviewer results: 1 keeps the pass,
# 2 repairs, 3 still has no program (fallback to the reviewed program, which also fails).
G_RESULTS = {"BigCodeBench/1": (1, "ok", "pass"), "BigCodeBench/2": (0, "ok", "fail"),
             "BigCodeBench/3": (0, "model_invalid", "no_program")}
V_RESULTS = {"BigCodeBench/1": (1, "ok", "pass"), "BigCodeBench/2": (1, "ok", "pass"),
             "BigCodeBench/3": (0, "model_invalid", "no_program")}


def write_results(requests: Path, results: Path, table: dict) -> None:
    rows = [json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines()]
    results.write_text("".join(json.dumps({"record_id": r["record_id"], "task_id": r["task_id"],
                                           "u": table[r["task_id"]][0], "status": table[r["task_id"]][1],
                                           "detail": table[r["task_id"]][2], "seconds": 0.1}) + "\n"
                               for r in rows), encoding="utf-8")


class ReviewPromptTests(unittest.TestCase):
    def test_review_prompt_renders_to_the_registered_messages(self) -> None:
        system, user = split_prompt(ROOT / bigcodebench_study.CODE_REVIEW_PROMPT)
        self.assertEqual(system, (
            "Review the proposed solution to the Python programming task for correctness.\n"
            "The proposed solution may be correct or incorrect. Carefully check its logic, edge cases, use "
            "of library APIs, and whether it meets every requirement of the task.\n"
            "If the proposed solution is correct, retain it. If it contains any error, correct the error and "
            "produce a correct solution.\n"
            "Return a complete final solution to the original task. End with exactly one complete, "
            "self-contained Python code block (```python ... ```) that includes all required imports and the "
            "requested function definition."))
        self.assertEqual(set(PROMPT_VARIABLE.findall(user)), {"problem", "y0"})
        self.assertEqual(bigcodebench_study.user_message(user, problem="T", y0="S {{x}}"),
                         "Task:\nT\n\nProposed solution:\nS {{x}}")

    def test_review_prompt_carries_no_experiment_framing(self) -> None:
        text = (ROOT / bigcodebench_study.CODE_REVIEW_PROMPT).read_text(encoding="utf-8").lower()
        for word in ("generator", "verifier", "experiment", "authority", "another model"):
            self.assertNotIn(word, text)


class ReviewEndToEndTest(unittest.TestCase):
    def test_solo_then_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            g_dir, v_dir = Path(tmp) / "g", Path(tmp) / "v"
            for d in (g_dir, v_dir):
                (d / "records").mkdir(parents=True)
                (d / "grading").mkdir()
            manifest = Path(tmp) / "manifest.jsonl"
            manifest.write_text("".join(json.dumps({"problem_id": p, "libs": "[]"}) + "\n" for p in ROWS),
                                encoding="utf-8")
            solo.configure("c100")
            server = FakeServer(g_dir / "sidecar.jsonl")
            with patch.object(solo, "api_call", server), patch.object(solo, "load_bigcodebench_rows", fake_rows):
                solo.stage_generate(argparse.Namespace(
                    run_id="c100__run", api_base="x", served_model="m", manifest=str(manifest),
                    output=str(g_dir / "records/pass1.jsonl"), capture_sidecar=str(server.sidecar),
                    logit_output=str(g_dir / "records/native_logit_trace.jsonl"), logit_run_id="c100__logits",
                    expected_rows=3, replicate_id="p01", base_seed=solo.BASE_SEED))
            solo.stage_grade_requests(argparse.Namespace(records=str(g_dir / "records/pass1.jsonl"),
                                                         output=str(g_dir / "grading/requests.jsonl")))
            write_results(g_dir / "grading/requests.jsonl", g_dir / "grading/results.jsonl", G_RESULTS)
            solo.stage_grade(argparse.Namespace(records=str(g_dir / "records/pass1.jsonl"),
                                                results=str(g_dir / "grading/results.jsonl"),
                                                output=str(g_dir / "records/grades.jsonl")))
            for name in ("pass1.jsonl", "native_logit_trace.jsonl", "grades.jsonl"):
                (v_dir / "records" / f"g_{name}").write_text((g_dir / "records" / name).read_text(encoding="utf-8"),
                                                             encoding="utf-8")

            review.configure("c110")
            real_condition = solo.condition

            def with_source(cid=None):
                resolved = real_condition(cid)
                if "generator_source" in resolved:
                    resolved["generator_source"]["pass1_run_id"] = "c100__run"
                return resolved

            server = FakeServer(v_dir / "sidecar.jsonl")
            records = v_dir / "records/review.jsonl"
            with patch.object(solo, "api_call", server), patch.object(solo, "load_bigcodebench_rows", fake_rows), \
                 patch.object(solo, "condition", with_source):
                review.stage_review(argparse.Namespace(
                    run_id="c110__run", api_base="x", served_model="m", manifest=str(manifest),
                    output=str(records), g_records=str(v_dir / "records/g_pass1.jsonl"),
                    capture_sidecar=str(server.sidecar), logit_output=str(v_dir / "records/native_logit_trace.jsonl"),
                    logit_run_id="c110__logits", expected_rows=3, replicate_id="p01", base_seed=solo.BASE_SEED))
            first = server.calls[0]["messages"][1]["content"]
            self.assertTrue(first.startswith("Task:\n" + ROWS["BigCodeBench/1"]["problem_text"] + "\n\nProposed solution:\n"))
            review.stage_evidence(argparse.Namespace(run_id="ev", records=str(records),
                                                     logit_traces=str(v_dir / "records/native_logit_trace.jsonl"),
                                                     output=str(v_dir / "records/scalar_evidence.jsonl")))
            solo.stage_grade_requests(argparse.Namespace(records=str(records),
                                                         output=str(v_dir / "grading/requests.jsonl")))
            write_results(v_dir / "grading/requests.jsonl", v_dir / "grading/results.jsonl", V_RESULTS)
            review.stage_grade(argparse.Namespace(records=str(records), results=str(v_dir / "grading/results.jsonl"),
                                                  g_grades=str(v_dir / "records/g_grades.jsonl"),
                                                  output=str(v_dir / "records/grades.jsonl")))
            review.stage_analyze(argparse.Namespace(
                run_id="an", records=str(records), evidence=str(v_dir / "records/scalar_evidence.jsonl"),
                grades=str(v_dir / "records/grades.jsonl"), g_records=str(v_dir / "records/g_pass1.jsonl"),
                g_grades=str(v_dir / "records/g_grades.jsonl"), expected_rows=3,
                output=str(v_dir / "reports/analysis.json")))
            report = json.loads((v_dir / "reports/analysis.json").read_text(encoding="utf-8"))
            self.assertEqual((report["gv_primary_passed"], report["gv_strict_passed"], report["fallback_rows"]),
                             (2, 2, 1))
            self.assertEqual((report["gv_vs_reviewed_repair"], report["gv_vs_reviewed_harm"]), (1, 0))
            (v_dir / "condition.resolved.yaml").write_text(yaml.safe_dump(real_condition("c110"), sort_keys=False),
                                                           encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_records.py"),
                                     "--bundle-dir", str(v_dir)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_token_bytes_repair(self) -> None:
        text = "e.g. `�` ok"
        pieces = [b"e.g. `", b"", b"` ok", b"<|im_end|>"]
        tokens, pos = [], 0
        for i, b in enumerate(pieces):
            tokens.append({"index": i, "token_id": 100 + i, "token": b.decode(), "bytes": list(b),
                           "start_byte": pos, "end_byte": pos + len(b), "logprob": -0.1})
            pos += len(b)
        broken = {"status": "text_mismatch", "logprobs_mode": "raw_logprobs", "tokens": tokens,
                  "reconstructed_text_sha256": "0" * 64, "content_start_byte": None, "content_end_byte": None}
        vocab = {101: "�".encode("utf-8")}
        fixed, repaired = review.repair_empty_token_bytes(broken, text, lambda tid: vocab[tid])
        self.assertEqual(fixed["status"], "ok")
        self.assertEqual((fixed["content_start_byte"], fixed["content_end_byte"]), (0, len(text.encode("utf-8"))))
        self.assertEqual(repaired, [{"index": 1, "token_id": 101, "bytes_hex": "efbfbd"}])
        self.assertEqual(fixed["tokens"][2]["start_byte"], len(b"e.g. `\xef\xbf\xbd"))
        self.assertEqual(broken["tokens"][1]["bytes"], [])                    # the input is not mutated
        # Wrong vocabulary bytes: the strict rule still rejects the row.
        still, _ = review.repair_empty_token_bytes(broken, text, lambda tid: b"?")
        self.assertEqual(still["status"], "text_mismatch")
        # A trace that already aligned is returned as is.
        ok = {**broken, "status": "ok"}
        self.assertIs(review.repair_empty_token_bytes(ok, text, lambda tid: vocab[tid])[0], ok)

    def test_byte_level_decoder_round_trips_every_byte(self) -> None:
        decode = review.byte_level_decoder()
        self.assertEqual(sorted(decode.values()), list(range(256)))
        self.assertEqual(bytes(decode[ch] for ch in "ï¿½"), b"\xef\xbf\xbd")
        self.assertEqual(bytes(decode[ch] for ch in "ĠA"), b" A")

    def test_only_the_repair_conditions_register_the_repair(self) -> None:
        for cid in review.REVIEWS:
            review.configure(cid)
            capture = review.require_registered(cid)["pass1_logit_capture"]
            self.assertEqual(review.REPAIR_KEY in capture, cid in ("c114", "c301", "c302"))

    def test_check_runnable(self) -> None:
        for cid in review.REVIEWS:
            review.configure(cid)
            self.assertEqual(review.require_registered(cid)["prompt_pass1"], bigcodebench_study.CODE_REVIEW_PROMPT)


if __name__ == "__main__":
    unittest.main()
