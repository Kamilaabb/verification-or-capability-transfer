"""End-to-end check of scripts/run_bigcodebench_solo.py (c100) on a simulated server.

A fake api_call answers from canned text, builds the token trace the real client would and
writes the same-forward capture the logits processor would. The runner's own generate,
evidence, grade-requests, grade and analyze stages then run unchanged on a simulated container
result file, and every written record is validated against the schemas the condition registers.
No data, model, container or GPU is needed.
"""

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

import run_bigcodebench_solo as runner  # noqa: E402

from authority_allocation.api import usage_from_response  # noqa: E402
from authority_allocation.generated_token_logprobs import build_generated_token_trace  # noqa: E402

ROWS = {
    "BigCodeBench/1": {"problem_text": "Return the sum.\n```\ndef task_func(a, b):\n```", "entry_point": "task_func"},
    "BigCodeBench/2": {"problem_text": "Return the product.\n```\ndef task_func(a, b):\n```", "entry_point": "task_func"},
    "BigCodeBench/3": {"problem_text": "Return nothing useful.", "entry_point": "task_func"},
}
TEXT = {"BigCodeBench/1": "Sum it.\n```python\ndef task_func(a, b):\n    return a + b\n```",
        "BigCodeBench/2": "```python\ndef task_func(a, b):\n    return a - b\n```",
        "BigCodeBench/3": "I cannot do this."}
RESULTS = {"BigCodeBench/1": (1, "ok", "pass"), "BigCodeBench/2": (0, "ok", "fail"),
           "BigCodeBench/3": (0, "model_invalid", "no_program")}


class FakeServer:
    def __init__(self, sidecar: Path) -> None:
        self.sidecar, self.calls = sidecar, []

    def __call__(self, *, messages, decoding, thinking, **_: object):
        self.calls.append({"messages": messages, "thinking": thinking, "decoding": decoding})
        user = messages[1]["content"]
        pid = next((p for p, r in ROWS.items() if r["problem_text"] in user), None)
        content = TEXT.get(pid, "```python\ndef task_func(a, b):\n    return a + b\n```")
        pieces = [content[i:i + 3] for i in range(0, len(content), 3)]
        lps = [-0.1 * (k % 4) for k in range(len(pieces))]
        ids = [1000 + k for k in range(len(pieces))]
        response = {"choices": [{"message": {"content": content}, "finish_reason": "stop", "token_ids": ids,
                                 "logprobs": {"content": [{"token": t, "logprob": lp, "token_id": i}
                                                          for t, lp, i in zip(pieces, lps, ids)]}}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": len(pieces)}}
        usage = usage_from_response(response, 9.0, retries=0)
        usage["token_logprobs"] = build_generated_token_trace(response, expected_text=content,
                                                              logprobs_mode="raw_logprobs")
        positions = [{"index": k, "sampled_token_id": tid, "sampled_raw_logit": 10.0, "max_raw_logit": 10.0,
                      "logsumexp": 10.0 - lp, "entropy": 0.5, "top1_top2_margin": 1.0,
                      "top20_token_ids": [tid] + [5000 + j for j in range(19)],
                      "top20_raw_logits": [10.0] + [9.0 - 0.1 * j for j in range(19)]}
                     for k, (lp, tid) in enumerate(zip(lps, ids))]
        with self.sidecar.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"capture_id": decoding["vllm_xargs"]["c014_capture_id"],
                                     "positions": positions, "output_token_ids": ids}) + "\n")
        return content, None, usage


def fake_rows(ids):
    return {pid: ROWS[pid] for pid in ids}


class RunnerEndToEndTest(unittest.TestCase):
    def test_generate_evidence_grade_analyze(self) -> None:
        runner.configure("c100")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "records").mkdir()
            manifest = work / "manifest.jsonl"
            manifest.write_text("".join(json.dumps({"problem_id": p, "libs": "['math']"}) + "\n" for p in ROWS),
                                encoding="utf-8")
            server = FakeServer(work / "sidecar.jsonl")
            records, traces = work / "records/pass1.jsonl", work / "records/native_logit_trace.jsonl"
            with patch.object(runner, "api_call", server), \
                 patch.object(runner, "load_bigcodebench_rows", fake_rows):
                runner.stage_generate(argparse.Namespace(
                    run_id="c100__run", api_base="http://fake", served_model="m", manifest=str(manifest),
                    output=str(records), capture_sidecar=str(server.sidecar), logit_output=str(traces),
                    logit_run_id="c100__logits", expected_rows=3, replicate_id="p01",
                    base_seed=runner.BASE_SEED))
            self.assertTrue(all(call["thinking"] is False for call in server.calls))
            self.assertTrue(all(call["decoding"]["repetition_penalty"] == 1.0 for call in server.calls))
            self.assertEqual(server.calls[0]["messages"][1]["content"], "Task:\n" + ROWS["BigCodeBench/1"]["problem_text"])
            runner.stage_evidence(argparse.Namespace(run_id="ev", records=str(records), logit_traces=str(traces),
                                                     output=str(work / "records/scalar_evidence.jsonl")))
            requests = work / "grading/requests.jsonl"
            requests.parent.mkdir()
            runner.stage_grade_requests(argparse.Namespace(records=str(records), output=str(requests)))
            rows = [json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["text"] for r in rows], [TEXT[p] for p in ROWS])
            results = work / "grading/results.jsonl"
            results.write_text("".join(json.dumps({"record_id": r["record_id"], "task_id": r["task_id"],
                                                   "u": RESULTS[r["task_id"]][0], "status": RESULTS[r["task_id"]][1],
                                                   "detail": RESULTS[r["task_id"]][2], "seconds": 0.1}) + "\n"
                                       for r in rows), encoding="utf-8")
            runner.stage_grade(argparse.Namespace(records=str(records), results=str(results),
                                                  output=str(work / "records/grades.jsonl")))
            runner.stage_analyze(argparse.Namespace(
                run_id="an", records=str(records), evidence=str(work / "records/scalar_evidence.jsonl"),
                grades=str(work / "records/grades.jsonl"), expected_rows=3,
                output=str(work / "reports/analysis.json")))
            report = json.loads((work / "reports/analysis.json").read_text(encoding="utf-8"))
            self.assertEqual((report["passed"], report["model_invalid_rows"]), (1, 1))
            self.assertEqual(report["python_block_count"], {"0": 1, "1": 2, "2_or_more": 0})
            (work / "condition.resolved.yaml").write_text(
                yaml.safe_dump(runner.condition("c100"), sort_keys=False), encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_records.py"),
                                     "--bundle-dir", str(work)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_grader_error_rejects_the_run(self) -> None:
        runner.configure("c100")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            records = work / "pass1.jsonl"
            records.write_text(json.dumps({"record_id": "r1", "problem_id": "BigCodeBench/1"}) + "\n",
                               encoding="utf-8")
            results = work / "results.jsonl"
            results.write_text(json.dumps({"record_id": "r1", "task_id": "BigCodeBench/1", "u": None,
                                           "status": "grader_error", "detail": "x", "seconds": 0}) + "\n",
                               encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runner.stage_grade(argparse.Namespace(records=str(records), results=str(results),
                                                      output=str(work / "grades.jsonl")))

    def test_check_runnable(self) -> None:
        for cid in runner.MODELS:
            runner.configure(cid)
            self.assertEqual(runner.require_registered(cid)["pass1_decoding"]["max_new_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
