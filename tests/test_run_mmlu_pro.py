"""End-to-end check of scripts/run_mmlu_pro.py (c070 then c072) on a simulated server.

A fake api_call answers from canned text, builds the token trace the real client would and
writes the same-forward capture the logits processor would. The runner's own generate, review,
evidence, grade and analyze stages then run unchanged, and every written record is validated
against the schemas the condition registers. No data, model or GPU is needed.
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

import run_mmlu_pro as runner  # noqa: E402

from authority_allocation.api import usage_from_response  # noqa: E402
from authority_allocation.generated_token_logprobs import build_generated_token_trace  # noqa: E402

ROWS = {
    "70": {"question": "Which organ pumps blood?", "options": ["heart", "lung", "liver"],
           "answer": "A", "category": "health", "origin": "ori_mmlu"},
    "71": {"question": "What is 2 + 2?", "options": ["3", "4", "5", "6"],
           "answer": "B", "category": "math", "origin": "theoremQA"},
    "72": {"question": "Who wrote Hamlet?", "options": ["Marlowe", "Shakespeare"],
           "answer": "B", "category": "history", "origin": "stemez"},
}
SOLVE_TEXT = {"70": "The heart pumps blood.\nAnswer: A", "71": "It is 5.\nAnswer: C", "72": "Hmm hmm hmm"}
REVIEW_TEXT = {"70": "Correct.\nAnswer: A",
               "71": "The proposed Answer: C is wrong; 2 + 2 = 4.\nAnswer: B",
               "72": "Not sure."}


def manifest_rows() -> list[dict]:
    return [{"problem_id": pid, "category": r["category"], "origin": r["origin"],
             "n_options": len(r["options"])} for pid, r in ROWS.items()]


class FakeServer:
    def __init__(self, sidecar: Path, texts: dict[str, str]) -> None:
        self.sidecar, self.texts, self.calls = sidecar, texts, []

    def __call__(self, *, messages, decoding, thinking, **_: object):
        self.calls.append({"messages": messages, "thinking": thinking, "decoding": decoding})
        user = messages[1]["content"]
        pid = next((p for p, r in ROWS.items() if r["question"] in user), None)
        content = self.texts.get(pid, "Boiling point.\nAnswer: A")
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
    def run_stage(self, condition_id: str, work: Path, texts: dict[str, str], g_dir: Path | None = None):
        runner.configure(condition_id)
        records = "review.jsonl" if runner.ROLE == "review" else "pass1.jsonl"
        (work / "records").mkdir(parents=True)
        manifest = work / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows()), encoding="utf-8")
        server = FakeServer(work / "sidecar.jsonl", texts)
        common = dict(run_id=f"{condition_id}__run", api_base="http://fake", served_model="m",
                      manifest=str(manifest), output=str(work / "records" / records),
                      capture_sidecar=str(server.sidecar),
                      logit_output=str(work / "records" / "native_logit_trace.jsonl"),
                      logit_run_id=f"{condition_id}__logits", expected_rows=3,
                      replicate_id="p01", base_seed=runner.BASE_SEED)
        real_condition = runner.condition

        def condition_with_source(cid=None):
            # A review cannot name its source archive before the source has run.
            resolved = real_condition(cid)
            if "generator_source" in resolved:
                resolved["generator_source"]["pass1_run_id"] = "c070__run"
            return resolved

        with patch.object(runner, "api_call", server), \
             patch.object(runner, "load_mmlu_pro_rows", fake_rows), \
             patch.object(runner, "condition", condition_with_source), \
             patch.object(runner, "require_registered", lambda cid: real_condition(cid)["model"]):
            if runner.ROLE == "review":
                for name in ("pass1.jsonl", "grades.jsonl"):
                    (work / "records" / f"g_{name}").write_text(
                        (g_dir / "records" / name).read_text(encoding="utf-8"), encoding="utf-8")
                runner.stage_review(argparse.Namespace(**common, g_records=str(work / "records/g_pass1.jsonl")))
                g_args = {"g_records": str(work / "records/g_pass1.jsonl"),
                          "g_grades": str(work / "records/g_grades.jsonl")}
            else:
                runner.stage_generate(argparse.Namespace(**common))
                g_args = {"g_records": None, "g_grades": None}
            runner.stage_evidence(argparse.Namespace(run_id="ev", records=common["output"],
                                                     logit_traces=common["logit_output"],
                                                     output=str(work / "records/scalar_evidence.jsonl")))
            runner.stage_grade(argparse.Namespace(records=common["output"], g_records=g_args["g_records"],
                                                  output=str(work / "records/grades.jsonl")))
            runner.stage_analyze(argparse.Namespace(
                run_id="an", records=common["output"], evidence=str(work / "records/scalar_evidence.jsonl"),
                grades=str(work / "records/grades.jsonl"), expected_rows=3,
                output=str(work / "reports/analysis.json"), **g_args))
        (work / "condition.resolved.yaml").write_text(
            yaml.safe_dump(real_condition(condition_id), sort_keys=False), encoding="utf-8")
        return server

    def validate_bundle(self, work: Path) -> None:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_records.py"),
                                 "--bundle-dir", str(work)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_solo_then_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            g_dir, v_dir = Path(tmp) / "g", Path(tmp) / "gv"
            server = self.run_stage("c070", g_dir, SOLVE_TEXT)
            self.assertTrue(all(call["thinking"] is False for call in server.calls))
            self.assertTrue(all(call["decoding"]["repetition_penalty"] == 1.0 for call in server.calls))
            self.assertIn("Options:\nA. heart\nB. lung\nC. liver", server.calls[0]["messages"][1]["content"])
            report = json.loads((g_dir / "reports/analysis.json").read_text())
            self.assertEqual((report["correct"], report["answer_parseable"]), (1, 2))
            evidence = {json.loads(line)["problem_id"]: json.loads(line)
                        for line in (g_dir / "records/scalar_evidence.jsonl").read_text().splitlines()}
            self.assertEqual(len(evidence["70"]["features"]), 22)
            self.assertIsNotNone(evidence["70"]["features"]["ans_lp"])
            self.validate_bundle(g_dir)

            self.run_stage("c072", v_dir, REVIEW_TEXT, g_dir=g_dir)
            review = {json.loads(line)["problem_id"]: json.loads(line)
                      for line in (v_dir / "records/review.jsonl").read_text(encoding="utf-8").splitlines()}
            # The reviewer quoted "the proposed Answer: C" mid-sentence; only its own line counts.
            self.assertEqual(review["71"]["outputs"][0]["extracted_answer"], "B")
            report = json.loads((v_dir / "reports/analysis.json").read_text())
            self.assertEqual((report["gv_primary_correct"], report["gv_strict_correct"]), (2, 2))
            self.assertEqual(report["fallback_rows"], 1)
            self.assertEqual((report["gv_vs_reviewed_repair"], report["gv_vs_reviewed_harm"]), (1, 0))
            self.validate_bundle(v_dir)

    def test_check_runnable(self) -> None:
        # Both review sources (c070 for c072, c071 for c073) are archived and filled in.
        for cid in ("c070", "c071", "c072", "c073", "c074", "c075", "c076", "c077", "c078", "c079", "c080", "c081", "c082", "c083", "c084", "c085", "c086", "c087", "c088", "c089", "c120", "c121", "c122", "c123", "c124", "c125", "c126", "c127", "c128", "c129", "c130", "c131", "c132", "c133", "c134", "c135", "c136", "c137", "c138", "c139", "c140", "c141", "c142", "c143", "c144", "c145", "c146", "c147", "c170", "c171", "c172", "c173", "c174", "c175", "c176", "c177", "c178", "c179", "c180", "c181", "c182", "c183", "c184", "c185", "c186", "c187", "c188", "c189", "c190", "c191", "c270", "c271", "c272", "c273", "c274", "c275", "c276", "c277", "c392", "c393"):
            runner.configure(cid)
            self.assertEqual(runner.require_registered(cid)["pass1_decoding"]["max_new_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
