"""End-to-end check of scripts/run_mmmu_pro.py (c036 then c038) on a simulated server.

A fake api_call answers from canned text, builds the token trace the real client would, reports
the <|image_pad|> count the registered pixel limits imply for the images it was actually sent,
and writes the same-forward capture the logits processor would. The runner's own generate or
review, evidence, grade and analyze stages then run unchanged, and the records they write are
validated against the schemas the condition registers. No data, model or GPU is needed.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mmmu_pro as runner  # noqa: E402

from authority_allocation import mmmu_pro  # noqa: E402
from authority_allocation.api import usage_from_response  # noqa: E402
from authority_allocation.generated_token_logprobs import build_generated_token_trace  # noqa: E402
from authority_allocation.serialization import sha256_bytes  # noqa: E402

PAD = 151655


def make_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"".join(b"\x00" + b"\x80" * (3 * width) for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def png_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack(">II", data[16:24])
    return height, width


ROWS = {
    "test_Art_1": {"question": "Which style is <image 1>?", "options": ["baroque", "cubist", "gothic"],
                   "answer": "A", "subject": "Art", "images": {1: make_png(320, 200)}},
    "validation_Math_2": {"question": "Compare <image 1> and <image 2>.", "options": ["x", "y", "z", "w"],
                          "answer": "C", "subject": "Math",
                          "images": {1: make_png(64, 64), 2: make_png(1000, 700)}},
    "test_Bio_3": {"question": "What is in <image 1>?", "options": ["cell", "leaf"],
                   "answer": "B", "subject": "Bio", "images": {1: make_png(500, 40)}},
}
SOLVE_TEXT = {"test_Art_1": "It has dramatic light.\nAnswer: A",
              "validation_Math_2": "Comparing both.\nAnswer: B",
              "test_Bio_3": "leaf leaf leaf leaf"}
REVIEW_TEXT = {"test_Art_1": "The solution is right.\nAnswer: A",
               "validation_Math_2": "The proposed Answer: B is wrong.\nAnswer: C",
               "test_Bio_3": "I cannot tell."}


def manifest_rows() -> list[dict]:
    return [{"problem_id": pid, "subject": row["subject"], "origin": pid.split("_")[0],
             "n_images": len(row["images"]), "n_options": len(row["options"])}
            for pid, row in ROWS.items()]


def audit() -> dict:
    out = {}
    for pid, row in ROWS.items():
        for n, data in row["images"].items():
            height, width = png_size(data)
            out[(pid, n)] = {"problem_id": pid, "image_number": n, "sha256": sha256_bytes(data),
                             "width": width, "height": height, "format": "PNG"}
    return out


class FakeServer:
    def __init__(self, sidecar: Path, texts: dict[str, str]) -> None:
        self.sidecar, self.texts, self.sent_images = sidecar, texts, []

    def __call__(self, *, messages, decoding, **_: object):
        user = messages[1]["content"]
        text_parts = "".join(p["text"] for p in user if p["type"] == "text")
        images = [base64.b64decode(p["image_url"]["url"].split(",", 1)[1])
                  for p in user if p["type"] == "image_url"]
        self.sent_images.append(images)
        pid = next((p for p in ROWS if ROWS[p]["question"].split("<")[0] in text_parts), None)
        content = self.texts.get(pid, "Looks red.\nAnswer: A") if pid else "Looks red.\nAnswer: A"
        pads = sum(mmmu_pro.visual_tokens(*png_size(data)) for data in images)
        pieces = [content[i:i + 3] for i in range(0, len(content), 3)]
        lps = [-0.1 * (k % 4) for k in range(len(pieces))]
        ids = [1000 + k for k in range(len(pieces))]
        response = {"choices": [{"message": {"content": content}, "finish_reason": "stop",
                                 "token_ids": ids,
                                 "logprobs": {"content": [{"token": t, "logprob": lp, "token_id": i}
                                                          for t, lp, i in zip(pieces, lps, ids)]}}],
                    "usage": {"prompt_tokens": 50 + pads, "completion_tokens": len(pieces)},
                    "prompt_token_ids": [1, 2] + [PAD] * pads + [3]}
        usage = usage_from_response(response, 12.0, retries=0)
        usage["token_logprobs"] = build_generated_token_trace(response, expected_text=content,
                                                              logprobs_mode="raw_logprobs")
        usage["prompt_token_ids"] = response["prompt_token_ids"]
        positions = []
        for k, (lp, tid) in enumerate(zip(lps, ids)):
            others = [5000 + j for j in range(19)]
            positions.append({"index": k, "sampled_token_id": tid, "sampled_raw_logit": 10.0,
                              "max_raw_logit": 10.0, "logsumexp": 10.0 - lp, "entropy": 0.5,
                              "top1_top2_margin": 1.0, "top20_token_ids": [tid] + others,
                              "top20_raw_logits": [10.0] + [9.0 - 0.1 * j for j in range(19)]})
        capture = {"capture_id": decoding["vllm_xargs"]["c014_capture_id"],
                   "positions": positions, "output_token_ids": ids}
        with self.sidecar.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(capture) + "\n")
        return content, None, usage


def fake_rows(ids):
    return {pid: ROWS[pid] for pid in ids}


class RunnerEndToEndTest(unittest.TestCase):
    def run_stage(self, condition_id: str, work: Path, texts: dict[str, str],
                  g_dir: Path | None = None) -> FakeServer:
        runner.configure(condition_id)
        runner._G_LETTERS.clear()
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

        def condition_with_g_source(cid=None):
            # c038 cannot name c036's archive until c036 has run; point it at the simulated one.
            resolved = real_condition(cid)
            if "generator_source" in resolved:
                resolved["generator_source"]["pass1_run_id"] = "c036__run"
            return resolved

        with patch.object(runner, "api_call", server), \
             patch.object(runner, "load_mmmu_pro_rows", fake_rows), \
             patch.object(runner, "image_audit", audit), \
             patch.object(runner, "condition", condition_with_g_source), \
             patch.object(runner, "require_registered", lambda cid: real_condition(cid)["model"]):
            if runner.ROLE == "review":
                g_records = work / "records" / "g_pass1.jsonl"
                g_records.write_text((g_dir / "records" / "pass1.jsonl").read_text(encoding="utf-8"),
                                     encoding="utf-8")
                (work / "records" / "g_grades.jsonl").write_text(
                    (g_dir / "records" / "grades.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
                runner.stage_review(argparse.Namespace(**common, g_records=str(g_records)))
            else:
                runner.stage_generate(argparse.Namespace(**common))
            g_args = ({"g_records": str(work / "records" / "g_pass1.jsonl"),
                       "g_grades": str(work / "records" / "g_grades.jsonl")}
                      if runner.ROLE == "review" else {"g_records": None, "g_grades": None})
            runner.stage_evidence(argparse.Namespace(
                run_id="ev", records=common["output"], logit_traces=common["logit_output"],
                output=str(work / "records" / "scalar_evidence.jsonl")))
            runner.stage_grade(argparse.Namespace(records=common["output"], g_records=g_args["g_records"],
                                                  output=str(work / "records" / "grades.jsonl")))
            runner.stage_analyze(argparse.Namespace(
                run_id="an", records=common["output"], evidence=str(work / "records" / "scalar_evidence.jsonl"),
                grades=str(work / "records" / "grades.jsonl"), expected_rows=3,
                output=str(work / "reports" / "analysis.json"), **g_args))
        (work / "condition.resolved.yaml").write_text(
            yaml.safe_dump(runner.condition(condition_id), sort_keys=False), encoding="utf-8")
        return server

    def validate_bundle(self, work: Path) -> None:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_records.py"),
                                 "--bundle-dir", str(work)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_solo_then_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            g_dir, v_dir = Path(tmp) / "g", Path(tmp) / "gv"
            server = self.run_stage("c036", g_dir, SOLVE_TEXT)
            # Images arrive as the original bytes, in first-reference order.
            self.assertIn([ROWS["validation_Math_2"]["images"][1], ROWS["validation_Math_2"]["images"][2]],
                          server.sent_images)
            pass1 = {json.loads(l)["problem_id"]: json.loads(l)
                     for l in (g_dir / "records/pass1.jsonl").read_text(encoding="utf-8").splitlines()}
            vis = pass1["validation_Math_2"]["visual_input"]
            self.assertEqual(vis["num_visual_tokens"], vis["expected_visual_tokens"])
            self.assertEqual(pass1["test_Bio_3"]["outputs"][0]["parse_status"], "model_invalid")
            native = [json.loads(l) for l in (g_dir / "records/native_logit_trace.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["status"] == "ok" for row in native))
            evidence = {json.loads(l)["problem_id"]: json.loads(l)
                        for l in (g_dir / "records/scalar_evidence.jsonl").read_text().splitlines()}
            art = evidence["test_Art_1"]["features"]
            self.assertEqual(art["answer_parseable"], 1)
            self.assertIsNotNone(art["ans_lp"])
            self.assertIsNone(evidence["test_Bio_3"]["features"]["ans_lp"])
            report = json.loads((g_dir / "reports/analysis.json").read_text())
            self.assertEqual((report["correct"], report["answer_parseable"]), (1, 2))
            self.assertTrue((g_dir / "reports/summary.md").is_file())
            self.validate_bundle(g_dir)

            self.run_stage("c038", v_dir, REVIEW_TEXT, g_dir=g_dir)
            review = {json.loads(l)["problem_id"]: json.loads(l)
                      for l in (v_dir / "records/review.jsonl").read_text(encoding="utf-8").splitlines()}
            # The reviewer quoted "the proposed Answer: B" mid-sentence; only its own line counts.
            self.assertEqual(review["validation_Math_2"]["outputs"][0]["extracted_answer"], "C")
            report = json.loads((v_dir / "reports/analysis.json").read_text())
            # Art: V A right. Math: V C right (G B wrong). Bio: V unparseable, G unparseable too.
            self.assertEqual(report["gv_primary_correct"], 2)
            self.assertEqual(report["gv_strict_correct"], 2)
            self.assertEqual(report["g_fallback_rows"], 1)
            self.assertEqual((report["gv_vs_g_repair"], report["gv_vs_g_harm"]), (1, 0))
            self.validate_bundle(v_dir)

    def test_changed_image_bytes_stop_the_run(self) -> None:
        bad = dict(audit())
        key = ("test_Art_1", 1)
        bad[key] = {**bad[key], "sha256": "0" * 64}
        with self.assertRaises(RuntimeError):
            runner.visual_input("test_Art_1", [1], ROWS["test_Art_1"]["images"], bad,
                                {"min_pixels": 65536, "max_pixels": 16777216})

    def test_check_runnable(self) -> None:
        # c038 and c039 name the archived c036 and c037 runs they review.
        for cid, budget in (("c033", 8192), ("c034", 8192), ("c036", 4096), ("c037", 4096),
                            ("c038", 4096), ("c039", 4096)):
            runner.configure(cid)
            self.assertEqual(runner.require_registered(cid)["pass1_decoding"]["max_new_tokens"], budget)
        for cid in ("c035",):
            runner.configure(cid)
            with self.assertRaises(SystemExit):
                runner.require_registered(cid)


if __name__ == "__main__":
    unittest.main()
