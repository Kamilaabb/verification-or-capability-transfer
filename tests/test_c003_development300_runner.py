from __future__ import annotations

import importlib
import json
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import scripts.run_c002_dev30 as runner_module
from authority_allocation import api as api_module


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class C003Development300RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = importlib.reload(runner_module)
        self.runner.configure_c003_development300()

    def tearDown(self) -> None:
        importlib.reload(runner_module)

    def test_profile_selects_new_frozen_identities(self) -> None:
        self.assertEqual(self.runner.CONDITION_ID, "c003")
        self.assertEqual(self.runner.PROTOCOL_ID, "authority_v6")
        self.assertEqual(self.runner.FEATURE_SET_ID, "scalar_single_v2")
        self.assertEqual(self.runner.EXPECTED_ROWS, 300)
        self.assertTrue(self.runner.CAPTURE_LOGPROBS)

    def test_api_requests_and_validates_sampled_token_logprobs(self) -> None:
        payload = {
            "choices": [{
                "message": {"content": "ok", "reasoning_content": None},
                "finish_reason": "stop",
                "token_ids": [17],
                "logprobs": {"content": [{
                    "token": "ok",
                    "bytes": [111, 107],
                    "logprob": -0.25,
                }]},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        decoding = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 8,
            "output_logprobs": {
                "enabled": True,
                "mode": "raw_logprobs",
                "top_logprobs": 0,
                "return_token_ids": True,
            },
        }
        with patch.object(
            api_module.request, "urlopen", return_value=_FakeResponse(payload)
        ) as urlopen:
            content, reasoning, usage = self.runner.api_call(
                api_base="http://127.0.0.1:8000/v1",
                served_model="model",
                messages=[{"role": "user", "content": "test"}],
                decoding=decoding,
                seed=1,
                thinking=False,
            )
        submitted = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertTrue(submitted["logprobs"])
        self.assertEqual(submitted["top_logprobs"], 0)
        self.assertTrue(submitted["return_token_ids"])
        self.assertEqual(content, "ok")
        self.assertIsNone(reasoning)
        self.assertEqual(usage["token_logprobs"]["status"], "ok")
        self.assertEqual(usage["token_logprobs"]["tokens"][0]["token_id"], 17)

    def test_api_forwards_vllm_xargs_without_mixing_them_into_decoding(self) -> None:
        payload = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        decoding = {
            "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 8,
            "vllm_xargs": {"c014_capture_id": "capture-1"},
        }
        with patch.object(
            api_module.request, "urlopen", return_value=_FakeResponse(payload)
        ) as urlopen:
            api_module.api_call(
                api_base="http://127.0.0.1:8000/v1", served_model="model",
                messages=[{"role": "user", "content": "test"}],
                decoding=decoding, seed=1, thinking=False,
            )
        submitted = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(submitted["vllm_xargs"], {"c014_capture_id": "capture-1"})

    def test_analysis_writes_lossless_multi_output_targets(self) -> None:
        self.runner.EXPECTED_ROWS = 3
        patterns = ((1, 1, 1), (0, 1, 1), (0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / f"{name}.jsonl"
                for name in ("counterfactuals", "decisions", "g", "v", "evidence")
            }
            counterfactuals = []
            decisions = []
            g_rows = []
            v_rows = []
            evidence_rows = []
            for index, pattern in enumerate(patterns):
                problem_id = f"p{index}"
                evidence_id = f"e{index}"
                evidence_hash = f"{index + 1:064x}"
                counterfactuals.append({
                    "problem_id": problem_id,
                    "cell_id": "V-A",
                    "source_evidence_record_id": evidence_id,
                    "evidence_sha256": evidence_hash,
                    "registered_request": "OVERRIDE",
                    "u0": pattern[0],
                    "z0": pattern[0],
                    "z1": pattern[1],
                    "z2": pattern[2],
                    "grader_status": {"y0": "ok", "y1": "ok", "y2": "ok"},
                })
                for menu in ("binary", "ternary"):
                    decisions.append({
                        "problem_id": problem_id,
                        "menu": menu,
                        "replicate_id": "c01",
                        "authority_cap": "M0",
                        "parse_status": "ok",
                    })
                trace = {"status": "ok"}
                g_rows.append({"outputs": [{"token_logprobs": trace, "usage": {}}]})
                v_rows.append({
                    "token_logprobs": (
                        trace if index != 2 else {"status": "text_mismatch"}
                    ),
                    "usage": {},
                })
                evidence_rows.append({
                    "feature_set_id": "scalar_single_v2",
                    "features": {"g_answer_extractable": 1},
                    "missingness": {"g_answer_extractable": False},
                })
            for name, rows in (
                ("counterfactuals", counterfactuals),
                ("decisions", decisions),
                ("g", g_rows),
                ("v", v_rows),
                ("evidence", evidence_rows),
            ):
                paths[name].write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            reports = root / "reports"
            self.runner.stage_analyze(argparse.Namespace(
                counterfactuals=str(paths["counterfactuals"]),
                controller_records=str(paths["decisions"]),
                g_records=str(paths["g"]),
                v_records=str(paths["v"]),
                rewrite_records=None,
                evidence_records=str(paths["evidence"]),
                output_dir=str(reports),
            ))
            targets = [
                json.loads(line)
                for line in (reports / "controller_training_targets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(targets[0]["cap_outcomes"], {"M0": 1, "M1": 1, "M2": 1})
            self.assertEqual(targets[1]["oracle_actions"], ["M1", "M2"])
            self.assertIsNone(targets[2]["minimal_oracle_action"])
            qa = json.loads((reports / "qa_report.json").read_text(encoding="utf-8"))
            self.assertEqual(qa["status"], "passed")
            self.assertEqual(qa["token_logprob_trace_failures"], {"G": 0, "V": 1})
            metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metrics["diagnostics"]["boundary_transitions"],
                {
                    "M0_to_M1": {"gain": 1, "harm": 0, "net": 1},
                    "M1_to_M2": {"gain": 0, "harm": 0, "net": 0},
                },
            )


if __name__ == "__main__":
    unittest.main()
