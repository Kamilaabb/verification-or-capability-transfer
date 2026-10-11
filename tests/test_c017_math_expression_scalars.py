import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

import scripts.validate_records as checker

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "run_c017_math_expression_scalars", SCRIPTS / "run_c017_math_expression_scalars.py"
)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


class C017EvidenceTests(unittest.TestCase):
    def test_exact_seven_scalars_use_all_math_tokens(self):
        text = "First $x=2$. Then $y=3$."
        chunks = ["First ", "$x=2", "$", ". Then ", "$y=3", "$."]
        tokens = []
        cursor = 0
        for index, chunk in enumerate(chunks):
            encoded = chunk.encode()
            tokens.append({"index": index, "token_id": index, "token": chunk,
                           "bytes": list(encoded), "start_byte": cursor,
                           "end_byte": cursor + len(encoded), "logprob": -0.1 * (index + 1)})
            cursor += len(encoded)
        trace = {"status": "ok", "logprobs_mode": "raw_logprobs", "tokens": tokens,
                 "reconstructed_text_sha256": "0" * 64, "content_start_byte": 0,
                 "content_end_byte": cursor}
        pass1 = {"record_id": "p1", "run_id": "pass1", "condition_id": "c017",
                 "problem_id": "p", "problem_text": "Problem?",
                 "outputs": [{"raw_output": text, "extracted_answer": "3",
                              "token_logprobs": trace, "usage": {"output_tokens": 6}}]}
        native = {"run_id": "logits", "status": "ok",
                  "positions": [{"index": i, "max_raw_logit": float(10 + i)}
                                for i in range(6)]}
        row = runner.derive_evidence(pass1, native, run_id="evidence")
        self.assertEqual(len(row["features"]), 7)
        self.assertEqual(row["math_token_indices"], [1, 4])
        self.assertTrue(math.isclose(
            row["features"]["g_math_expression_mean_max_logit"], 12.5))
        self.assertTrue(math.isclose(
            row["features"]["g_math_expression_mean_logprob"], -0.35))
        self.assertTrue(math.isclose(row["features"]["g_output_mean_max_logit"], 12.5))
        self.assertTrue(math.isclose(row["features"]["g_output_mean_logprob"], -0.35))
        self.assertEqual(row["features"]["g_answer_parseable"], 1)
        self.assertEqual(row["features"]["problem_char_count"], 8)
        self.assertEqual(row["features"]["g_output_char_count"], len(text))
        schema_path = (ROOT / "protocols/math_expression_scalar_v1/schemas/"
                       "evidence_record_v1.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        errors = []
        checker.validate(row, schema, checker.SchemaStore(), schema_path, schema,
                         "evidence", errors)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
