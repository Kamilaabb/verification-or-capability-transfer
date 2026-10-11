"""Provenance checks for c014, exercised on a synthetic snapshot.

The point of these is that the failure modes fail. A provenance checker that passes on
a mismatched tokenizer is worse than none, because the teacher-forced logit stage
replays archived token ids and a token id under a different vocabulary is a different
token.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_router_models import chat_template_hash, tokenizer_hashes, verify  # noqa: E402
from write_model_provenance import fill_condition, scan  # noqa: E402

REVISION = "b" * 40
TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


def build_snapshot(root: Path, template: str = TEMPLATE, vocab: str = '{"a":0}') -> Path:
    snapshot = root / "Qwen--Qwen3-4B" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}), encoding="utf-8")
    (snapshot / "tokenizer.json").write_text(vocab, encoding="utf-8")
    (snapshot / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def build_condition(root: Path, snapshot: Path, **overrides) -> Path:
    model = {"revision": REVISION, "server_path": str(snapshot),
             "chat_template_hash": chat_template_hash(snapshot),
             "model_id": "Qwen/Qwen3-4B", "thinking": False,
             "pass1_decoding": {"temperature": 0.0},
             "pass2_decoding": {"temperature": 0.0}}
    model.update(overrides)          # overrides win; they are the point of the fixture
    path = root / "condition.yaml"
    import yaml
    path.write_text(yaml.safe_dump({"model": model}), encoding="utf-8")
    return path


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.snapshot = build_snapshot(self.root)
        self.provenance = self.root / "prov.json"
        self.provenance.write_text(json.dumps(
            scan(self.snapshot, "Qwen/Qwen3-4B", REVISION, "main")), encoding="utf-8")
        self.condition = build_condition(self.root, self.snapshot)

    def run_verify(self, **kwargs):
        return verify(self.condition, self.provenance, check_weights=kwargs.pop("weights", True))

    def test_a_consistent_snapshot_verifies(self):
        result = self.run_verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["revision"], REVISION)

    def test_a_changed_chat_template_is_caught(self):
        (self.snapshot / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ 'different' }}"}), encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            self.run_verify()
        self.assertIn("chat template on disk", str(caught.exception))

    def test_a_changed_tokenizer_vocabulary_is_caught(self):
        # The chat template is untouched, so a template-only check would pass here.
        (self.snapshot / "tokenizer.json").write_text('{"a":0,"b":1}', encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            self.run_verify()
        self.assertIn("tokenizer artefacts differ", str(caught.exception))

    def test_changed_weights_are_caught_when_requested(self):
        (self.snapshot / "model.safetensors").write_bytes(b"other weights")
        with self.assertRaises(SystemExit) as caught:
            self.run_verify()
        self.assertIn("weight files differ", str(caught.exception))

    def test_condition_disagreeing_with_provenance_is_caught(self):
        self.condition = build_condition(self.root, self.snapshot, revision="c" * 40)
        with self.assertRaises(SystemExit) as caught:
            self.run_verify()
        self.assertIn("disagrees with provenance", str(caught.exception))

    def test_a_tokenizer_from_another_commit_is_rejected(self):
        provenance = json.loads(self.provenance.read_text(encoding="utf-8"))
        provenance["tokenizer_revision"] = "d" * 40
        self.provenance.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            self.run_verify()
        self.assertIn("tokenizer_revision differs", str(caught.exception))

    def test_thinking_on_or_a_nonzero_temperature_is_rejected(self):
        for overrides, expected in (({"thinking": True}, "thinking must be false"),
                                    ({"pass1_decoding": {"temperature": 0.7}}, "temperature")):
            with self.subTest(overrides=overrides):
                self.condition = build_condition(self.root, self.snapshot, **overrides)
                with self.assertRaises(SystemExit) as caught:
                    self.run_verify()
                self.assertIn(expected, str(caught.exception))

    def test_an_unfrozen_condition_says_so_plainly(self):
        self.condition = build_condition(self.root, self.snapshot, revision=None)
        with self.assertRaises(SystemExit) as caught:
            self.run_verify()
        self.assertIn("not frozen", str(caught.exception))


class ScanTests(unittest.TestCase):
    def test_scan_records_tokenizer_and_weight_hashes(self):
        root = Path(tempfile.mkdtemp())
        snapshot = build_snapshot(root)
        provenance = scan(snapshot, "Qwen/Qwen3-4B", REVISION, "main")
        self.assertEqual(provenance["revision"], REVISION)
        self.assertEqual(provenance["tokenizer_revision"], REVISION)
        self.assertIn("tokenizer.json", provenance["tokenizer_file_sha256"])
        self.assertEqual(
            provenance["weight_file_sha256"]["model.safetensors"],
            hashlib.sha256(b"weights").hexdigest())

    def test_a_directory_not_named_for_the_revision_is_rejected(self):
        root = Path(tempfile.mkdtemp())
        snapshot = build_snapshot(root)
        renamed = snapshot.parent / "main"
        snapshot.rename(renamed)
        with self.assertRaises(SystemExit):
            scan(renamed, "Qwen/Qwen3-4B", REVISION, "main")


class FillConditionTests(unittest.TestCase):
    PROVENANCE = {"revision": REVISION, "server_path": "/scratch/x/" + REVISION,
                  "chat_template_hash": "e" * 64, "tokenizer_revision": REVISION,
                  "resolved_from": "main"}

    def test_fills_the_three_null_identity_fields(self):
        text = ("model:\n  revision: null\n  server_path: null\n"
                "  chat_template_hash: null\n  revision_must_be_filled_by: freeze_and_preflight_job\n")
        filled = fill_condition(text, self.PROVENANCE)
        self.assertIn(f"  revision: {REVISION}", filled)
        self.assertIn("  server_path: /scratch/x/", filled)
        self.assertIn("  chat_template_hash: " + "e" * 64, filled)
        self.assertIn(f"  tokenizer_revision: {REVISION}", filled)
        self.assertNotIn("revision_must_be_filled_by", filled)

    def test_refuses_to_overwrite_an_already_frozen_condition(self):
        text = f"model:\n  revision: {REVISION}\n  server_path: /x\n  chat_template_hash: y\n"
        with self.assertRaises(SystemExit) as caught:
            fill_condition(text, self.PROVENANCE)
        self.assertIn("already filled", str(caught.exception))

    def test_the_real_condition_matches_the_committed_provenance(self):
        import yaml

        model = yaml.safe_load(
            (ROOT / "conditions/c014/condition.yaml").read_text(encoding="utf-8"))["model"]
        provenance = json.loads(
            (ROOT / "conditions/c014/model_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(model["model_id"], provenance["repo_id"])
        for field in ("revision", "server_path", "chat_template_hash"):
            self.assertTrue(model[field])
            self.assertEqual(model[field], provenance[field])
        self.assertEqual(model["tokenizer_revision"], provenance["tokenizer_revision"])


if __name__ == "__main__":
    unittest.main()
