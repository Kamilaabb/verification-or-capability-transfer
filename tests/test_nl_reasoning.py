"""The natural-language-reasoning domain: the manifest builder on synthetic source files, and the
runner wrapper's loader and isolation. No data, model or GPU is needed."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_nlreason_manifests as builder  # noqa: E402
import run_mmlu_pro as mc  # noqa: E402
import run_nl_reasoning as nl  # noqa: E402

GPQA_FIELDS = ["Record ID", "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2",
               "Incorrect Answer 3", "High-level domain"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sources(tmp: Path, *, ambiguous_gpqa: bool = False) -> tuple[Path, Path, Path]:
    gpqa = tmp / "gpqa_diamond.csv"
    with gpqa.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GPQA_FIELDS)
        writer.writeheader()
        writer.writerow({"Record ID": "recB", "Question": " What is heavier? ", "Correct Answer": "lead ",
                         "Incorrect Answer 1": "air", "Incorrect Answer 2": "foam",
                         "Incorrect Answer 3": "air", "High-level domain": "Physics"})
        writer.writerow({"Record ID": "recA", "Question": "Which is a noble gas?", "Correct Answer": "neon",
                         "Incorrect Answer 1": "oxygen" if not ambiguous_gpqa else "neon",
                         "Incorrect Answer 2": "iron", "Incorrect Answer 3": "carbon",
                         "High-level domain": "Chemistry"})
    reclor = tmp / "val.json"
    reclor.write_text(json.dumps([
        {"context": "All cats purr.", "question": "Which follows?", "answers": ["a", "b", "c", "d"],
         "label": 2, "id_string": "val_0"},
        {"context": "Some dogs bark.", "question": "Which follows?", "answers": ["e", "f", "g", "h"],
         "label": 0, "id_string": "val_1"}]), encoding="utf-8")
    folio = tmp / "folio.jsonl"
    folio.write_text("\n".join(json.dumps(r) for r in [
        {"story_id": 1, "premises": "All men are mortal.\nSocrates is a man.", "conclusion": "Socrates is mortal.",
         "label": "True", "example_id": 7},
        {"story_id": 2, "premises": "No fish fly.", "conclusion": "Nemo flies.", "label": "Uncertain",
         "example_id": 8}]) + "\n", encoding="utf-8")
    return gpqa, reclor, folio


def run_build(gpqa: Path, reclor: Path, folio: Path):
    hashes = {"gpqa_diamond_csv": sha(gpqa), "reclor_val_json": sha(reclor),
              "folio_v2_validation_jsonl": sha(folio)}
    return builder.build(gpqa, reclor, folio, input_sha256=hashes,
                         expected_rows={"gpqa_diamond": 2, "reclor": 2, "folio": 2})


class BuilderTest(unittest.TestCase):
    def test_rows_and_reference_letters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ordered, report = run_build(*write_sources(Path(tmp)))
        by_id = {r["problem_id"]: r for r in ordered}
        self.assertEqual(report["rows_after_exclusion"], 6)
        self.assertEqual(report["excluded"], {})
        self.assertEqual(set(by_id), {"gpqa/recA", "gpqa/recB", "reclor/val_0", "reclor/val_1",
                                      "folio/7", "folio/8"})
        for pid in ("gpqa/recA", "gpqa/recB"):
            row = by_id[pid]
            self.assertEqual(sorted(row["option_permutation"]), [0, 1, 2, 3])
            self.assertEqual(row["options"][builder.LETTERS.index(row["answer"])], row["correct_text"])
        self.assertEqual(by_id["gpqa/recB"]["question"], "What is heavier?")
        self.assertEqual(by_id["gpqa/recB"]["origin"], "gpqa_physics")
        self.assertEqual(by_id["reclor/val_0"]["answer"], "C")
        self.assertEqual(by_id["reclor/val_0"]["question"], "All cats purr.\n\nWhich follows?")
        self.assertEqual(by_id["folio/7"]["options"], ["True", "False", "Uncertain"])
        self.assertEqual((by_id["folio/7"]["answer"], by_id["folio/8"]["answer"]), ("A", "C"))
        self.assertIn("Socrates is a man.", by_id["folio/7"]["question"])

    def test_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sources = write_sources(Path(tmp))
            first, _ = run_build(*sources)
            second, _ = run_build(*sources)
        self.assertEqual(first, second)

    def test_ambiguous_gold_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ordered, report = run_build(*write_sources(Path(tmp), ambiguous_gpqa=True))
        self.assertEqual(report["excluded"], {"gold_ambiguous": ["gpqa/recA"]})
        self.assertNotIn("gpqa/recA", {r["problem_id"] for r in ordered})

    def test_wrong_input_hash_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                builder.build(*write_sources(Path(tmp)))

    def test_manifest_is_id_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ordered, _ = run_build(*write_sources(Path(tmp)))
        row = builder.manifest_row(ordered[0])
        self.assertEqual(set(row), {"problem_id", "category", "origin", "n_options"})
        self.assertNotIn("answer", builder.manifest_row(ordered[0]))


class WrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {name: getattr(mc, name) for name in
                      ("DATASET_ID", "DATASET_PATH", "EXPECTED_ROWS", "CONDITIONS", "load_mmlu_pro_rows",
                       "configure", "stage_analyze", "__doc__")}

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            setattr(mc, name, value)

    def test_import_leaves_the_mmlu_runner_alone(self) -> None:
        self.assertEqual(mc.DATASET_ID, "mmlupro-rb189ec7-s20260927")
        self.assertEqual(mc.EXPECTED_ROWS, 1000)
        self.assertNotIn("c192", mc.CONDITIONS)

    def test_every_ordered_pair_of_the_eight_models_is_reviewed_once(self) -> None:
        solos = ["c192", "c193", "c196", "c303", "c304", "c305", "c306", "c307"]
        models = {c: nl.CONDITIONS[c]["model_id"] for c in solos}
        self.assertEqual(len(set(models.values())), 8)
        seen = []
        for cid, spec in nl.CONDITIONS.items():
            if spec["role"] == "review":
                seen.append((spec["model_id"], models[spec["g"]]))
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {(a, b) for a in models.values() for b in models.values() if a != b})

    def test_condition_table(self) -> None:
        self.assertEqual(set(nl.CONDITIONS), {"c192", "c193", "c194", "c195", "c196", "c197", "c198",
                                              "c303", "c304", "c305", "c306", "c307"} | {f"c{n}" for n in range(308, 360)})
        self.assertEqual((nl.CONDITIONS["c197"]["g"], nl.CONDITIONS["c198"]["g"]), ("c196", "c193"))
        expected = {"c308": ("c303", "Qwen/Qwen3-32B"), "c309": ("c307", "Qwen/Qwen2.5-3B-Instruct"),
                    "c310": ("c192", "mistralai/Mistral-Small-24B-Instruct-2501"),
                    "c311": ("c306", "meta-llama/Llama-3.2-3B-Instruct"), "c312": ("c306", "Qwen/Qwen3-32B"),
                    "c313": ("c307", "mistralai/Mistral-Small-24B-Instruct-2501"),
                    "c314": ("c303", "meta-llama/Llama-3.2-3B-Instruct"), "c315": ("c192", "Qwen/Qwen2.5-3B-Instruct")}
        for cid, (g, reviewer) in expected.items():
            self.assertEqual((nl.CONDITIONS[cid]["role"], nl.CONDITIONS[cid]["g"], nl.CONDITIONS[cid]["model_id"]),
                             ("review", g, reviewer))
        self.assertEqual(nl.CONDITIONS["c196"]["model_id"], "Qwen/Qwen3-4B")
        self.assertEqual(nl.CONDITIONS["c192"]["model_id"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(nl.CONDITIONS["c193"]["model_id"], "Qwen/Qwen3-8B")
        self.assertEqual((nl.CONDITIONS["c194"]["role"], nl.CONDITIONS["c194"]["g"]), ("review", "c192"))
        self.assertEqual((nl.CONDITIONS["c195"]["role"], nl.CONDITIONS["c195"]["g"]), ("review", "c193"))
        self.assertEqual(nl.CONDITIONS["c194"]["model_id"], "Qwen/Qwen3-8B")
        self.assertEqual(nl.CONDITIONS["c195"]["model_id"], "meta-llama/Llama-3.2-3B-Instruct")
        for spec in nl.CONDITIONS.values():
            self.assertEqual(spec["manifest"], "eval")

    def test_loader_checks_the_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ordered, _ = run_build(*write_sources(Path(tmp)))
            out = Path(tmp) / "manifests"
            out.mkdir()
            builder.write_jsonl(out / "eval.jsonl", [builder.manifest_row(r) for r in ordered])
            builder.write_jsonl(out / "source_rows.jsonl", [builder.source_row(r) for r in ordered])
            registration = Path(tmp) / "dataset.yaml"
            registration.write_text(yaml.safe_dump({
                "manifests": {"eval": {"server_path": str(out / "eval.jsonl")}},
                "source_rows": {"sha256": sha(out / "source_rows.jsonl")}}), encoding="utf-8")
            nl.install()
            mc.DATASET_PATH = registration
            rows = nl.load_nl_rows(["folio/7", "gpqa/recA"])
            self.assertEqual(rows["folio/7"]["answer"], "A")
            self.assertEqual(len(rows["gpqa/recA"]["options"]), 4)
            with self.assertRaises(ValueError):
                nl.load_nl_rows(["folio/999"])
            (out / "source_rows.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                nl.load_nl_rows(["folio/7"])


if __name__ == "__main__":
    unittest.main()
