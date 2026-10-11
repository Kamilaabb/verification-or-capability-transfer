#!/usr/bin/env python3
"""The natural-language-reasoning domain of the GV vs max(G, V) study
(docs/plan/nl_reasoning_study.md): ReClor validation, FOLIO v2 validation and GPQA Diamond pooled
into one 901-row manifest, run for two pre-chosen pairs, Qwen3-8B x Llama-3.2-3B and
Qwen3-4B x Qwen3-8B.

    c192  Llama-3.2-3B alone           (c074 with only the dataset changed)
    c193  Qwen3-8B alone               (c071 with only the dataset changed)
    c194  Qwen3-8B reviewing c192      (c135 with only the dataset and reviewed outputs changed)
    c195  Llama-3.2-3B reviewing c193  (c134 with only the dataset and reviewed outputs changed)
    c196  Qwen3-4B alone               (c070 with only the dataset changed)
    c197  Qwen3-8B reviewing c196      (c072 with only the dataset and reviewed outputs changed)
    c198  Qwen3-4B reviewing c193      (c073 with only the dataset and reviewed outputs changed)
    c303-c307  Qwen2.5-3B, Qwen2.5-7B, Ministral-8B, Mistral-Small-24B, Qwen3-32B alone
               (c075-c079 with only the dataset changed)
    c308-c315  reviews for four more pairs: Qwen3-32B/Qwen2.5-3B (c308, c309), Mistral-Small-24B/Llama-3.2-3B (c310, c311),
               Qwen3-32B/Mistral-Small-24B (c312, c313), Llama-3.2-3B/Qwen2.5-3B (c314, c315)
    c316-c359  reviews of the other 22 pairs of the eight models, both directions

Usage and commands are scripts/run_mmlu_pro.py's. This file imports that runner and swaps only
the dataset: its id, registration, row count, condition table and the loader that joins the
id-only manifest to its content file. Prompts, decoding, capture, answer parsing, evidence,
grading and both GV delivery readings are the MMLU-Pro runner's own code, unchanged.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mmlu_pro as mc  # noqa: E402

from authority_allocation import mmlu_pro  # noqa: E402
from authority_allocation.records import read_jsonl  # noqa: E402

DATASET_ID = "nlreason-gpqad-reclorval-foliov2-s20261001"
DATASET_PATH = ROOT / "datasets" / DATASET_ID / "dataset.yaml"
MANIFEST = "eval"
EXPECTED_ROWS = 901
STAGE_PREFIX = "nlreason-"

# Each condition differs from its MMLU-Pro dev1000 base only in these sections (the dataset, the
# evidence schema id, the seed cell, text and, for a review, the reviewed outputs and stage names).
ALLOWED_SECTIONS = {"condition_id", "status", "purpose", "gate", "runnable_gate", "replicates",
                    "dataset", "protocol", "generator_source", "stages"}
_SPECS = {"c192": ("c074", None), "c193": ("c071", None), "c194": ("c135", "c192"),
          "c195": ("c134", "c193"),
          # Second pair (decided 2026-10-01): Qwen3-4B x Qwen3-8B, the small-gap pair also run
          # in the knowledge, math and code domains. Qwen3-8B alone is c193.
          "c196": ("c070", None), "c197": ("c072", "c196"), "c198": ("c073", "c193"),
          # Solo runs of five more models (decided 2026-10-03), so further pairs can be reviewed later.
          "c303": ("c075", None), "c304": ("c076", None), "c305": ("c077", None),
          "c306": ("c078", None), "c307": ("c079", None),
          # Reviews of four more pairs from the knowledge-domain stratified design (decided 2026-10-03):
          # strong/strong, weak/weak and two strong/weak, chosen without looking at NL accuracy.
          "c308": ("c172", "c303"), "c309": ("c173", "c307"), "c310": ("c174", "c192"), "c311": ("c175", "c306"),
          "c312": ("c184", "c306"), "c313": ("c185", "c307"), "c314": ("c130", "c303"), "c315": ("c131", "c192"),
          # The remaining 22 pairs of the eight models, both directions (decided 2026-10-03), so the NL
          # domain covers all 28 pairs. The base is any knowledge-domain review condition with the same reviewer.
          "c316": ("c073", "c192"), "c317": ("c134", "c196"), "c318": ("c073", "c303"),
          "c319": ("c131", "c196"), "c320": ("c073", "c304"), "c321": ("c126", "c196"),
          "c322": ("c073", "c305"), "c323": ("c128", "c196"), "c324": ("c135", "c303"),
          "c325": ("c131", "c193"), "c326": ("c135", "c304"), "c327": ("c126", "c193"),
          "c328": ("c135", "c305"), "c329": ("c128", "c193"), "c330": ("c134", "c304"),
          "c331": ("c126", "c192"), "c332": ("c134", "c305"), "c333": ("c128", "c192"),
          "c334": ("c131", "c304"), "c335": ("c126", "c303"), "c336": ("c131", "c305"),
          "c337": ("c128", "c303"), "c338": ("c126", "c305"), "c339": ("c128", "c304"),
          "c340": ("c073", "c306"), "c341": ("c174", "c196"), "c342": ("c135", "c306"),
          "c343": ("c174", "c193"), "c344": ("c131", "c306"), "c345": ("c174", "c303"),
          "c346": ("c126", "c306"), "c347": ("c174", "c304"), "c348": ("c128", "c306"),
          "c349": ("c174", "c305"), "c350": ("c073", "c307"), "c351": ("c172", "c196"),
          "c352": ("c135", "c307"), "c353": ("c172", "c193"), "c354": ("c134", "c307"),
          "c355": ("c172", "c192"), "c356": ("c126", "c307"), "c357": ("c172", "c304"),
          "c358": ("c128", "c307"), "c359": ("c172", "c305")}
CONDITIONS: dict[str, dict[str, Any]] = {}
for _cid, (_base, _g) in _SPECS.items():
    CONDITIONS[_cid] = {**{k: v for k, v in mc.CONDITIONS[_base].items() if k not in ("base", "g")},
                        "base": _base, "manifest": MANIFEST, "allowed_sections": ALLOWED_SECTIONS,
                        "allowed_model_keys": set(), **({"g": _g} if _g else {})}


def source_rows_path() -> Path:
    manifest = mc.registered_dataset()["manifests"][MANIFEST]
    return Path(manifest["server_path"]).with_name("source_rows.jsonl")


def load_nl_rows(problem_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Join an id-only manifest to the content file written once by the builder, after checking
    the file's sha256 against the dataset registration."""
    path = source_rows_path()
    expected = mc.registered_dataset()["source_rows"]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{path} does not have the registered sha256")
    wanted = set(problem_ids)
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pid = str(row["problem_id"])
        if pid not in wanted:
            continue
        if pid in rows:
            raise ValueError(f"duplicate source row {pid}")
        options = [str(o) for o in row["options"]]
        if not mmlu_pro.answer_is_option(str(row["answer"]), len(options)):
            raise ValueError(f"{pid} reference letter is not one of its options")
        rows[pid] = {"question": str(row["question"]), "options": options, "answer": str(row["answer"]),
                     "category": str(row["category"]), "origin": str(row["origin"])}
    missing = wanted.difference(rows)
    if missing:
        raise ValueError(f"manifest ids missing from the content file: {sorted(missing)[:5]}")
    return rows


_mmlu_configure = mc.configure
_mmlu_analyze = mc.stage_analyze


def configure(condition_id: str) -> None:
    _mmlu_configure(condition_id)
    mc.shared.STAGE = STAGE_PREFIX + mc.ROLE


def stage_analyze(args) -> None:  # noqa: ANN001
    _mmlu_analyze(args)
    summary = Path(args.output).parent / "summary.md"
    lines = summary.read_text(encoding="utf-8").split("\n")
    lines[0] = lines[0].replace("MMLU-Pro", "NL-reasoning (ReClor + FOLIO + GPQA Diamond)", 1)
    summary.write_text("\n".join(lines), encoding="utf-8")


def install() -> None:
    """Point the MMLU-Pro runner at this dataset. Its functions read these module globals at
    call time, so nothing else changes."""
    mc.DATASET_ID = DATASET_ID
    mc.DATASET_PATH = DATASET_PATH
    mc.EXPECTED_ROWS = EXPECTED_ROWS
    mc.CONDITIONS = CONDITIONS
    mc.load_mmlu_pro_rows = load_nl_rows
    mc.configure = configure
    mc.stage_analyze = stage_analyze
    mc.__doc__ = __doc__


def main() -> int:
    install()
    return mc.main()


if __name__ == "__main__":
    raise SystemExit(main())
