#!/usr/bin/env python3
"""Emit the c014 model provenance, and fill the condition's null identity fields.

The freeze job resolves a branch into an immutable commit and downloads it. What it
produces has to end up in two places that are checked against each other later: the
condition's model block, which a run reads, and a committed provenance file carrying
the hashes nobody needs at run time but everybody needs when a result is questioned.

This writes both from one scan of the snapshot, so they cannot disagree at birth. It is
run on the login node, its output is reviewed as a diff, and it is committed by a human.
A Slurm job editing a registered condition on its own is exactly what "completed runs
are immutable" exists to prevent, so the freeze sbatch calls this with --dry-run and
prints the diff instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_router_models import chat_template_hash, sha256_file, tokenizer_hashes  # noqa: E402


def scan(snapshot: Path, repo_id: str, revision: str, resolved_from: str) -> dict:
    if not snapshot.is_dir():
        raise SystemExit(f"snapshot directory is absent: {snapshot}")
    if snapshot.name != revision:
        raise SystemExit(
            f"snapshot directory {snapshot.name!r} is not named for revision {revision!r}; "
            "freeze_model_snapshot.py writes <root>/<repo--with--dashes>/<sha>")
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        raise SystemExit(f"no safetensors weights in {snapshot}")
    return {
        "schema_version": "model-provenance-v1",
        "repo_id": repo_id,
        "resolved_from": resolved_from,
        "revision": revision,
        "tokenizer_revision": revision,
        "tokenizer_revision_note": (
            "The tokenizer ships inside the same snapshot, so it carries the weights' "
            "commit by construction. It is recorded explicitly because a sampled "
            "token id interpreted against a different vocabulary is a different token."),
        "server_path": str(snapshot),
        "chat_template_hash": chat_template_hash(snapshot),
        "tokenizer_file_sha256": tokenizer_hashes(snapshot),
        "weight_file_sha256": {p.name: sha256_file(p) for p in weights},
        "config_sha256": sha256_file(snapshot / "config.json")
        if (snapshot / "config.json").is_file() else None,
        "total_bytes": sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file()),
    }


def fill_condition(text: str, provenance: dict) -> str:
    """Replace the three null identity fields in the condition's model block.

    Line-oriented on purpose: a YAML round trip would reflow the whole file and bury the
    three lines that matter in a diff nobody can review.
    """
    replacements = {
        "  revision: null": f'  revision: {provenance["revision"]}',
        "  server_path: null": f'  server_path: {provenance["server_path"]}',
        "  chat_template_hash: null": f'  chat_template_hash: {provenance["chat_template_hash"]}',
    }
    for old, new in replacements.items():
        if old not in text:
            raise SystemExit(
                f"expected {old!r} in the condition; it is already filled or was edited "
                "by hand. Refusing to guess.")
        text = text.replace(old, new, 1)
    marker = "  revision_must_be_filled_by: freeze_and_preflight_job"
    if marker in text:
        text = text.replace(
            marker,
            "  revision_resolved_from: " + provenance["resolved_from"] + "\n"
            "  tokenizer_revision: " + provenance["tokenizer_revision"] + "\n"
            "  provenance: conditions/c014/model_provenance.json", 1)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--resolved-from", default="main")
    parser.add_argument("--condition", default=str(ROOT / "conditions/c014/condition.yaml"))
    parser.add_argument("--provenance",
                        default=str(ROOT / "conditions/c014/model_provenance.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be written and change nothing")
    args = parser.parse_args()

    provenance = scan(Path(args.snapshot), args.repo_id, args.revision, args.resolved_from)
    condition_path = Path(args.condition)
    filled = fill_condition(condition_path.read_text(encoding="utf-8"), provenance)
    body = json.dumps(provenance, indent=2, sort_keys=True) + "\n"

    if args.dry_run:
        print(body)
        print("--- condition.yaml model block, as it would be written ---")
        for line in filled.splitlines():
            if line.startswith(("  revision:", "  server_path:", "  chat_template_hash:",
                                "  tokenizer_revision:", "  revision_resolved_from:",
                                "  provenance:")):
                print(line)
        print("\nDry run: nothing written. Re-run without --dry-run on the login node, "
              "review the diff, and commit it.")
        return 0

    Path(args.provenance).write_text(body, encoding="utf-8")
    condition_path.write_text(filled, encoding="utf-8")
    print(json.dumps({"wrote": [args.provenance, str(condition_path)],
                      "revision": provenance["revision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
