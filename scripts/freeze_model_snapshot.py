#!/usr/bin/env python3
"""Download a model snapshot at a resolved commit and report the values to pin.

prepare_c002_models.py verifies snapshots whose revision and chat-template hash are
already frozen. A model entering the stack for the first time has neither, so this
script resolves what a branch currently points at, downloads that exact commit, and
prints the fields a condition config needs. The printed values are not themselves a
pin: they are copied into a prepare script and committed, and that committed table
is what later jobs check against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def model_path(root: Path, repo_id: str, revision: str) -> Path:
    return root / repo_id.replace("/", "--") / revision


def chat_template_hash(path: Path) -> str:
    config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise RuntimeError(f"missing chat_template in {path / 'tokenizer_config.json'}")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--branch",
        default="main",
        help="branch or tag to resolve into an immutable commit sha",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/path/to/scratch/authority-allocation/models"),
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(args.repo_id, revision=args.branch).sha
    if not revision:
        raise RuntimeError(f"could not resolve {args.repo_id}@{args.branch}")

    target = model_path(args.model_root, args.repo_id, revision)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo_id, revision=revision, local_dir=target)

    safetensors = sorted(target.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError(f"no safetensors weights found in {target}")

    print("PIN repo_id             ", args.repo_id)
    print("PIN resolved_from       ", args.branch)
    print("PIN revision            ", revision)
    print("PIN chat_template_sha256", chat_template_hash(target))
    print("PIN server_path         ", target)
    print("PIN safetensors_files   ", len(safetensors))
    print("PIN bytes               ", sum(f.stat().st_size for f in target.rglob("*") if f.is_file()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
