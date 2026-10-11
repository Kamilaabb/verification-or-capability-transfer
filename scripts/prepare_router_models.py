#!/usr/bin/env python3
"""Verify the c014 snapshot on disk against the provenance the repository committed.

prepare_code_models.py keeps its pin table inside the script. c014 does not, because
conditions/c014/condition.yaml already declares revision, server_path and
chat_template_hash, and the condition hash covers them. A second copy of the same
values would be a second thing to drift. So the committed provenance lives in
conditions/c014/model_provenance.json, the condition carries the fields a run reads,
and this script checks three things against each other and against the disk:

    condition.yaml  <->  model_provenance.json  <->  the snapshot directory

Any disagreement stops the job. That matters more here than in the earlier conditions:
the same-forward capture and ordinary sampled-token response are comparable only under
the registered weights, tokenizer and chat template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation.condition_resolution import resolve_condition  # noqa: E402

TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chat_template_hash(path: Path) -> str:
    config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise RuntimeError(f"missing chat_template in {path / 'tokenizer_config.json'}")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def tokenizer_hashes(path: Path) -> dict[str, str]:
    """Hash every tokenizer artefact present, not just the chat template.

    A chat template hash proves the prompt renders the same. It says nothing about the
    vocabulary or merges, and a sampled token id interpreted against a different
    vocabulary is a different token.
    """
    return {name: sha256_file(path / name) for name in TOKENIZER_FILES
            if (path / name).is_file()}


def load_condition(condition_path: Path) -> dict:
    return resolve_condition(condition_path, repository_root=ROOT)


def verify(condition_path: Path, provenance_path: Path, *, check_weights: bool) -> dict:
    condition = load_condition(condition_path)
    model = condition["model"]
    problems: list[str] = []

    for field in ("revision", "server_path", "chat_template_hash"):
        if not model.get(field):
            problems.append(f"condition model.{field} is null; the snapshot is not frozen")
    if problems:
        raise SystemExit("\n".join(problems))

    if not provenance_path.is_file():
        raise SystemExit(f"missing committed provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    # 1. condition <-> provenance
    for field, key in (("model_id", "repo_id"), ("revision", "revision"),
                       ("server_path", "server_path"),
                       ("chat_template_hash", "chat_template_hash")):
        if str(model[field]) != str(provenance.get(key)):
            problems.append(
                f"condition model.{field}={model[field]!r} disagrees with "
                f"provenance {key}={provenance.get(key)!r}")
    if provenance.get("tokenizer_revision") != provenance.get("revision"):
        problems.append(
            "tokenizer_revision differs from revision; the tokenizer must come from the "
            "same immutable commit as the weights")

    # 2. provenance <-> disk
    snapshot = Path(model["server_path"])
    if not snapshot.is_dir():
        problems.append(f"snapshot directory is absent: {snapshot}")
    else:
        if snapshot.name != model["revision"]:
            problems.append(
                f"snapshot directory {snapshot.name!r} is not named for the pinned "
                f"revision {model['revision']!r}")
        observed_template = chat_template_hash(snapshot)
        if observed_template != model["chat_template_hash"]:
            problems.append(
                f"chat template on disk hashes to {observed_template}, "
                f"condition pins {model['chat_template_hash']}")
        observed_tokenizer = tokenizer_hashes(snapshot)
        expected_tokenizer = provenance.get("tokenizer_file_sha256") or {}
        if observed_tokenizer != expected_tokenizer:
            differing = sorted(set(observed_tokenizer) ^ set(expected_tokenizer)) or [
                name for name in observed_tokenizer
                if observed_tokenizer[name] != expected_tokenizer.get(name)]
            problems.append(f"tokenizer artefacts differ from provenance: {differing}")
        if check_weights:
            expected_weights = provenance.get("weight_file_sha256") or {}
            observed_weights = {p.name: sha256_file(p)
                                for p in sorted(snapshot.glob("*.safetensors"))}
            if observed_weights != expected_weights:
                differing = sorted(set(observed_weights) ^ set(expected_weights)) or [
                    name for name in observed_weights
                    if observed_weights[name] != expected_weights.get(name)]
                problems.append(f"weight files differ from provenance: {differing}")

    # 3. the greedy decoding settings the same-forward identity check depends on
    if model.get("thinking") is not False:
        problems.append("model.thinking must be false for a recoverable greedy trace")
    for key in ("pass1_decoding", "pass2_decoding"):
        if float(model[key]["temperature"]) != 0.0:
            problems.append(f"model.{key}.temperature must be 0.0")

    if problems:
        raise SystemExit("PROVENANCE CHECK FAILED\n  " + "\n  ".join(problems))

    result = {"status": "verified", "repo_id": model["model_id"],
              "revision": model["revision"], "server_path": str(snapshot),
              "chat_template_hash": model["chat_template_hash"],
              "tokenizer_files_checked": len(provenance.get("tokenizer_file_sha256") or {}),
              "weights_checked": bool(check_weights)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default=str(ROOT / "conditions/c014/condition.yaml"))
    parser.add_argument("--provenance",
                        default=str(ROOT / "conditions/c014/model_provenance.json"))
    parser.add_argument("--check-weights", action="store_true",
                        help="also hash every safetensors shard; slow, used before the "
                             "pilot rather than on every preflight iteration")
    args = parser.parse_args()
    verify(Path(args.condition), Path(args.provenance), check_weights=args.check_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
