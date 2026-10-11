"""Pure helpers for the MMLU-Pro knowledge study (docs/plan/mmlu_pro_study.md).

The answer format, answer parser and option rendering are the MMMU-Pro study's, imported
unchanged from authority_allocation.mmmu_pro, so the two multiple-choice domains read and
grade answers identically. Nothing here reads the dataset.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from authority_allocation.mmmu_pro import (  # noqa: F401  (re-exported for the runner)
    OPTION_LETTERS, answer_is_option, locate_answer_letter, parse_answer_letter, render_options,
)
from authority_allocation.prompts import PROMPT_VARIABLE, render

# Prompts frozen for the MMLU-Pro study, dev and test alike (docs/plan/mmlu_pro_study.md
# section 4). Hashes are over LF line endings, as the repository stores them. Changing a frozen
# prompt starts a new study on this dataset, not a new condition.
MC_SOLVE_PROMPT = "conditions/c070/prompts/mc_solve_v1.txt"
MC_REVIEW_PROMPT = "conditions/c072/prompts/mc_review_v1.txt"
# The commit-first review prompt of the prompt-variant sanity check (c392, c393;
# docs/plan/prompt_variant_test500.md); frozen 2026-10-09, used by no other condition.
MC_REVIEW_COMMIT_FIRST_PROMPT = "conditions/c392/prompts/mc_commit_first_v1.txt"
FROZEN_PROMPTS: dict[str, str] = {
    MC_SOLVE_PROMPT: "493392f32bace2c53e29cc10a7dc12a28e5e27639ad553ce4e5eb5c3ad7a571d",
    MC_REVIEW_PROMPT: "4eee45c0ca6802868eaca389d6f81bba103cc80e927c6ad367eacc410add3dc7",
    MC_REVIEW_COMMIT_FIRST_PROMPT: "01e4c9e04ddde0fc219fc0cd53b107e3a133007ba0f7a6e655ae4b7c56538ede",
}


def require_frozen_prompt(root: Path, relative: str) -> None:
    if relative not in FROZEN_PROMPTS:
        raise SystemExit(f"{relative} is not a frozen MMLU-Pro prompt")
    data = (root / relative).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != FROZEN_PROMPTS[relative]:
        raise SystemExit(f"frozen prompt {relative} no longer has its registered hash")


def user_message(template: str, *, question: str, options: Sequence[str],
                 y0: str | None = None) -> str:
    """The user turn: the template with the question, the rendered options and, for a
    review, the proposed solution, all as plain text."""
    variables = set(PROMPT_VARIABLE.findall(template))
    expected = {"question", "options"} | ({"y0"} if y0 is not None else set())
    if variables != expected:
        raise ValueError(f"template variables {sorted(variables)} != {sorted(expected)}")
    values = {"question": question, "options": render_options(options)}
    if y0 is not None:
        values["y0"] = y0
    return render(template, **values)


def official_parse_without_random(response: str | None, n_options: int) -> str | None:
    """MMLU-Pro's evaluate_from_local.py extraction chain with its random fallback removed.

    "answer is (X)", then the last "Answer: X", then the last standalone capital letter.
    Where the official code guesses at random this returns None. Used only to compare with
    published numbers; the registered reading is parse_answer_letter.
    """
    if not response:
        return None
    found = None
    match = re.search(r"answer is \(?([A-J])\)?", response)
    if match:
        found = match.group(1)
    else:
        match = re.search(r".*[aA]nswer:\s*([A-J])", response)
        if match:
            found = match.group(1)
        else:
            match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", response, re.DOTALL)
            if match:
                found = match.group(0)
    return found if found is not None and answer_is_option(found, n_options) else None


def source_origin(src: str) -> str:
    """The origin of an MMLU-Pro row: the ``src`` prefix before the first '-'
    (ori_mmlu, stemez, theoremQA, scibench)."""
    return src.split("-", 1)[0]


def question_key(question: str, options: Sequence[str]) -> str:
    import json

    payload = json.dumps([question.strip(), [o.strip() for o in options]], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"problem_id": str(row["question_id"]), "category": str(row["category"]),
            "origin": source_origin(str(row["src"])), "n_options": len(row["options"])}
