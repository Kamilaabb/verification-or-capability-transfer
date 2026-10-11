"""BigCodeBench code-domain study helpers and its frozen prompt
(docs/plan/code_bigcodebench_study.md).

Both prompts were fixed on 2026-09-27 and are frozen by their LF-normalized
sha256; editing either is a new study.
Programs are extracted and graded only by scripts/bigcodebench_grade.py inside the pinned
evaluate container; the host side here only counts fenced Python blocks for evidence.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from authority_allocation.prompts import PROMPT_VARIABLE, render

CODE_SOLVE_PROMPT = "conditions/c100/prompts/code_solve_v1.txt"
CODE_REVIEW_PROMPT = "conditions/c110/prompts/code_review_v1.txt"
FROZEN_PROMPTS: dict[str, str] = {
    CODE_SOLVE_PROMPT: "7e0b778c016fc7989aacb6c5dd0a59865579c266b0dcb0328d88172d69120468",
    # Fixed on 2026-09-27 with the pair Qwen3-4B x Ministral-8B (c110, c111).
    CODE_REVIEW_PROMPT: "017dfd738a89763d91d45f9cda0ed43c152cf251b99cea23ea36085ee56a7426",
}

# A complete fenced block opened as ```python or ```py; the language tag is case-insensitive.
PYTHON_BLOCK = re.compile(r"```[ \t]*(?:python3?|py)[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def require_frozen_prompt(root: Path, relative: str) -> None:
    if relative not in FROZEN_PROMPTS:
        raise SystemExit(f"{relative} is not a frozen BigCodeBench prompt")
    data = (root / relative).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != FROZEN_PROMPTS[relative]:
        raise SystemExit(f"frozen prompt {relative} no longer has its registered hash")


def user_message(template: str, *, problem: str, y0: str | None = None) -> str:
    """The user turn: the task text (the dataset's instruct_prompt, verbatim) and, for a
    review, the proposed solution, both as plain text."""
    variables = set(PROMPT_VARIABLE.findall(template))
    expected = {"problem"} | ({"y0"} if y0 is not None else set())
    if variables != expected:
        raise ValueError(f"template variables {sorted(variables)} != {sorted(expected)}")
    values = {"problem": problem}
    if y0 is not None:
        values["y0"] = y0
    return render(template, **values)


def python_block_count(text: str | None) -> int:
    """Number of complete fenced Python blocks in the output (evidence only, never grading)."""
    return len(PYTHON_BLOCK.findall(text or ""))
