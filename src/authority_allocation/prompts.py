"""Prompt file parsing and variable substitution.

Only `{{name}}` occurrences in the raw template are treated as variables, so an
inserted value may safely contain braces — LaTeX, JSON, or a Python dict literal.
Relaxing that rule once stopped a run mid-flight.
"""

from __future__ import annotations

import re
from pathlib import Path

PROMPT_VARIABLE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


def split_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    marker = "\nUser template:\n"
    if not text.startswith("System:\n") or marker not in text:
        raise ValueError(f"prompt lacks System/User template sections: {path}")
    system, template = text[len("System:\n") :].split(marker, 1)
    return system.strip(), template.strip()


def render(template: str, **values: str) -> str:
    variables = set(PROMPT_VARIABLE.findall(template))
    missing = sorted(variables.difference(values))
    if missing:
        raise ValueError(f"missing prompt template variables: {', '.join(missing)}")
    return PROMPT_VARIABLE.sub(lambda match: values[match.group(1)], template)
