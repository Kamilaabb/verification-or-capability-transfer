"""Locate auditable mathematical-expression spans in model output.

The parser is deliberately lexical: it identifies explicit Markdown/LaTeX math
islands, math environments, balanced standalone math commands, and conservative
plain equations.  It never uses the gold answer or grading output.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, order=True)
class MathSpan:
    start: int
    end: int
    kind: str


MATH_ENVIRONMENTS = {
    "math", "displaymath", "equation", "equation*", "align", "align*",
    "aligned", "alignat", "alignat*", "gather", "gather*", "gathered",
    "multline", "multline*", "cases", "matrix", "pmatrix", "bmatrix",
    "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "array",
}

MATH_COMMAND_ARITY = {
    "boxed": 1, "sqrt": 1, "overline": 1, "underline": 1,
    "vec": 1, "hat": 1, "bar": 1,
    "frac": 2, "dfrac": 2, "tfrac": 2, "binom": 2,
}

PLAIN_EQUATION = re.compile(
    r"(?<![\w\\])"
    r"(?:[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?)"
    r"(?:\s*(?:==?|<=?|>=?|!=|≈|≠|≤|≥|\+|\*|×|÷|\^|/)\s*"
    r"(?:[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?|\([^()\n]{1,80}\))){1,}"
    r"(?![\w])"
)


def _escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _code_ranges(text: str) -> list[tuple[int, int]]:
    """Return fenced and inline Markdown code ranges to exclude from math parsing."""

    ranges: list[tuple[int, int]] = []
    fence = re.compile(r"(?m)^[ \t]*(```+|~~~+).*$")
    cursor = 0
    while match := fence.search(text, cursor):
        marker = match.group(1)
        close = re.compile(rf"(?m)^[ \t]*{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$")
        end_match = close.search(text, match.end())
        end = len(text) if end_match is None else end_match.end()
        ranges.append((match.start(), end))
        cursor = end

    masked = list(ranges)
    cursor = 0
    while cursor < len(text):
        if any(start <= cursor < end for start, end in ranges):
            cursor = next(end for start, end in ranges if start <= cursor < end)
            continue
        if text[cursor] != "`" or _escaped(text, cursor):
            cursor += 1
            continue
        run = 1
        while cursor + run < len(text) and text[cursor + run] == "`":
            run += 1
        close = text.find("`" * run, cursor + run)
        if close < 0:
            cursor += run
            continue
        masked.append((cursor, close + run))
        cursor = close + run
    return sorted(masked)


def _inside(ranges: Iterable[tuple[int, int]], index: int) -> bool:
    return any(start <= index < end for start, end in ranges)


def _find_closer(text: str, marker: str, start: int, *, same_line: bool) -> int | None:
    cursor = start
    while True:
        found = text.find(marker, cursor)
        if found < 0 or (same_line and "\n" in text[start:found]):
            return None
        if not _escaped(text, found):
            return found
        cursor = found + len(marker)


def _balanced_group_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if _escaped(text, index):
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _merge(spans: Iterable[MathSpan]) -> list[MathSpan]:
    ordered = sorted((span for span in spans if span.start < span.end), key=lambda s: (s.start, s.end))
    merged: list[MathSpan] = []
    for span in ordered:
        if not merged or span.start >= merged[-1].end:
            merged.append(span)
            continue
        previous = merged[-1]
        merged[-1] = MathSpan(previous.start, max(previous.end, span.end),
                                   previous.kind if previous.kind == span.kind else "merged")
    return merged


def parse_math_spans(text: str) -> list[MathSpan]:
    """Return non-overlapping character spans for every recognized math expression."""

    excluded = _code_ranges(text)
    spans: list[MathSpan] = []
    occupied: list[tuple[int, int]] = list(excluded)

    # Longer dollar markers must be consumed before inline dollars.
    cursor = 0
    while cursor < len(text):
        start = text.find("$$", cursor)
        if start < 0:
            break
        if _escaped(text, start) or _inside(occupied, start):
            cursor = start + 2
            continue
        close = _find_closer(text, "$$", start + 2, same_line=False)
        if close is None:
            cursor = start + 2
            continue
        spans.append(MathSpan(start + 2, close, "display_dollar"))
        occupied.append((start, close + 2))
        cursor = close + 2

    cursor = 0
    while cursor < len(text):
        start = text.find("$", cursor)
        if start < 0:
            break
        if _escaped(text, start) or _inside(occupied, start):
            cursor = start + 1
            continue
        if start + 1 >= len(text) or text[start + 1].isspace() or text[start + 1] == "$":
            cursor = start + 1
            continue
        close = _find_closer(text, "$", start + 1, same_line=True)
        if close is None or close == start + 1 or text[close - 1].isspace():
            cursor = start + 1
            continue
        spans.append(MathSpan(start + 1, close, "inline_dollar"))
        occupied.append((start, close + 1))
        cursor = close + 1

    for opening, closing, kind in (("\\(", "\\)", "inline_parenthesis"),
                                    ("\\[", "\\]", "display_bracket")):
        cursor = 0
        while cursor < len(text):
            start = text.find(opening, cursor)
            if start < 0:
                break
            if _escaped(text, start) or _inside(occupied, start):
                cursor = start + len(opening)
                continue
            close = _find_closer(text, closing, start + len(opening), same_line=opening == "\\(")
            if close is None:
                cursor = start + len(opening)
                continue
            spans.append(MathSpan(start + len(opening), close, kind))
            occupied.append((start, close + len(closing)))
            cursor = close + len(closing)

    environment = re.compile(r"\\begin\{([^{}]+)\}")
    for match in environment.finditer(text):
        name = match.group(1)
        if name not in MATH_ENVIRONMENTS or _inside(excluded, match.start()):
            continue
        closing = f"\\end{{{name}}}"
        close = text.find(closing, match.end())
        if close >= 0:
            spans.append(MathSpan(match.end(), close, f"environment:{name}"))

    command = re.compile(r"\\([A-Za-z]+)\s*")
    for match in command.finditer(text):
        name = match.group(1)
        arity = MATH_COMMAND_ARITY.get(name)
        if arity is None or _inside(excluded, match.start()):
            continue
        end = match.end()
        valid = True
        for _ in range(arity):
            while end < len(text) and text[end].isspace():
                end += 1
            group_end = _balanced_group_end(text, end)
            if group_end is None:
                valid = False
                break
            end = group_end
        if valid:
            spans.append(MathSpan(match.start(), end, f"command:{name}"))

    for match in PLAIN_EQUATION.finditer(text):
        if not _inside(excluded, match.start()):
            spans.append(MathSpan(match.start(), match.end(), "plain_equation"))

    return _merge(spans)


def overlapping_token_indices(
    trace: Mapping[str, Any], text: str, spans: Iterable[MathSpan]
) -> list[int]:
    """Map character spans to generated tokens by UTF-8 byte overlap."""

    if trace.get("status") != "ok":
        return []
    content_start = trace.get("content_start_byte")
    if not isinstance(content_start, int):
        return []
    byte_spans = [
        (content_start + len(text[:span.start].encode("utf-8")),
         content_start + len(text[:span.end].encode("utf-8")))
        for span in spans
    ]
    return [
        int(token["index"])
        for token in trace.get("tokens", [])
        if any(token["end_byte"] > start and token["start_byte"] < end
               for start, end in byte_spans)
    ]


def content_token_indices(trace: Mapping[str, Any]) -> list[int]:
    if trace.get("status") != "ok":
        return []
    start, end = trace.get("content_start_byte"), trace.get("content_end_byte")
    if not isinstance(start, int) or not isinstance(end, int):
        return []
    return [int(token["index"]) for token in trace.get("tokens", [])
            if token["end_byte"] > start and token["start_byte"] < end]
