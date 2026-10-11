"""Selector features for a BigCodeBench model pair (feature contract code_pair_scalar_v2).

Computed offline from each model's archived solo record (the task text, the complete output and
its token trace), its same-forward native trace and its code_solve_scalar_v1 evidence, before any
grade is read. The tests, the canonical solution, the grades and the problem id never enter a
feature, and no program is executed: everything below is static, from Python's own parser.

The groups mirror scripts/selector_dev1000.py for the math domain, plus what only code has:

    evidence    the 17 registered code_solve_scalar_v1 scalars
    spans       token statistics inside the final fenced code and in the prose around it
                (math: inside math expressions and inside the boxed answer)
    structure   size, nesting, branching and call counts of the final program
    task        agreement between the program and what the task text asks for: required
                libraries imported, and the exceptions the task says the function should raise
    hygiene     names used but never bound, unseeded randomness, prints, a main guard, file I/O
    text        output length and self-correction words, as in math
    cross       whether the two programs are the same, how similar, and differences of scalars

v2 replaced v1 on 2026-09-27 before any selector run (v1 lacked the task, hygiene and extra
structure groups).
"""

from __future__ import annotations

import ast
import builtins
import difflib
import math
import re
from typing import Any, Mapping

import numpy as np

from authority_allocation.code_solve_features import FEATURES as EVIDENCE_FEATURES
from authority_allocation.math_expression_spans import content_token_indices

FEATURE_CONTRACT = "code_pair_scalar_v2"
LOW = math.log(0.5)
BLOCK = re.compile(r"```[ \t]*(?:python3?|py)[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
RAISE_CLAUSE = re.compile(r"should raise the exception for:(.*?)(?:The function should output|"
                          r"You should write|\Z)", re.IGNORECASE | re.DOTALL)
EXCEPTION_NAME = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception))\b")
SELF_CORRECT = ("wait", "however", "actually")
SIMILARITY_CHARS = 20000

STAT_NAMES = ("lp_mean", "lp_min", "lp_p10", "lp_std", "ent_mean", "ent_p90", "ent_max",
              "margin_mean", "margin_p10", "lowconf_frac", "lowconf_run")
SPAN_FEATURES = tuple(f"{span}_{s}" for span in ("code", "prose") for s in STAT_NAMES)
STRUCTURE = ("code_lines", "code_chars", "ast_parses", "defines_entry_point", "n_functions",
             "n_imports", "n_loops", "n_try", "n_raise", "n_returns", "ast_nodes", "max_depth",
             "n_branches", "n_calls", "n_distinct_calls", "n_library_calls")
TASK = ("required_lib_share", "raises_when_asked", "asked_exception_share")
HYGIENE = ("undefined_names", "unseeded_random", "n_prints", "has_main_guard", "n_file_io")
TEXT = ("n_output_chars", "n_selfcorrect")
MODEL_FEATURES = tuple(EVIDENCE_FEATURES) + SPAN_FEATURES + STRUCTURE + TASK + HYGIENE + TEXT
PROBLEM_FEATURES = ("n_required_libs", "task_asks_raise", "n_asked_exceptions")
DIFF_KEYS = ("out_lp_mean", "out_lp_min", "out_ent_mean", "out_margin_mean", "out_lowconf_frac",
             "code_lp_mean", "code_lp_min", "code_ent_mean", "code_margin_mean", "code_lowconf_frac",
             "output_token_count", "code_lines", "ast_nodes", "n_branches", "undefined_names")
CROSS_FEATURES = (("program_identical", "ast_identical", "program_similarity", "import_jaccard",
                   "call_jaccard", "raise_type_jaccard", "xor_ast_parses", "xor_truncated",
                   "xor_defines_entry_point", "xor_raises_when_asked")
                  + tuple(f"diff_{k}" for k in DIFF_KEYS))
BUILTIN_NAMES = frozenset(dir(builtins)) | {"__name__", "__file__", "__doc__"}


# ------------------------------------------------------------------ token spans

def span_stats(prefix: str, positions: list[Mapping[str, Any]]) -> dict[str, float | None]:
    """selector_dev1000._stats over one token span; all None when the span is empty."""
    names = [f"{prefix}_{s}" for s in STAT_NAMES]
    if not positions:
        return dict.fromkeys(names)
    lp = np.array([p["sampled_logprob"] for p in positions], dtype=float)
    ent = np.array([p["entropy"] for p in positions], dtype=float)
    mar = np.array([p["top1_top2_margin"] for p in positions], dtype=float)
    low = lp < LOW
    run = best = 0
    for flag in low:
        run = run + 1 if flag else 0
        best = max(best, run)
    values = [lp.mean(), lp.min(), np.percentile(lp, 10), lp.std(), ent.mean(), np.percentile(ent, 90),
              ent.max(), mar.mean(), np.percentile(mar, 10), low.mean(), best]
    return {n: float(v) for n, v in zip(names, values)}


def block_char_spans(text: str) -> list[tuple[int, int]]:
    return [m.span(1) for m in BLOCK.finditer(text or "")]


def final_program(text: str) -> str | None:
    spans = block_char_spans(text)
    return text[spans[-1][0]:spans[-1][1]] if spans else None


def token_split(text: str, trace: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """(code, prose) content-token indices: a token is code when it overlaps any fenced block body."""
    content = content_token_indices(trace)
    if not content:
        return [], []
    offset = int(trace.get("content_start_byte") or 0)
    byte_spans = [(offset + len(text[:a].encode("utf-8")), offset + len(text[:b].encode("utf-8")))
                  for a, b in block_char_spans(text)]
    tokens = {int(t["index"]): t for t in trace.get("tokens", [])}
    code, prose = [], []
    for i in content:
        t = tokens[i]
        inside = any(t["end_byte"] > a and t["start_byte"] < b for a, b in byte_spans)
        (code if inside else prose).append(i)
    return code, prose


def span_features(text: str, trace: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, float | None]:
    by_index = {int(p["index"]): p for p in native.get("positions", [])}
    f: dict[str, float | None] = {}
    for prefix, idx in zip(("code", "prose"), token_split(text, trace)):
        pos = [by_index[i] for i in idx if i in by_index]
        f.update(span_stats(prefix, pos if len(pos) == len(idx) else []))
    return f


# ------------------------------------------------------------------ the task text

def parse_libs(libs: str) -> set[str]:
    try:
        value = ast.literal_eval(libs or "[]")
    except (ValueError, SyntaxError):
        return set()
    return {str(x).split(".")[0] for x in value} if isinstance(value, (list, tuple, set)) else set()


def asked_exceptions(task: str) -> tuple[bool, set[str]]:
    """Whether the task says the function should raise, and the exception names it lists."""
    match = RAISE_CLAUSE.search(task or "")
    if match is None:
        return False, set()
    return True, set(EXCEPTION_NAME.findall(match.group(1)))


def problem_features(record: Mapping[str, Any]) -> dict[str, float]:
    asks, names = asked_exceptions(record["problem_text"])
    return {"n_required_libs": float(len(parse_libs(record["problem_metadata"]["libs"]))),
            "task_asks_raise": float(asks), "n_asked_exceptions": float(len(names))}


# ------------------------------------------------------------------ the program

def dotted(node: ast.AST) -> str | None:
    """'a.b.c' for a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + max((depth(c) for c in children), default=0)


def bound_names(tree: ast.AST) -> set[str]:
    """Every name the module binds anywhere, ignoring scope (so undefined counts are conservative)."""
    bound: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
        elif isinstance(n, ast.alias) and n.name == "*":
            bound.add("*")
        elif hasattr(ast, "MatchAs") and isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            bound.add(n.name)
    return bound


def program_features(program: str | None, entry_point: str, required: set[str],
                     asked: tuple[bool, set[str]]) -> tuple[dict[str, float | None], dict[str, Any]]:
    f: dict[str, float | None] = dict.fromkeys(STRUCTURE + TASK + HYGIENE)
    extra: dict[str, Any] = {"ast_dump": None, "imports": set(), "calls": set(), "raise_types": set()}
    asks, asked_names = asked
    if program is None:
        f.update({"code_lines": 0.0, "code_chars": 0.0, "ast_parses": 0.0, "defines_entry_point": 0.0})
        return f, extra
    f["code_lines"] = float(sum(1 for line in program.splitlines() if line.strip()))
    f["code_chars"] = float(len(program))
    try:
        tree = ast.parse(program)
        max_depth = depth(tree)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        f.update({"ast_parses": 0.0, "defines_entry_point": 0.0})
        return f, extra
    nodes = list(ast.walk(tree))
    functions = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    imports: set[str] = set()
    for n in nodes:
        if isinstance(n, ast.Import):
            imports.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imports.add(n.module.split(".")[0])
    aliases = {(a.asname or a.name).split(".")[0]: a.name.split(".")[0]
               for n in nodes if isinstance(n, ast.Import) for a in n.names}
    aliases.update({(a.asname or a.name): n.module.split(".")[0]
                    for n in nodes if isinstance(n, ast.ImportFrom) and n.module for a in n.names})
    calls = [c for c in nodes if isinstance(c, ast.Call)]
    call_names = {name for name in (dotted(c.func) for c in calls) if name}
    library_calls = [name for name in (dotted(c.func) for c in calls) if name and name.split(".")[0] in aliases]
    raise_types = set()
    for n in nodes:
        if isinstance(n, ast.Raise) and n.exc is not None:
            target = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
            name = dotted(target)
            if name:
                raise_types.add(name.split(".")[-1])
    loads = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    bound = bound_names(tree)
    undefined = set() if "*" in bound else loads - bound - BUILTIN_NAMES
    random_calls = [name for name in call_names if name.split(".")[0] in aliases
                    and (aliases[name.split(".")[0]] == "random"
                         or (aliases[name.split(".")[0]] == "numpy" and ".random." in f".{name}."))]
    seeded = any(name.endswith(("seed", "default_rng", "RandomState")) for name in call_names)
    main_guard = any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                     and dotted(n.test.left) == "__name__" for n in tree.body)
    f.update({
        "ast_parses": 1.0,
        "defines_entry_point": float(any(n.name == entry_point for n in functions)),
        "n_functions": float(len(functions)),
        "n_imports": float(sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)),
        "n_loops": float(sum(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in nodes)),
        "n_try": float(sum(isinstance(n, ast.Try) for n in nodes)),
        "n_raise": float(sum(isinstance(n, ast.Raise) for n in nodes)),
        "n_returns": float(sum(isinstance(n, ast.Return) for n in nodes)),
        "ast_nodes": float(len(nodes)),
        "max_depth": float(max_depth),
        "n_branches": float(sum(isinstance(n, (ast.If, ast.IfExp, ast.For, ast.While, ast.ExceptHandler,
                                               ast.BoolOp, ast.comprehension)) for n in nodes)),
        "n_calls": float(len(calls)),
        "n_distinct_calls": float(len(call_names)),
        "n_library_calls": float(len(library_calls)),
        "required_lib_share": (len(required & imports) / len(required)) if required else None,
        "raises_when_asked": float(bool(raise_types)) if asks else None,
        "asked_exception_share": (len(asked_names & raise_types) / len(asked_names)) if asked_names else None,
        "undefined_names": float(len(undefined)),
        "unseeded_random": float(bool(random_calls) and not seeded),
        "n_prints": float(sum(1 for name in (dotted(c.func) for c in calls) if name == "print")),
        "has_main_guard": float(main_guard),
        "n_file_io": float(sum(1 for name in (dotted(c.func) for c in calls) if name == "open")),
    })
    extra.update({"ast_dump": ast.dump(tree, annotate_fields=False), "imports": imports,
                  "calls": call_names, "raise_types": raise_types})
    return f, extra


# ------------------------------------------------------------------ one model, and the pair

def model_features(record: Mapping[str, Any], native: Mapping[str, Any],
                   evidence: Mapping[str, Any]) -> tuple[dict[str, float | None], dict[str, Any]]:
    """One model's features and the extras the cross features need."""
    out = record["outputs"][0]
    text = out.get("raw_output") or ""
    trace = out.get("token_logprobs") or {}
    f: dict[str, float | None] = {name: evidence["features"][name] for name in EVIDENCE_FEATURES}
    f.update(span_features(text, trace, native))
    program = final_program(text)
    required = parse_libs(record["problem_metadata"]["libs"])
    prog, extra = program_features(program, record["problem_metadata"]["entry_point"], required,
                                   asked_exceptions(record["problem_text"]))
    f.update({name: prog[name] for name in STRUCTURE + TASK + HYGIENE})
    lower = text.lower()
    f["n_output_chars"] = float(len(text))
    f["n_selfcorrect"] = float(sum(lower.count(w) for w in SELF_CORRECT))
    if tuple(f) != MODEL_FEATURES:
        raise AssertionError("model feature order drifted from the contract")
    extra["program"] = program
    return f, extra


def normalized(program: str | None) -> str | None:
    return None if program is None else re.sub(r"\s+", " ", program).strip()


def jaccard(a: set, b: set) -> float | None:
    return len(a & b) / len(a | b) if a | b else None


def cross_features(fg: Mapping[str, Any], fv: Mapping[str, Any], eg: Mapping[str, Any],
                   ev: Mapping[str, Any]) -> dict[str, float | None]:
    pg, pv = normalized(eg["program"]), normalized(ev["program"])
    both = pg is not None and pv is not None
    xor = lambda key: None if fg[key] is None or fv[key] is None else float(fg[key] != fv[key])  # noqa: E731
    c: dict[str, float | None] = {
        "program_identical": float(pg == pv) if both else None,
        "ast_identical": (float(eg["ast_dump"] == ev["ast_dump"])
                          if eg["ast_dump"] is not None and ev["ast_dump"] is not None else None),
        "program_similarity": (difflib.SequenceMatcher(None, pg[:SIMILARITY_CHARS], pv[:SIMILARITY_CHARS],
                                                       autojunk=False).ratio() if both else None),
        "import_jaccard": jaccard(eg["imports"], ev["imports"]),
        "call_jaccard": jaccard(eg["calls"], ev["calls"]),
        "raise_type_jaccard": jaccard(eg["raise_types"], ev["raise_types"]),
        "xor_ast_parses": xor("ast_parses"),
        "xor_truncated": xor("truncated"),
        "xor_defines_entry_point": xor("defines_entry_point"),
        "xor_raises_when_asked": xor("raises_when_asked"),
    }
    for key in DIFF_KEYS:
        g, v = fg[key], fv[key]
        c[f"diff_{key}"] = None if g is None or v is None else float(v) - float(g)
    if tuple(c) != CROSS_FEATURES:
        raise AssertionError("cross feature order drifted from the contract")
    return c
