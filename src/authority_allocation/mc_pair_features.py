"""Selector features for an MMLU-Pro model pair (feature contract mc_pair_scalar_v1).

Computed offline from each model's archived solo record (question, complete output, token trace),
its same-forward native trace and its text_mc_scalar_v1 evidence, before any grade is read. The
reference answer, the grades and the problem id never enter a feature. Mirrors
scripts/selector_dev1000.py (math) and authority_allocation.code_pair_features (code), plus what
only a multiple-choice answer has:

    evidence     the 22 registered text_mc_scalar_v1 scalars
    options      how the answer position's probability is spread over the option letters, read
                 from the native top-20 logits at the registered answer token
    trajectory   how many different letters the output entertained, whether its first letter was
                 its last, hedging and elimination words
    computation  digits, numbers and equals signs, separating calculation from recall
    problem      subject and source one-hot (the math selector uses subject and level)
    cross        whether the two models chose the same letter, and how each one's option
                 distribution rates the other's letter
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

import numpy as np

from authority_allocation.text_mc_features import FEATURES as EVIDENCE_FEATURES

FEATURE_CONTRACT = "mc_pair_scalar_v1"
LETTERS = "ABCDEFGHIJ"
PLAUSIBLE = 0.05
CATEGORIES = ("biology", "business", "chemistry", "computer science", "economics", "engineering",
              "health", "history", "law", "math", "other", "philosophy", "physics", "psychology")
ORIGINS = ("ori_mmlu", "stemez", "theoremQA", "scibench")
HEDGES = ("likely", "probably", "i think", "not sure", "might", "perhaps", "unsure", "approximately",
          "seems", "possibly")
ELIMINATIONS = ("incorrect", "not correct", "eliminate", "rule out", "ruled out", "cannot be", "is wrong")
SELF_CORRECT = ("wait", "however", "actually")
MENTION = re.compile(r"(?:\(([A-J])\)|\b(?:option|answer is|answer:|choice)\s*\(?([A-J])\b)", re.IGNORECASE)
ANSWER_LINE = re.compile(r"^\s*(?:\*\*)?\s*(?:final\s+)?answer\s*:", re.IGNORECASE | re.MULTILINE)

OPTION_FEATURES = ("opt_p_chosen", "opt_mass", "opt_entropy", "opt_margin", "opt_n_plausible",
                   "opt_logit_gap", "opt_chosen_is_top")
TRAJECTORY = ("n_letters_mentioned", "n_answer_lines", "first_letter_is_final", "n_hedges",
              "n_eliminations", "n_selfcorrect")
COMPUTATION = ("digit_share", "n_numbers", "n_equals")
MODEL_FEATURES = tuple(EVIDENCE_FEATURES) + OPTION_FEATURES + TRAJECTORY + COMPUTATION
PROBLEM_FEATURES = tuple(f"category_{c.replace(' ', '_')}" for c in CATEGORIES) + \
    tuple(f"origin_{o}" for o in ORIGINS)
DIFF_KEYS = ("out_lp_mean", "out_ent_mean", "out_margin_mean", "out_lowconf_frac", "ans_lp", "ans_ent",
             "ans_margin", "opt_p_chosen", "opt_entropy", "opt_margin", "output_token_count",
             "n_letters_mentioned")
CROSS_FEATURES = (("same_letter", "both_parseable", "xor_parseable", "xor_truncated", "option_js",
                   "g_letter_rank_in_v", "v_letter_rank_in_g", "v_prob_of_g_letter", "g_prob_of_v_letter")
                  + tuple(f"diff_{k}" for k in DIFF_KEYS))


# ------------------------------------------------------------------ option letters

def letter_of_token(token: str, n_options: int) -> str | None:
    """The option letter a vocabulary entry spells, allowing a leading space; else None."""
    text = token.replace("Ġ", " ").replace("▁", " ").strip()
    return text if len(text) == 1 and text in LETTERS[:n_options] else None


def letter_map(id_to_token: Mapping[int, str], n_options: int = 10) -> dict[int, str]:
    """token id -> letter for every vocabulary entry that spells an option letter."""
    out = {}
    for tid, tok in id_to_token.items():
        letter = letter_of_token(tok, n_options)
        if letter is not None:
            out[int(tid)] = letter
    return out


def option_distribution(position: Mapping[str, Any] | None, letters: Mapping[int, str],
                        n_options: int) -> dict[str, float] | None:
    """Probability of each option letter at the answer position (summed over spellings), from
    the top-20 raw logits; letters outside the top 20 get 0."""
    if position is None:
        return None
    lse = float(position["logsumexp"])
    dist: dict[str, float] = {}
    for tid, logit in zip(position["top20_token_ids"], position["top20_raw_logits"]):
        letter = letters.get(int(tid))
        if letter is not None and letter in LETTERS[:n_options]:
            dist[letter] = dist.get(letter, 0.0) + math.exp(float(logit) - lse)
    return dist


def option_features(dist: dict[str, float] | None, chosen: str | None,
                    position: Mapping[str, Any] | None, letters: Mapping[int, str]) -> dict[str, float | None]:
    f: dict[str, float | None] = dict.fromkeys(OPTION_FEATURES)
    if not dist or chosen is None:
        return f
    mass = sum(dist.values())
    norm = {k: v / mass for k, v in dist.items()}
    ranked = sorted(norm.values(), reverse=True)
    best_logit: dict[str, float] = {}
    for tid, logit in zip(position["top20_token_ids"], position["top20_raw_logits"]):
        letter = letters.get(int(tid))
        if letter is not None:
            best_logit[letter] = max(best_logit.get(letter, -math.inf), float(logit))
    others = [v for k, v in best_logit.items() if k != chosen]
    f.update({
        "opt_p_chosen": norm.get(chosen, 0.0),
        "opt_mass": mass,
        "opt_entropy": float(-sum(p * math.log(p) for p in norm.values() if p > 0)),
        "opt_margin": ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0),
        "opt_n_plausible": float(sum(p >= PLAUSIBLE for p in norm.values())),
        "opt_logit_gap": (best_logit[chosen] - max(others)) if chosen in best_logit and others else None,
        "opt_chosen_is_top": float(max(norm, key=norm.get) == chosen),
    })
    return f


# ------------------------------------------------------------------ the output text

def trajectory_features(text: str, final: str | None) -> dict[str, float | None]:
    mentions = [a or b for a, b in MENTION.findall(text or "")]
    mentions = [m.upper() for m in mentions]
    lower = (text or "").lower()
    return {
        "n_letters_mentioned": float(len(set(mentions))),
        "n_answer_lines": float(len(ANSWER_LINE.findall(text or ""))),
        "first_letter_is_final": float(mentions[0] == final) if mentions and final is not None else None,
        "n_hedges": float(sum(lower.count(w) for w in HEDGES)),
        "n_eliminations": float(sum(lower.count(w) for w in ELIMINATIONS)),
        "n_selfcorrect": float(sum(lower.count(w) for w in SELF_CORRECT)),
    }


def computation_features(text: str) -> dict[str, float]:
    text = text or ""
    return {"digit_share": (sum(ch.isdigit() for ch in text) / len(text)) if text else 0.0,
            "n_numbers": float(len(re.findall(r"\d+(?:\.\d+)?", text))),
            "n_equals": float(text.count("="))}


# ------------------------------------------------------------------ one model, the problem, the pair

def model_features(record: Mapping[str, Any], native: Mapping[str, Any], evidence: Mapping[str, Any],
                   letters: Mapping[int, str]) -> tuple[dict[str, float | None], dict[str, Any]]:
    out = record["outputs"][0]
    text = out.get("raw_output") or ""
    n_options = int(record["problem_metadata"]["n_options"])
    chosen = out.get("extracted_answer")
    f: dict[str, float | None] = {name: evidence["features"][name] for name in EVIDENCE_FEATURES}
    index = evidence["audit"].get("answer_token_index")
    by_index = {int(p["index"]): p for p in native.get("positions", [])}
    position = by_index.get(int(index)) if index is not None else None
    dist = option_distribution(position, letters, n_options)
    f.update(option_features(dist, chosen, position, letters))
    f.update(trajectory_features(text, chosen))
    f.update(computation_features(text))
    if tuple(f) != MODEL_FEATURES:
        raise AssertionError("model feature order drifted from the contract")
    norm = None
    if dist:
        mass = sum(dist.values())
        norm = {k: v / mass for k, v in dist.items()}
    return f, {"letter": chosen, "dist": norm}


def problem_features(record: Mapping[str, Any]) -> dict[str, float]:
    meta = record["problem_metadata"]
    f = {f"category_{c.replace(' ', '_')}": float(meta["category"] == c) for c in CATEGORIES}
    f.update({f"origin_{o}": float(meta["origin"] == o) for o in ORIGINS})
    return f


def rank_of(letter: str | None, dist: Mapping[str, float] | None) -> float | None:
    if letter is None or not dist:
        return None
    ordered = sorted(dist, key=dist.get, reverse=True)
    return float(ordered.index(letter) + 1) if letter in ordered else float(len(ordered) + 1)


def js_divergence(p: Mapping[str, float] | None, q: Mapping[str, float] | None) -> float | None:
    if not p or not q:
        return None
    keys = set(p) | set(q)
    a = np.array([p.get(k, 0.0) for k in keys])
    b = np.array([q.get(k, 0.0) for k in keys])
    m = (a + b) / 2
    kl = lambda x, y: float(sum(xi * math.log(xi / yi) for xi, yi in zip(x, y) if xi > 0))  # noqa: E731
    return (kl(a, m) + kl(b, m)) / 2


def cross_features(fg: Mapping[str, Any], fv: Mapping[str, Any], eg: Mapping[str, Any],
                   ev: Mapping[str, Any]) -> dict[str, float | None]:
    lg, lv = eg["letter"], ev["letter"]
    both = lg is not None and lv is not None
    c: dict[str, float | None] = {
        "same_letter": float(lg == lv) if both else None,
        "both_parseable": float(both),
        "xor_parseable": float((lg is None) != (lv is None)),
        "xor_truncated": float(fg["truncated"] != fv["truncated"]),
        "option_js": js_divergence(eg["dist"], ev["dist"]),
        "g_letter_rank_in_v": rank_of(lg, ev["dist"]),
        "v_letter_rank_in_g": rank_of(lv, eg["dist"]),
        "v_prob_of_g_letter": ev["dist"].get(lg, 0.0) if ev["dist"] and lg is not None else None,
        "g_prob_of_v_letter": eg["dist"].get(lv, 0.0) if eg["dist"] and lv is not None else None,
    }
    for key in DIFF_KEYS:
        g, v = fg[key], fv[key]
        c[f"diff_{key}"] = None if g is None or v is None else float(v) - float(g)
    if tuple(c) != CROSS_FEATURES:
        raise AssertionError("cross feature order drifted from the contract")
    return c
