"""Cost of a model call for RQ3/RQ4 (2026-10-04).

One call costs parameters x (input + output tokens): the forward-pass compute up to the constant 2,
ignoring attention and KV-cache reuse. Input and output tokens weigh the same in the main analysis;
the appendix repeats it with output tokens weighted 4x. Costs are in billion-parameter tokens.

The selector only reads the two independent answers, so a selector system costs exactly the two solo
calls. JEV is a closed API priced in dollars on input tokens only; its tokens are never added to the
open models' costs.
"""
from __future__ import annotations

from typing import Mapping

# Total parameters in billions, from each model card; checked on the server against the weight files.
PARAMS_B = {
    "Qwen3-4B": 4.0,
    "Qwen3-8B": 8.2,
    "Qwen3-32B": 32.8,
    "Qwen2.5-3B-Instruct": 3.09,
    "Qwen2.5-7B-Instruct": 7.61,
    "Qwen2.5-72B-Instruct": 72.7,
    "Qwen2.5-Coder-7B-Instruct": 7.61,
    "Seed-Coder-8B-Instruct": 8.25,
    "Llama-3.2-3B-Instruct": 3.21,
    "Llama-3.1-8B-Instruct": 8.03,
    "Ministral-8B-Instruct-2410": 8.02,
    "Mistral-Small-24B-Instruct-2501": 23.6,
}
JEV_USD_PER_M_INPUT = 0.042  # output tokens are free
OUTPUT_WEIGHTS = {"main": 1.0, "output_x4": 4.0}

# Systems of one pair (S = stronger, W = weaker by development accuracy) and the calls each one makes.
SYSTEMS = ("weaker_alone", "stronger_alone", "review_by_stronger", "review_by_weaker",
           "selector_primary", "selector_secondary", "JEV", "oracle")
# A review's first call is the answer V actually read, taken from the review's own generator archive.
CALLS = {
    "weaker_alone": ("solo_w",),
    "stronger_alone": ("solo_s",),
    "review_by_stronger": ("g_of_review_by_s", "review_by_s"),  # W answers, S reviews
    "review_by_weaker": ("g_of_review_by_w", "review_by_w"),    # S answers, W reviews
    "selector_primary": ("solo_w", "solo_s"),
    "selector_secondary": ("solo_w", "solo_s"),
    "JEV": ("solo_w", "solo_s"),                      # plus the API, reported separately in dollars
    "oracle": ("solo_w", "solo_s"),
}
CALL_MODEL = {"solo_w": "w", "solo_s": "s", "g_of_review_by_s": "w", "review_by_s": "s",
              "g_of_review_by_w": "s", "review_by_w": "w"}


def usage_of(record: Mapping) -> tuple[int, int]:
    """(input, output) tokens of the y0 output of one archived record; raises if either is missing."""
    outputs = record.get("outputs")
    usage = (outputs[0].get("usage") if outputs else None) or record.get("usage") or {}
    pair = usage.get("input_tokens"), usage.get("output_tokens")
    if any(type(x) is not int or x < 0 for x in pair):
        raise ValueError(f"record {record.get('problem_id')!r}: no integer input/output token counts")
    return pair


def call_cost(params_b: float, tokens: tuple[int, int], output_weight: float = 1.0) -> float:
    return params_b * (tokens[0] + output_weight * tokens[1])


def system_cost(row: Mapping, system: str, params: Mapping[str, float], output_weight: float = 1.0) -> float:
    """Cost of one system on one question; `row` maps each call in CALL_MODEL to [input, output] tokens."""
    return sum(call_cost(params[CALL_MODEL[c]], tuple(row[c]), output_weight) for c in CALLS[system])
