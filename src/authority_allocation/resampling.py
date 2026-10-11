"""Percentiles, paired bootstrap and usage roll-ups.

The bootstrap is paired because every policy is evaluated on the same items; an
unpaired interval would overstate the uncertainty of a difference.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Iterable, Mapping


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    index = (len(ordered) - 1) * proportion
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_difference(
    left: list[int], right: list[int], *, seed: int, repetitions: int
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap needs equal non-empty arrays")
    rng = random.Random(seed)
    n = len(left)
    samples: list[float] = []
    for _ in range(repetitions):
        difference = 0
        for _ in range(n):
            index = rng.randrange(n)
            difference += left[index] - right[index]
        samples.append(difference / n)
    return {
        "difference": statistics.fmean(left) - statistics.fmean(right),
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
    }


def summarize_usage(usages: Iterable[Mapping[str, Any] | None]) -> dict[str, float | int]:
    rows = [usage for usage in usages if usage is not None]
    return {
        "calls": len(rows),
        "input_tokens": sum((usage.get("input_tokens") or 0) for usage in rows),
        "output_tokens": sum((usage.get("output_tokens") or 0) for usage in rows),
        "reasoning_tokens": sum((usage.get("reasoning_tokens") or 0) for usage in rows),
        "latency_seconds": sum((usage.get("latency_ms") or 0) for usage in rows) / 1000,
    }
