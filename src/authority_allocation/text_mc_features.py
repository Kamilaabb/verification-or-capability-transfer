"""Scalar evidence for the MMLU-Pro knowledge study (feature contract text_mc_scalar_v1).

vision_mc_scalar_v1 without its two image features: the same output statistics, computed by
the same code, so the vision and knowledge domains describe an output identically.
"""

from __future__ import annotations

from typing import Any, Mapping

from authority_allocation import vision_mc_features

FEATURE_CONTRACT = "text_mc_scalar_v1"
IMAGE_FEATURES = ("n_images", "num_visual_tokens")
FEATURES = tuple(name for name in vision_mc_features.FEATURES if name not in IMAGE_FEATURES)


def derive_features(*, output: Mapping[str, Any], native: Mapping[str, Any], n_options: int,
                    question_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    features, audit = vision_mc_features.derive_features(
        output=output, native=native, n_options=n_options, n_images=0,
        question_text=question_text, num_visual_tokens=None)
    features = {name: features[name] for name in FEATURES}
    return features, audit
