"""Low-capacity, leakage-safe training primitives for c008."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


CAPS = ("M0", "M1", "M2")


def explicit_missing_matrix(
    rows: Sequence[Mapping[str, float | int | None]], feature_names: Sequence[str]
) -> np.ndarray:
    """Return values followed by one explicit missing flag per registered feature."""

    values = np.asarray(
        [
            [np.nan if row[name] is None else float(row[name]) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )
    return np.concatenate((values, np.isnan(values).astype(float)), axis=1)


def make_pipeline(c_value: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=False, keep_empty_features=True
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=None,
                    fit_intercept=True,
                ),
            ),
        ]
    )


def choose_action(
    probabilities: Mapping[str, float],
    menu: Iterable[str],
    *,
    lower_authority_tolerance: float = 0.0,
) -> str:
    """Choose the lowest-authority cap within tolerance of the best probability."""

    ordered = tuple(cap for cap in CAPS if cap in set(menu))
    if not ordered:
        raise ValueError("menu must contain at least one registered cap")
    best = max(float(probabilities[cap]) for cap in ordered)
    return next(
        cap
        for cap in ordered
        if best - float(probabilities[cap]) <= lower_authority_tolerance
    )


def select_c(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    c_grid: Sequence[float],
    folds: int,
    seed: int,
) -> float:
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[tuple[float, float]] = []
    for c_value in c_grid:
        fold_scores: list[float] = []
        for train, valid in splitter.split(x, groups=groups):
            heads: list[float] = []
            for head in range(3):
                model = make_pipeline(float(c_value))
                model.fit(x[train], y[train, head])
                predicted = model.predict_proba(x[valid])[:, 1]
                heads.append(float(log_loss(y[valid, head], predicted, labels=[0, 1])))
            fold_scores.append(float(np.mean(heads)))
        scores.append((float(np.mean(fold_scores)), float(c_value)))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


__all__ = [
    "CAPS",
    "choose_action",
    "explicit_missing_matrix",
    "make_pipeline",
    "select_c",
]
