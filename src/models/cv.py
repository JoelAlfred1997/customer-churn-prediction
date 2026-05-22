"""Stratified k-fold cross-validation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def stratified_cv_score(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run stratified k-fold CV and return per-fold metrics.

    Args:
        estimator: Unfitted sklearn-compatible estimator.
        X: Feature matrix.
        y: Binary target vector.
        n_splits: Number of CV folds.
        seed: Random seed for fold generation.
        threshold: Decision threshold for F1/recall/precision metrics.

    Returns:
        DataFrame with columns fold, roc_auc, f1, recall, precision,
        average_precision — one row per fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = clone(estimator)
        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_val)[:, 1]
        preds = (proba >= threshold).astype(int)

        rows.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y_val, proba)),
            "f1": float(f1_score(y_val, preds, zero_division=0)),
            "recall": float(recall_score(y_val, preds, zero_division=0)),
            "precision": float(precision_score(y_val, preds, zero_division=0)),
            "average_precision": float(average_precision_score(y_val, proba)),
        })

    return pd.DataFrame(rows)


def cv_summary(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-fold CV results as mean ± std.

    Args:
        cv_df: DataFrame returned by :func:`stratified_cv_score`.

    Returns:
        DataFrame with columns metric, mean, std — one row per metric.
    """
    metric_cols = [c for c in cv_df.columns if c != "fold"]
    rows = [
        {
            "metric": col,
            "mean": float(cv_df[col].mean()),
            "std": float(cv_df[col].std()),
        }
        for col in metric_cols
    ]
    return pd.DataFrame(rows)
