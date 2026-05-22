"""Imbalance-handling strategy comparison: SMOTE, class weights, and no correction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

_STRATEGY_LABELS = {
    "none": "No balancing",
    "class_weight": "class_weight='balanced'",
    "smote": "SMOTE",
}


def _build_estimators(seed: int) -> dict[str, Any]:
    """Return unfitted estimator for each imbalance strategy."""
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        smote_estimator: Any = ImbPipeline([
            ("smote", SMOTE(random_state=seed, k_neighbors=5)),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)),
        ])
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE comparison. "
            "Install it with: pip install imbalanced-learn"
        ) from exc

    return {
        "none": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed),
        "class_weight": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=seed
        ),
        "smote": smote_estimator,
    }


def _cv_metrics_for(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> dict[str, float]:
    """Run stratified CV and return mean metrics dict."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, f1s, recalls, precs, avg_precs = [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = clone(estimator)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, proba))
        f1s.append(f1_score(y_val, preds, zero_division=0))
        recalls.append(recall_score(y_val, preds, zero_division=0))
        precs.append(precision_score(y_val, preds, zero_division=0))
        avg_precs.append(average_precision_score(y_val, proba))

    return {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "cv_recall_mean": float(np.mean(recalls)),
        "cv_recall_std": float(np.std(recalls)),
        "cv_precision_mean": float(np.mean(precs)),
        "cv_precision_std": float(np.std(precs)),
        "cv_avg_precision_mean": float(np.mean(avg_precs)),
        "cv_avg_precision_std": float(np.std(avg_precs)),
    }


def _fit_and_proba(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Any, np.ndarray, dict[str, float]]:
    """Fit on X_train, return (model, val_probas, val_metrics)."""
    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "val_auc": float(roc_auc_score(y_val, proba)),
        "val_f1": float(f1_score(y_val, preds, zero_division=0)),
        "val_recall": float(recall_score(y_val, preds, zero_division=0)),
        "val_precision": float(precision_score(y_val, preds, zero_division=0)),
        "val_avg_precision": float(average_precision_score(y_val, proba)),
    }
    return estimator, proba, metrics


def compare_imbalance_strategies(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 42,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compare three imbalance strategies on a Logistic Regression base model.

    The three strategies are:
    - ``none``: no class-imbalance correction.
    - ``class_weight``: ``class_weight='balanced'`` re-weights samples by
      inverse class frequency, the simplest correction.
    - ``smote``: SMOTE over-samples the minority class inside each CV fold via
      an imblearn Pipeline to avoid data leakage.

    Logistic Regression is used as the base model because the effect of
    imbalance handling is most transparent on linear classifiers.  The
    resulting comparison informs which strategy to carry forward for the
    final GBM models.

    Args:
        X_train: Preprocessed training features (already scaled).
        y_train: Binary training labels.
        X_val: Preprocessed validation features.
        y_val: Binary validation labels.
        seed: Random seed for SMOTE, CV folds, and LR.
        n_splits: Number of stratified CV folds.

    Returns:
        A tuple of:
        - ``comparison``: DataFrame with one row per strategy and columns for
          CV and held-out validation metrics.
        - ``val_probas``: dict mapping strategy key to predicted probabilities
          on X_val (useful for plotting PR curves without re-fitting).
    """
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])

    estimators = _build_estimators(seed)
    rows: list[dict[str, Any]] = []
    val_probas: dict[str, np.ndarray] = {}

    for key, estimator in estimators.items():
        logger.info("Strategy: %s", _STRATEGY_LABELS[key])

        cv_m = _cv_metrics_for(estimator, X_cv, y_cv, n_splits=n_splits, seed=seed)

        fitted, proba, val_m = _fit_and_proba(
            clone(estimator), X_train, y_train, X_val, y_val
        )
        val_probas[key] = proba

        row = {"strategy": _STRATEGY_LABELS[key], **cv_m, **val_m}
        rows.append(row)

        logger.info(
            "  CV AUC=%.4f±%.4f  Recall=%.4f±%.4f  Val AUC=%.4f  Val Recall=%.4f",
            cv_m["cv_auc_mean"], cv_m["cv_auc_std"],
            cv_m["cv_recall_mean"], cv_m["cv_recall_std"],
            val_m["val_auc"], val_m["val_recall"],
        )

    comparison = pd.DataFrame(rows)
    return comparison, val_probas
