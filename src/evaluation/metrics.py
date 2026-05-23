"""Extended classification metrics for comprehensive model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report_extended(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
    model_name: str = "model",
) -> dict[str, float | str]:
    """Compute a comprehensive set of binary classification metrics.

    Covers probability-based metrics (ROC-AUC, PR-AUC, Brier score) and
    threshold-dependent metrics (accuracy, precision, recall, F1) at a
    specified decision threshold.

    Args:
        y_true: True binary labels (1 = churn, 0 = no churn).
        y_proba: Predicted churn probabilities in [0, 1].
        threshold: Decision threshold for hard predictions. Default: 0.5.
        model_name: Identifier stored under the ``model`` key in the result.

    Returns:
        Dict with keys: model, threshold, accuracy, precision, recall,
        f1, roc_auc, pr_auc, brier_score.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def compare_models(
    model_results: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame for multiple models.

    Args:
        model_results: List of ``(model_name, y_true, y_proba)`` tuples.
        threshold: Shared decision threshold applied to all models.

    Returns:
        DataFrame indexed by model name with one metric column each.
    """
    rows = [
        classification_report_extended(
            y_true, y_proba, threshold=threshold, model_name=name
        )
        for name, y_true, y_proba in model_results
    ]
    df = pd.DataFrame(rows).set_index("model")
    numeric_cols = [c for c in df.columns if c != "threshold"]
    df[numeric_cols] = df[numeric_cols].round(4)
    return df
