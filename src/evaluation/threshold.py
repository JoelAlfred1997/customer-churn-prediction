"""Cost-sensitive threshold optimisation for churn prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cost_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """Compute expected business cost across a grid of decision thresholds.

    At each threshold *t*:

        expected_cost = FN(t) × cost_fn + FP(t) × cost_fp

    where FN is the number of churners missed and FP is the number of
    non-churners incorrectly flagged for a retention offer.

    Args:
        y_true: True binary labels (1 = churn, 0 = no churn).
        y_proba: Predicted churn probabilities from a trained model.
        cost_fn: Business cost of a false negative (missed churner).
            Defaults to 500 (estimated lost customer lifetime value).
        cost_fp: Business cost of a false positive (unnecessary offer).
            Defaults to 50 (offer discount + handling cost).
        n_thresholds: Number of threshold candidates in [0.01, 0.99].

    Returns:
        DataFrame with columns: threshold, tp, fp, fn, tn, recall,
        precision, expected_cost, normalised_cost.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows: list[dict] = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())

        rows.append({
            "threshold": float(t),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "expected_cost": fn * cost_fn + fp * cost_fp,
        })

    df = pd.DataFrame(rows)
    max_cost = df["expected_cost"].max()
    df["normalised_cost"] = df["expected_cost"] / max_cost if max_cost > 0 else 0.0
    return df


def optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
    n_thresholds: int = 200,
) -> tuple[float, pd.DataFrame]:
    """Find the decision threshold that minimises expected business cost.

    The cost model is::

        expected_cost(t) = FN(t) × cost_fn + FP(t) × cost_fp

    A false negative (FN) represents a churner who was not offered retention
    and subsequently left — the loss is the customer's lifetime value.
    A false positive (FP) represents a loyal customer who received an
    unnecessary retention offer — the cost is the offer value and handling.

    Args:
        y_true: True binary labels (1 = churn, 0 = no churn).
        y_proba: Predicted churn probabilities from a trained model.
        cost_fn: Business cost per false negative. Default: 500.
        cost_fp: Business cost per false positive. Default: 50.
        n_thresholds: Number of threshold candidates to evaluate.

    Returns:
        A tuple of:
        - ``best_threshold``: The threshold minimising expected cost.
        - ``df``: Full cost curve DataFrame from :func:`cost_curve`.
    """
    df = cost_curve(
        y_true,
        y_proba,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        n_thresholds=n_thresholds,
    )
    best_idx = int(df["expected_cost"].idxmin())
    best_threshold = float(df.loc[best_idx, "threshold"])
    return best_threshold, df


def threshold_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    cost_fp: float = 50.0,
    fn_fp_ratios: list[float] | None = None,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """Analyse how the optimal threshold shifts as the FN/FP cost ratio varies.

    Holds ``cost_fp`` fixed and sweeps ``cost_fn = ratio × cost_fp`` across a
    range of ratios.  This reveals how sensitive the operating point is to
    uncertainty in the cost assumptions.

    Args:
        y_true: True binary labels.
        y_proba: Predicted churn probabilities.
        cost_fp: Fixed cost per false positive.
        fn_fp_ratios: FN/FP ratios to evaluate.  Defaults to a log-spaced
            range from 1 to 50.
        n_thresholds: Number of threshold candidates.

    Returns:
        DataFrame with columns: fn_fp_ratio, cost_fn, optimal_threshold,
        recall_at_opt, precision_at_opt, expected_cost.
    """
    if fn_fp_ratios is None:
        fn_fp_ratios = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

    rows: list[dict] = []
    for ratio in fn_fp_ratios:
        cost_fn = ratio * cost_fp
        best_t, df = optimal_threshold(
            y_true,
            y_proba,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            n_thresholds=n_thresholds,
        )
        best_row = df[df["threshold"] == best_t].iloc[0]
        rows.append({
            "fn_fp_ratio": ratio,
            "cost_fn": cost_fn,
            "optimal_threshold": best_t,
            "recall_at_opt": float(best_row["recall"]),
            "precision_at_opt": float(best_row["precision"]),
            "expected_cost": float(best_row["expected_cost"]),
        })

    return pd.DataFrame(rows)
