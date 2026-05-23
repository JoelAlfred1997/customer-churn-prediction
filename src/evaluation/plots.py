"""Evaluation plots: ROC, PR, calibration, lift/gains, confusion matrix, DCA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils.plotting import PALETTE, save_fig, set_plot_style

# Type alias for a result dict passed to multi-model plot functions.
# Required keys: "name" (str), "y_true" (array-like), "y_proba" (array-like).
ModelResult = dict[str, Any]


def plot_roc(
    results: list[ModelResult],
    *,
    ax: plt.Axes | None = None,
    title: str = "ROC Curves",
    figures_dir: Path | str | None = None,
) -> plt.Axes:
    """Plot ROC curves for one or more models on a single axes.

    Args:
        results: List of dicts with keys ``name``, ``y_true``, ``y_proba``.
        ax: Axes to draw on; creates a new figure if None.
        title: Plot title.
        figures_dir: If provided, save the figure as ``roc_curves.png``.

    Returns:
        The axes used.
    """
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        auc_val = roc_auc_score(res["y_true"], res["y_proba"])
        ax.plot(
            fpr, tpr,
            color=PALETTE[i % len(PALETTE)],
            lw=1.8,
            label=f"{res['name']} (AUC={auc_val:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)

    if figures_dir is not None:
        save_fig(fig, "roc_curves", figures_dir)

    return ax


def plot_pr(
    results: list[ModelResult],
    *,
    ax: plt.Axes | None = None,
    title: str = "Precision-Recall Curves",
    figures_dir: Path | str | None = None,
) -> plt.Axes:
    """Plot Precision-Recall curves for one or more models.

    Args:
        results: List of dicts with keys ``name``, ``y_true``, ``y_proba``.
        ax: Axes to draw on; creates a new figure if None.
        title: Plot title.
        figures_dir: If provided, save the figure as ``pr_curves.png``.

    Returns:
        The axes used.
    """
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    for i, res in enumerate(results):
        precision, recall, _ = precision_recall_curve(res["y_true"], res["y_proba"])
        ap_val = average_precision_score(res["y_true"], res["y_proba"])
        ax.plot(
            recall, precision,
            color=PALETTE[i % len(PALETTE)],
            lw=1.8,
            label=f"{res['name']} (AP={ap_val:.3f})",
        )

    if results:
        baseline = float(np.mean(results[0]["y_true"]))
        ax.axhline(
            baseline, color="k", linestyle="--", lw=1,
            label=f"No-skill baseline ({baseline:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)

    if figures_dir is not None:
        save_fig(fig, "pr_curves", figures_dir)

    return ax


def plot_calibration(
    results: list[ModelResult],
    *,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    title: str = "Calibration (Reliability Diagram)",
    figures_dir: Path | str | None = None,
) -> plt.Axes:
    """Plot reliability diagrams to assess probability calibration.

    A well-calibrated model has points close to the diagonal. Points above
    the diagonal indicate under-confidence; points below indicate
    over-confidence.

    Args:
        results: List of dicts with keys ``name``, ``y_true``, ``y_proba``.
        n_bins: Number of equally-spaced bins for the calibration curve.
        ax: Axes to draw on; creates a new figure if None.
        title: Plot title.
        figures_dir: If provided, save the figure as ``calibration.png``.

    Returns:
        The axes used.
    """
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for i, res in enumerate(results):
        frac_pos, mean_pred = calibration_curve(
            res["y_true"], res["y_proba"], n_bins=n_bins, strategy="uniform"
        )
        ax.plot(
            mean_pred, frac_pos, "s-",
            color=PALETTE[i % len(PALETTE)],
            lw=1.5, markersize=5,
            label=res["name"],
        )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)

    if figures_dir is not None:
        save_fig(fig, "calibration", figures_dir)

    return ax


def plot_lift_gains(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    model_name: str = "model",
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    figures_dir: Path | str | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot cumulative gains and lift charts for a single model.

    The gains chart shows the cumulative percentage of churners captured as
    the percentage of customers targeted increases. The lift chart is the
    ratio of the gains curve to the random baseline.

    Args:
        y_true: True binary labels (1 = churn).
        y_proba: Predicted churn probabilities.
        model_name: Legend label.
        axes: Pair ``(gains_ax, lift_ax)``; creates a new figure if None.
        figures_dir: If provided, save the figure as ``lift_gains.png``.

    Returns:
        Tuple of ``(gains_ax, lift_ax)``.
    """
    set_plot_style()
    if axes is None:
        fig, (ax_gains, ax_lift) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        ax_gains, ax_lift = axes
        fig = ax_gains.get_figure()

    y_true_arr = np.asarray(y_true)
    order = np.argsort(y_proba)[::-1]
    y_sorted = y_true_arr[order]

    n = len(y_sorted)
    total_pos = int(y_sorted.sum())

    pct_pop = np.arange(1, n + 1) / n
    cum_gains = np.cumsum(y_sorted) / total_pos
    lift = cum_gains / pct_pop

    ax_gains.plot(pct_pop * 100, cum_gains * 100, color=PALETTE[0], lw=2, label=model_name)
    ax_gains.plot([0, 100], [0, 100], "k--", lw=1, label="Random")
    ax_gains.set_xlabel("% of Customers Targeted")
    ax_gains.set_ylabel("% of Churners Captured")
    ax_gains.set_title("Cumulative Gains")
    ax_gains.legend(frameon=False)

    ax_lift.plot(pct_pop * 100, lift, color=PALETTE[0], lw=2, label=model_name)
    ax_lift.axhline(1.0, color="k", linestyle="--", lw=1, label="Random (lift=1)")
    ax_lift.set_xlabel("% of Customers Targeted")
    ax_lift.set_ylabel("Lift")
    ax_lift.set_title("Lift Chart")
    ax_lift.legend(frameon=False)

    if figures_dir is not None:
        save_fig(fig, "lift_gains", figures_dir)

    return ax_gains, ax_lift


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    model_name: str = "model",
    threshold: float = 0.5,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
    ax: plt.Axes | None = None,
    figures_dir: Path | str | None = None,
) -> plt.Axes:
    """Plot a confusion matrix annotated with expected business cost.

    Args:
        y_true: True binary labels.
        y_proba: Predicted churn probabilities.
        model_name: Model identifier for the title.
        threshold: Decision threshold for hard predictions.
        cost_fn: Per-customer cost of a false negative (missed churner).
        cost_fp: Per-customer cost of a false positive (unnecessary offer).
        ax: Axes to draw on; creates a new figure if None.
        figures_dir: If provided, save the figure as
            ``confusion_matrix_<model_name>.png``.

    Returns:
        The axes used.
    """
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    expected_cost = fn * cost_fn + fp * cost_fp

    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"{model_name}  [threshold={threshold:.2f}]\n"
        f"Expected cost: ${expected_cost:,.0f}  "
        f"(FN×${int(cost_fn):,} + FP×${int(cost_fp):,})"
    )

    if figures_dir is not None:
        slug = model_name.lower().replace(" ", "_")
        save_fig(fig, f"confusion_matrix_{slug}", figures_dir)

    return ax


def plot_decision_curve(
    results: list[ModelResult],
    *,
    ax: plt.Axes | None = None,
    title: str = "Decision Curve Analysis",
    figures_dir: Path | str | None = None,
) -> plt.Axes:
    """Plot decision curves (net benefit vs. threshold) for one or more models.

    Decision curve analysis (DCA) evaluates a model's clinical or business
    utility across a range of threshold probabilities. At threshold *t*:

        NB(t) = TP/N  −  FP/N × t / (1 − t)

    The "treat all" strategy (flag every customer for retention) and the
    "treat none" strategy (flag nobody) serve as benchmarks.

    Args:
        results: List of dicts with keys ``name``, ``y_true``, ``y_proba``.
        ax: Axes to draw on; creates a new figure if None.
        title: Plot title.
        figures_dir: If provided, save the figure as ``decision_curve.png``.

    Returns:
        The axes used.
    """
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    thresholds = np.linspace(0.01, 0.99, 200)

    if results:
        y_true_ref = np.asarray(results[0]["y_true"])
        n = len(y_true_ref)
        prevalence = y_true_ref.mean()

        # "Treat all" net benefit
        nb_treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        ax.plot(
            thresholds, nb_treat_all,
            color="#6B7280", lw=1.5, linestyle="-.",
            label="Treat all",
        )
        # "Treat none" is always NB=0
        ax.axhline(0.0, color="#6B7280", lw=1.2, linestyle=":", label="Treat none")

    for i, res in enumerate(results):
        y_true = np.asarray(res["y_true"])
        y_proba = np.asarray(res["y_proba"])
        n = len(y_true)
        nb_vals = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            nb = tp / n - fp / n * t / (1 - t)
            nb_vals.append(nb)

        ax.plot(
            thresholds, nb_vals,
            color=PALETTE[i % len(PALETTE)],
            lw=2,
            label=res["name"],
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, None)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.legend(frameon=False)

    if figures_dir is not None:
        save_fig(fig, "decision_curve", figures_dir)

    return ax
