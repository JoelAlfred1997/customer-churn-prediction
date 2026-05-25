"""Model performance summary: metrics table, ROC, PR, and calibration plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from app.utils import available_models, load_app_data, load_model
from src.evaluation.metrics import classification_report_extended
from src.evaluation.threshold import cost_curve

_PALETTE = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]


def _metrics_table(
    models: list[tuple[str, str]],
    y_test: np.ndarray,
    X_test: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for label, stem in models:
        model = load_model(stem)
        if model is None:
            continue
        proba = model.predict_proba(X_test)[:, 1]
        row = classification_report_extended(
            y_test, proba, threshold=threshold, model_name=label
        )
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    df = df.drop(columns=["threshold"])
    return df.round(4)


def _plot_roc(models: list[tuple[str, str]], y_test: np.ndarray, X_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (label, stem) in enumerate(models):
        model = load_model(stem)
        if model is None:
            continue
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=_PALETTE[i % len(_PALETTE)], lw=2,
                label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_pr(models: list[tuple[str, str]], y_test: np.ndarray, X_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    baseline = float(np.mean(y_test))
    ax.axhline(baseline, color="#6b7280", linestyle="--", lw=1,
               label=f"No-skill baseline ({baseline:.3f})")
    for i, (label, stem) in enumerate(models):
        model = load_model(stem)
        if model is None:
            continue
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, color=_PALETTE[i % len(_PALETTE)], lw=2,
                label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_calibration(
    models: list[tuple[str, str]], y_test: np.ndarray, X_test: np.ndarray
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for i, (label, stem) in enumerate(models):
        model = load_model(stem)
        if model is None:
            continue
        proba = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", color=_PALETTE[i % len(_PALETTE)],
                lw=1.5, markersize=5, label=label)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration (Reliability Diagram)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_cost_curve(
    models: list[tuple[str, str]],
    y_test: np.ndarray,
    X_test: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (label, stem) in enumerate(models):
        model = load_model(stem)
        if model is None:
            continue
        proba = model.predict_proba(X_test)[:, 1]
        df = cost_curve(y_test, proba, cost_fn=cost_fn, cost_fp=cost_fp)
        best_idx = int(df["expected_cost"].idxmin())
        best_t = float(df.loc[best_idx, "threshold"])
        best_cost = float(df.loc[best_idx, "expected_cost"])
        ax.plot(df["threshold"], df["expected_cost"],
                color=_PALETTE[i % len(_PALETTE)], lw=2, label=label)
        ax.axvline(best_t, color=_PALETTE[i % len(_PALETTE)], lw=1, linestyle=":",
                   alpha=0.7)
        ax.scatter([best_t], [best_cost], color=_PALETTE[i % len(_PALETTE)],
                   zorder=5, s=60)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel(f"Expected Cost ($)")
    ax.set_title(f"Cost Curve  (FN=${cost_fn:,.0f}, FP=${cost_fp:,.0f})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render() -> None:
    """Render the model performance summary tab."""
    app_data = load_app_data()

    if app_data is None:
        st.error(
            "Raw dataset not found. Place the Telco CSV at `data/raw/` "
            "and restart the dashboard."
        )
        return

    models = available_models()
    if not models:
        st.warning(
            "No trained models found in `models/`. "
            "Run `python scripts/train_gbm.py` to train and save the models."
        )
        return

    y_test = app_data["y_test"]
    X_test = app_data["X_test"]

    # ── Threshold and cost controls ─────────────────────────────────────────
    with st.expander("Cost & threshold settings", expanded=False):
        ccol1, ccol2, ccol3 = st.columns(3)
        threshold = ccol1.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01, key="perf_thr")
        cost_fn = ccol2.number_input("FN cost ($)", 100, 5000, 500, 50, key="perf_fn")
        cost_fp = ccol3.number_input("FP cost ($)", 10, 500, 50, 10, key="perf_fp")

    # ── Metrics table ────────────────────────────────────────────────────────
    st.markdown("### Test-set Metrics")
    st.caption(f"All metrics computed on the held-out test set at threshold = {threshold:.2f}.")

    metrics_df = _metrics_table(models, y_test, X_test, threshold)

    def _colour_metric(val: float) -> str:
        return "color: #22c55e" if val >= 0.80 else "color: #f97316" if val >= 0.60 else ""

    styled = metrics_df.style.format("{:.4f}").map(
        _colour_metric,
        subset=["roc_auc", "pr_auc", "f1", "recall", "precision"],
    )
    st.dataframe(styled, use_container_width=True)

    st.divider()

    # ── Curve plots ──────────────────────────────────────────────────────────
    st.markdown("### Evaluation Curves")
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        _plot_roc(models, y_test, X_test)
    with curve_col2:
        _plot_pr(models, y_test, X_test)

    cal_col, cost_col = st.columns(2)
    with cal_col:
        _plot_calibration(models, y_test, X_test)
    with cost_col:
        _plot_cost_curve(models, y_test, X_test, float(cost_fn), float(cost_fp))

    # ── Interpretation guidance ──────────────────────────────────────────────
    st.divider()
    st.markdown("### Interpretation notes")
    st.markdown(
        """
- **ROC-AUC** measures rank-ordering ability independent of threshold.
- **PR-AUC** (average precision) is more informative for imbalanced classes (~26 % churn).
- **Calibration** shows whether predicted probabilities match observed churn rates.
  Points above the diagonal = under-confident; below = over-confident.
- **Cost curve** shows expected business cost at each threshold given the FN/FP cost ratio.
  Dots mark the cost-minimising threshold for each model.
"""
    )
