"""Cohort-level churn risk analysis with interactive segment filters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.utils import available_models, load_app_data, load_model

_RISK_THRESHOLD = 0.50


def _get_predictions(model_name: str, X_test: np.ndarray) -> np.ndarray:
    """Return churn probabilities for the test set."""
    model = load_model(model_name)
    if model is None:
        return np.zeros(len(X_test))
    return model.predict_proba(X_test)[:, 1]


def _segment_bar(df: pd.DataFrame, col: str, title: str) -> None:
    """Plot average churn probability per category in *col*."""
    grouped = (
        df.groupby(col)["churn_prob"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    grouped.columns = [col, "avg_churn_prob", "n_customers"]

    fig, ax = plt.subplots(figsize=(7, max(3, len(grouped) * 0.55)))
    colors = [
        "#ef4444" if v >= 0.50 else "#f97316" if v >= 0.30 else "#22c55e"
        for v in grouped["avg_churn_prob"]
    ]
    bars = ax.barh(grouped[col], grouped["avg_churn_prob"], color=colors)
    ax.bar_label(
        bars,
        labels=[f"{v:.1%}" for v in grouped["avg_churn_prob"]],
        padding=4,
        fontsize=9,
    )
    ax.axvline(0.50, color="#6b7280", linewidth=1, linestyle="--", label="50 % threshold")
    ax.set_xlabel("Average Churn Probability")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render(model_name: str = "xgb_v1") -> None:
    """Render the cohort risk analysis tab.

    Args:
        model_name: File stem of the pickled model to use for scoring.
    """
    app_data = load_app_data()

    if app_data is None:
        st.error(
            "Raw dataset not found. Place the Telco CSV at `data/raw/` "
            "and restart the dashboard."
        )
        return

    if not available_models():
        st.warning(
            "No trained models found in `models/`. "
            "Run `python scripts/train_gbm.py` to train the models."
        )
        return

    test_df = app_data["test_df_raw"].copy()
    X_test = app_data["X_test"]
    y_test = app_data["y_test"]

    test_df["churn_prob"] = _get_predictions(model_name, X_test)
    test_df["churned"] = y_test
    test_df["high_risk"] = test_df["churn_prob"] >= _RISK_THRESHOLD

    # ── Filters ─────────────────────────────────────────────────────────────
    st.markdown("### Segment Filters")
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)

    with fcol1:
        contracts = ["All"] + sorted(test_df["Contract"].dropna().unique().tolist())
        contract_sel = st.selectbox("Contract", contracts, key="coh_contract")
    with fcol2:
        inet_types = ["All"] + sorted(test_df["InternetService"].dropna().unique().tolist())
        inet_sel = st.selectbox("Internet Service", inet_types, key="coh_inet")
    with fcol3:
        pay_methods = ["All"] + sorted(test_df["PaymentMethod"].dropna().unique().tolist())
        pay_sel = st.selectbox("Payment Method", pay_methods, key="coh_pay")
    with fcol4:
        buckets = ["All"] + ["new", "developing", "established", "loyal"]
        bucket_sel = st.selectbox("Tenure Bucket", buckets, key="coh_bucket")

    mask = pd.Series(True, index=test_df.index)
    if contract_sel != "All":
        mask &= test_df["Contract"] == contract_sel
    if inet_sel != "All":
        mask &= test_df["InternetService"] == inet_sel
    if pay_sel != "All":
        mask &= test_df["PaymentMethod"] == pay_sel
    if bucket_sel != "All":
        mask &= test_df["tenure_bucket"] == bucket_sel

    df_filt = test_df[mask].copy()

    if df_filt.empty:
        st.info("No customers match the selected filters.")
        return

    # ── KPIs ────────────────────────────────────────────────────────────────
    st.divider()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Customers in segment", f"{len(df_filt):,}")
    kpi2.metric("High-risk (≥ 50 %)", f"{df_filt['high_risk'].sum():,}")
    kpi3.metric("Avg churn probability", f"{df_filt['churn_prob'].mean():.1%}")
    kpi4.metric("Actual churn rate", f"{df_filt['churned'].mean():.1%}")

    st.divider()

    # ── Risk distribution ───────────────────────────────────────────────────
    col_hist, col_seg = st.columns(2)

    with col_hist:
        st.markdown("#### Churn probability distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            df_filt["churn_prob"],
            bins=20,
            range=(0, 1),
            color="#3b82f6",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(
            _RISK_THRESHOLD,
            color="#ef4444",
            linewidth=1.5,
            linestyle="--",
            label=f"Threshold ({_RISK_THRESHOLD:.0%})",
        )
        ax.set_xlabel("Predicted Churn Probability")
        ax.set_ylabel("Number of Customers")
        ax.set_title(f"Risk Distribution  (n={len(df_filt):,})")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_seg:
        st.markdown("#### Average risk by Contract type")
        _segment_bar(df_filt, "Contract", "Churn Risk by Contract")

    st.divider()

    col_pay, col_inet = st.columns(2)
    with col_pay:
        st.markdown("#### Average risk by Payment Method")
        _segment_bar(df_filt, "PaymentMethod", "Churn Risk by Payment Method")
    with col_inet:
        st.markdown("#### Average risk by Internet Service")
        _segment_bar(df_filt, "InternetService", "Churn Risk by Internet Service")

    # ── High-risk customer table ─────────────────────────────────────────────
    st.divider()
    st.markdown("#### Highest-risk customers in segment")

    display_cols = [
        "tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "churn_prob",
        "churned",
    ]
    top_n = st.slider("Show top N at-risk customers", 5, 50, 20, key="top_n")
    top_df = (
        df_filt[display_cols]
        .sort_values("churn_prob", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_df.index += 1
    top_df["churn_prob"] = top_df["churn_prob"].map("{:.1%}".format)
    top_df["churned"] = top_df["churned"].map({1: "Yes", 0: "No"})
    top_df = top_df.rename(
        columns={
            "churn_prob": "Predicted Risk",
            "churned": "Actually Churned",
            "MonthlyCharges": "Monthly ($)",
        }
    )
    st.dataframe(top_df, use_container_width=True)
