"""Single-customer churn prediction with SHAP waterfall explanation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from app.utils import get_explainer, load_app_data, load_model, preprocess_customer
from src.evaluation.interpretability import get_shap_values

_INTERNET_SERVICES = ["No", "DSL", "Fiber optic"]
_PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
_CONTRACTS = ["Month-to-month", "One year", "Two year"]


def _customer_form() -> pd.DataFrame:
    """Render all customer input widgets and return a one-row raw-feature DataFrame."""
    col_demo, col_services, col_billing = st.columns(3)

    with col_demo:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No",
            key="senior",
        )
        partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
        dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")

    with col_services:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multiple_lines = (
            st.selectbox("Multiple Lines", ["No", "Yes"], key="mlines")
            if phone_service == "Yes"
            else "No phone service"
        )

        internet_service = st.selectbox(
            "Internet Service", _INTERNET_SERVICES, index=2, key="internet"
        )
        no_inet = internet_service == "No"
        _inet_opts = ["No", "Yes"]
        _na_val = "No internet service"

        online_security = (
            _na_val
            if no_inet
            else st.selectbox("Online Security", _inet_opts, key="osec")
        )
        online_backup = (
            _na_val
            if no_inet
            else st.selectbox("Online Backup", _inet_opts, key="obak")
        )
        device_protection = (
            _na_val
            if no_inet
            else st.selectbox("Device Protection", _inet_opts, key="dprot")
        )
        tech_support = (
            _na_val
            if no_inet
            else st.selectbox("Tech Support", _inet_opts, key="tsup")
        )
        streaming_tv = (
            _na_val
            if no_inet
            else st.selectbox("Streaming TV", _inet_opts, key="stv")
        )
        streaming_movies = (
            _na_val
            if no_inet
            else st.selectbox("Streaming Movies", _inet_opts, key="smov")
        )

        if no_inet:
            st.caption("Internet-related services are N/A.")

    with col_billing:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract", _CONTRACTS, key="contract")
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
        payment = st.selectbox("Payment Method", _PAYMENT_METHODS, key="payment")
        monthly = st.number_input(
            "Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5, key="monthly"
        )
        total_default = float(tenure * monthly)
        total = st.number_input(
            "Total Charges ($)",
            0.0,
            10_000.0,
            total_default,
            step=1.0,
            help="Typically tenure × monthly charges. Edit if the actual value differs.",
            key="total",
        )

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": int(senior),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": str(total),
            }
        ]
    )


def _risk_badge(prob: float) -> tuple[str, str]:
    if prob >= 0.60:
        return "High Risk", "🔴"
    if prob >= 0.30:
        return "Medium Risk", "🟡"
    return "Low Risk", "🟢"


def render(model_name: str = "xgb_v1") -> None:
    """Render the single-customer prediction tab.

    Args:
        model_name: File stem of the pickled model to use (e.g. ``"xgb_v1"``).
    """
    app_data = load_app_data()
    model = load_model(model_name)

    if app_data is None:
        st.error(
            "Raw dataset not found. Place the Telco CSV at `data/raw/` "
            "and restart the dashboard."
        )
        return

    if model is None:
        st.warning(
            f"Model **{model_name}** not found in `models/`. "
            "Run `python scripts/train_gbm.py` to train and save the models."
        )
        return

    st.markdown(
        "Enter a customer profile to get a real-time churn probability and a "
        "feature-level explanation powered by SHAP."
    )

    row_df = _customer_form()

    st.divider()

    if not st.button("Predict churn risk", type="primary"):
        st.caption("Fill in the form above and click **Predict churn risk**.")
        return

    preprocessor = app_data["preprocessor"]
    feature_names = app_data["feature_names"]

    X_customer = preprocess_customer(row_df, preprocessor)
    prob = float(model.predict_proba(X_customer)[0, 1])

    label, icon = _risk_badge(prob)

    col_metric, col_bar = st.columns([1, 3])
    with col_metric:
        st.metric("Churn Probability", f"{prob:.1%}")
        st.markdown(f"**{icon} {label}**")
    with col_bar:
        st.progress(prob)

    st.subheader("Why this prediction?")
    st.caption(
        "Each bar shows how much that feature pushes the prediction above "
        "(red) or below (blue) the average churn rate across the training set."
    )

    explainer = get_explainer(model_name)
    if explainer is None:
        st.info("SHAP explainer unavailable.")
        return

    sv = get_shap_values(explainer, X_customer, feature_names=feature_names)

    plt.close("all")
    shap.plots.waterfall(sv[0], max_display=15, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
