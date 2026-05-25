"""Streamlit dashboard for the customer churn prediction system.

Run with:
    streamlit run app/streamlit_app.py
Or via the Makefile:
    make app
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the repo root importable so src.* and app.* resolve correctly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.utils import available_models  # noqa: E402
from app.components import cohort, performance, predict  # noqa: E402

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📉 Churn Dashboard")
    st.markdown(
        "Production-grade churn prediction for a telco with ~7,000 customers. "
        "Built with XGBoost, LightGBM, SHAP, and Streamlit."
    )
    st.divider()

    models = available_models()
    if models:
        model_label = st.selectbox(
            "Active model",
            options=[lbl for lbl, _ in models],
            key="sidebar_model",
        )
        model_name = next(stem for lbl, stem in models if lbl == model_label)
    else:
        st.warning("No trained models found. Run `python scripts/train_gbm.py` first.")
        model_name = "xgb_v1"
        model_label = "XGBoost"

    st.divider()
    st.markdown(
        "**Useful commands**\n"
        "```bash\n"
        "# Train models\n"
        "python scripts/train_gbm.py\n\n"
        "# Run dashboard\n"
        "make app\n"
        "```"
    )
    st.markdown("---")
    st.caption("IBM Telco Customer Churn · MIT License")

# ── Main content ─────────────────────────────────────────────────────────────
st.title("Customer Churn Prediction")
st.markdown(
    f"Active model: **{model_label}** &nbsp;|&nbsp; "
    "Dataset: IBM Telco (~7,043 customers, 26 % churn rate)"
)

tab_pred, tab_cohort, tab_perf = st.tabs(
    ["🔍 Predict", "📊 Cohort Analysis", "📈 Model Performance"]
)

with tab_pred:
    predict.render(model_name=model_name)

with tab_cohort:
    cohort.render(model_name=model_name)

with tab_perf:
    performance.render()
