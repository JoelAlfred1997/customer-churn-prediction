"""Model loading, data preparation, and caching helpers for the Streamlit dashboard."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Ensure the repo root is importable so src.* packages resolve correctly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.load import load_raw  # noqa: E402
from src.data.preprocess import build_preprocessor  # noqa: E402
from src.data.split import stratified_split  # noqa: E402
from src.evaluation.interpretability import build_explainer, clean_feature_names  # noqa: E402
from src.features.build_features import engineer_features  # noqa: E402


class AppData(TypedDict):
    """Preprocessed dataset cached for the dashboard session."""

    preprocessor: Any
    feature_names: list[str]
    X_test: np.ndarray
    y_test: np.ndarray
    test_df_raw: pd.DataFrame  # feature-engineered but not preprocessed, reset index
    X_train_sample: np.ndarray  # background sample for KernelExplainer


@st.cache_resource(show_spinner="Loading project config…")
def load_config() -> dict:
    """Load configs/config.yaml."""
    return yaml.safe_load((_REPO_ROOT / "configs" / "config.yaml").read_text())


def _raw_data_path() -> Path:
    cfg = load_config()
    return _REPO_ROOT / cfg["paths"]["raw_data"]


def _models_dir() -> Path:
    cfg = load_config()
    return _REPO_ROOT / cfg["paths"]["models"]


@st.cache_resource(show_spinner="Preparing dataset and preprocessing pipeline…")
def load_app_data() -> AppData | None:
    """Run the full data pipeline and cache the result for the session.

    Returns None when the raw CSV is absent so callers can show a friendly
    error instead of crashing.
    """
    raw_path = _raw_data_path()
    if not raw_path.exists():
        return None

    cfg = load_config()
    seed: int = cfg["random_seed"]
    target: str = cfg["data"]["target_column"]
    val_size: float = cfg["data"]["val_size"]
    test_size: float = cfg["data"]["test_size"]

    df = load_raw(raw_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = engineer_features(df)
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)

    y_binary = (df[target] == "Yes").astype(int).rename(target)
    df_features = df.drop(columns=["customerID", target])
    df_with_target = df_features.assign(**{target: y_binary})

    train_df, val_df, test_df = stratified_split(
        df_with_target,
        target=target,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    y_train = train_df.pop(target).to_numpy()
    _y_val = val_df.pop(target).to_numpy()
    y_test = test_df.pop(target).to_numpy()

    preprocessor = build_preprocessor()
    X_train: np.ndarray = preprocessor.fit_transform(train_df)
    preprocessor.transform(val_df)
    X_test: np.ndarray = preprocessor.transform(test_df)

    raw_names: list[str] = preprocessor.get_feature_names_out().tolist()
    feature_names = clean_feature_names(raw_names)

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)

    return AppData(
        preprocessor=preprocessor,
        feature_names=feature_names,
        X_test=X_test,
        y_test=y_test,
        test_df_raw=test_df.reset_index(drop=True),
        X_train_sample=X_train[bg_idx],
    )


@st.cache_resource(show_spinner="Loading model…")
def load_model(name: str) -> Any | None:
    """Load a pickled model by filename stem; return None if not found."""
    path = _models_dir() / f"{name}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


@st.cache_resource(show_spinner="Building SHAP explainer…")
def get_explainer(model_name: str) -> Any | None:
    """Build and cache a SHAP TreeExplainer for *model_name*."""
    model = load_model(model_name)
    if model is None:
        return None
    return build_explainer(model)


def available_models() -> list[tuple[str, str]]:
    """Return ``(display_name, file_stem)`` pairs for every trained model on disk."""
    candidates = [("XGBoost", "xgb_v1"), ("LightGBM", "lgbm_v1")]
    return [
        (label, stem)
        for label, stem in candidates
        if (_models_dir() / f"{stem}.pkl").exists()
    ]


def preprocess_customer(row: pd.DataFrame, preprocessor: Any) -> np.ndarray:
    """Apply feature engineering and preprocessing to a single-row raw DataFrame.

    Args:
        row: One-row DataFrame with the 19 original Telco columns (no customerID, no Churn).
        preprocessor: Fitted ColumnTransformer from :func:`load_app_data`.

    Returns:
        Preprocessed feature array of shape ``(1, n_features)``.
    """
    df = row.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = engineer_features(df)
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)
    return preprocessor.transform(df)
