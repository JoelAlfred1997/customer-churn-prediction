"""Tests for the preprocessing pipeline and stratified split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data.preprocess import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    ORDINAL_COLS,
    build_preprocessor,
)
from src.data.split import stratified_split
from src.features.build_features import engineer_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(n: int = 4, seed: int = 0) -> pd.DataFrame:
    """Return a minimal raw Telco-schema DataFrame with *n* rows."""
    rng = np.random.default_rng(seed)
    contracts = ["Month-to-month", "One year", "Two year"]
    payments = [
        "Electronic check",
        "Bank transfer (automatic)",
        "Mailed check",
        "Credit card (automatic)",
    ]
    internet = ["Fiber optic", "DSL", "No"]
    yn = ["Yes", "No"]
    addons = ["Yes", "No", "No internet service"]

    return pd.DataFrame({
        "customerID": [f"c{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n).tolist(),
        "SeniorCitizen": rng.integers(0, 2, n).tolist(),
        "Partner": rng.choice(yn, n).tolist(),
        "Dependents": rng.choice(yn, n).tolist(),
        "tenure": rng.integers(0, 73, n).tolist(),
        "PhoneService": rng.choice(yn, n).tolist(),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n).tolist(),
        "InternetService": rng.choice(internet, n).tolist(),
        "OnlineSecurity": rng.choice(addons, n).tolist(),
        "OnlineBackup": rng.choice(addons, n).tolist(),
        "DeviceProtection": rng.choice(addons, n).tolist(),
        "TechSupport": rng.choice(addons, n).tolist(),
        "StreamingTV": rng.choice(addons, n).tolist(),
        "StreamingMovies": rng.choice(addons, n).tolist(),
        "Contract": rng.choice(contracts, n).tolist(),
        "PaperlessBilling": rng.choice(yn, n).tolist(),
        "PaymentMethod": rng.choice(payments, n).tolist(),
        "MonthlyCharges": rng.uniform(20, 100, n).round(2).tolist(),
        "TotalCharges": [str(round(v, 2)) for v in rng.uniform(0, 8000, n)],
        "Churn": rng.choice(yn, n, p=[0.27, 0.73]).tolist(),
    })


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same TotalCharges + engineer_features steps as pipeline.py."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = engineer_features(df)
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_engineered() -> pd.DataFrame:
    """20-row engineered DataFrame (no customerID, no Churn)."""
    df = _make_raw(n=20, seed=1)
    df = _engineer(df)
    return df.drop(columns=["customerID", "Churn"])


@pytest.fixture(scope="module")
def small_engineered_with_nan_tc() -> pd.DataFrame:
    """8-row engineered DataFrame with one blank TotalCharges (→ NaN after conversion)."""
    df = _make_raw(n=8, seed=2)
    # Force one blank string to trigger the NaN imputation path.
    df.at[0, "TotalCharges"] = "  "
    df = _engineer(df)
    return df.drop(columns=["customerID", "Churn"])


@pytest.fixture(scope="module")
def large_raw() -> pd.DataFrame:
    """150-row raw DataFrame for split tests — enough for stable stratification."""
    return _make_raw(n=150, seed=42)


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------


def test_returns_column_transformer() -> None:
    assert isinstance(build_preprocessor(), ColumnTransformer)


def test_column_lists_no_overlap() -> None:
    all_cols = NUMERIC_COLS + CATEGORICAL_COLS + ORDINAL_COLS
    assert len(all_cols) == len(set(all_cols)), "Duplicate columns in preprocessing lists."


def test_output_no_nan(small_engineered: pd.DataFrame) -> None:
    X = build_preprocessor().fit_transform(small_engineered)
    assert not np.isnan(X).any()


def test_output_shape(small_engineered: pd.DataFrame) -> None:
    X = build_preprocessor().fit_transform(small_engineered)
    assert X.shape[0] == len(small_engineered)
    # Output must have at least the numeric + ordinal columns.
    assert X.shape[1] >= len(NUMERIC_COLS) + len(ORDINAL_COLS)


def test_output_is_dense(small_engineered: pd.DataFrame) -> None:
    X = build_preprocessor().fit_transform(small_engineered)
    assert isinstance(X, np.ndarray)


def test_reproducible(small_engineered: pd.DataFrame) -> None:
    X1 = build_preprocessor().fit_transform(small_engineered)
    X2 = build_preprocessor().fit_transform(small_engineered)
    np.testing.assert_array_equal(X1, X2)


def test_nan_total_charges_imputed(small_engineered_with_nan_tc: pd.DataFrame) -> None:
    """NaN introduced by blank TotalCharges must be imputed, not propagated."""
    df = small_engineered_with_nan_tc
    assert df["TotalCharges"].isna().any(), "Fixture must have at least one NaN TotalCharges."
    X = build_preprocessor().fit_transform(df)
    assert not np.isnan(X).any()


def test_transform_consistent_shape(small_engineered: pd.DataFrame) -> None:
    """Transform on unseen data must have same n_columns as fit."""
    ct = build_preprocessor()
    X_fit = ct.fit_transform(small_engineered)
    X_new = ct.transform(small_engineered.iloc[:5])
    assert X_new.shape[1] == X_fit.shape[1]


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------


def test_split_sizes(large_raw: pd.DataFrame) -> None:
    train, val, test = stratified_split(
        large_raw, target="Churn", val_size=0.1, test_size=0.2, seed=42
    )
    total = len(large_raw)
    assert len(train) + len(val) + len(test) == total
    assert abs(len(test) - round(total * 0.2)) <= 2
    assert abs(len(val) - round(total * 0.1)) <= 2


def test_split_no_overlap(large_raw: pd.DataFrame) -> None:
    train, val, test = stratified_split(
        large_raw, target="Churn", val_size=0.1, test_size=0.2, seed=42
    )
    assert set(train.index).isdisjoint(val.index), "train and val share rows."
    assert set(train.index).isdisjoint(test.index), "train and test share rows."
    assert set(val.index).isdisjoint(test.index), "val and test share rows."


def test_split_target_balance(large_raw: pd.DataFrame) -> None:
    train, val, test = stratified_split(
        large_raw, target="Churn", val_size=0.1, test_size=0.2, seed=42
    )
    overall = (large_raw["Churn"] == "Yes").mean()
    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        rate = (split["Churn"] == "Yes").mean()
        assert abs(rate - overall) < 0.15, (
            f"{name} churn rate {rate:.2%} deviates >15 pp from overall {overall:.2%}"
        )


def test_split_seed_reproducible(large_raw: pd.DataFrame) -> None:
    train_a, val_a, test_a = stratified_split(large_raw, "Churn", seed=7)
    train_b, val_b, test_b = stratified_split(large_raw, "Churn", seed=7)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_split_invalid_fractions() -> None:
    df = pd.DataFrame({"x": range(10), "Churn": ["Yes", "No"] * 5})
    with pytest.raises(ValueError):
        stratified_split(df, "Churn", val_size=0.5, test_size=0.6)


def test_split_fraction_out_of_range() -> None:
    df = pd.DataFrame({"x": range(10), "Churn": ["Yes", "No"] * 5})
    with pytest.raises(ValueError):
        stratified_split(df, "Churn", val_size=0.0, test_size=0.2)


# ---------------------------------------------------------------------------
# No-leakage test
# ---------------------------------------------------------------------------


def test_no_leakage_scaler_fit_on_train_only(large_raw: pd.DataFrame) -> None:
    """StandardScaler mean must match training data, not the full dataset."""
    df = _engineer(large_raw)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])

    train, val, _ = stratified_split(df, "Churn", val_size=0.1, test_size=0.2, seed=42)
    y_train = train.pop("Churn")  # noqa: F841
    _y_val = val.pop("Churn")

    ct = build_preprocessor()
    ct.fit(train)

    scaler = ct.named_transformers_["num"]["scaler"]
    tenure_idx = NUMERIC_COLS.index("tenure")
    scaler_mean = scaler.mean_[tenure_idx]
    train_mean = train["tenure"].mean()

    assert abs(scaler_mean - train_mean) < 1e-6, (
        f"Scaler mean {scaler_mean:.4f} != training mean {train_mean:.4f} — "
        "preprocessor may have been fit on more than the training set."
    )
