"""Unit tests for src/features/build_features.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import engineer_features

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_EXPECTED_NEW_COLS = {
    "tenure_bucket",
    "total_services",
    "services_intensity",
    "contract_risk_score",
    "payment_risk",
    "tenure_x_contract",
    "monthly_charges_per_service",
    "has_premium_services",
    "autopay_flag",
}


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Four Telco-schema rows covering diverse customer profiles.

    Row 0 — high-risk: month-to-month, electronic check, fiber, new customer
    Row 1 — medium-risk: one-year, bank transfer (auto), DSL, mid-tenure
    Row 2 — low-risk: two-year, mailed check, no internet, long-tenured
    Row 3 — mixed: month-to-month, credit card (auto), DSL, zero tenure
    """
    return pd.DataFrame(
        {
            "customerID": ["a", "b", "c", "d"],
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [1, 24, 60, 0],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service", "No"],
            "InternetService": ["Fiber optic", "DSL", "No", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "No"],
            "OnlineBackup": ["No", "Yes", "No internet service", "Yes"],
            "DeviceProtection": ["No", "No", "No internet service", "No"],
            "TechSupport": ["No", "Yes", "No internet service", "No"],
            "StreamingTV": ["Yes", "No", "No internet service", "No"],
            "StreamingMovies": ["No", "No", "No internet service", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Bank transfer (automatic)",
                "Mailed check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [70.0, 55.0, 20.0, 45.0],
            "TotalCharges": ["70.0", "1320.0", "1200.0", "0.0"],
            "Churn": ["Yes", "No", "No", "No"],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_has_all_engineered_columns(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    assert _EXPECTED_NEW_COLS.issubset(result.columns)


def test_input_not_mutated(sample_df: pd.DataFrame) -> None:
    original_cols = list(sample_df.columns)
    engineer_features(sample_df)
    assert list(sample_df.columns) == original_cols
    assert _EXPECTED_NEW_COLS.isdisjoint(sample_df.columns)


def test_tenure_bucket_labels(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    buckets = result["tenure_bucket"].astype(str).tolist()
    assert buckets[0] == "new"           # tenure=1  → (0, 12]
    assert buckets[1] == "developing"    # tenure=24 → (12, 24]
    assert buckets[2] == "loyal"         # tenure=60 → (48, 72]
    assert buckets[3] == "new"           # tenure=0  → [0, 12]


def test_contract_risk_score_ordering(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    scores = result["contract_risk_score"].tolist()
    assert scores[0] == 2   # Month-to-month → highest risk
    assert scores[1] == 1   # One year → medium
    assert scores[2] == 0   # Two year → lowest risk


def test_payment_risk_ordering(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    risks = result["payment_risk"].tolist()
    assert risks[0] == 2   # Electronic check → highest risk
    assert risks[1] == 0   # Bank transfer (automatic) → no risk
    assert risks[2] == 1   # Mailed check → medium
    assert risks[3] == 0   # Credit card (automatic) → no risk


def test_autopay_flag(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    flags = result["autopay_flag"].tolist()
    assert flags[0] == 0   # Electronic check — manual
    assert flags[1] == 1   # Bank transfer (automatic)
    assert flags[2] == 0   # Mailed check — manual
    assert flags[3] == 1   # Credit card (automatic)


def test_total_services_counts(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    # Row 0: PhoneService=Yes(1), MultipleLines=No(0), InternetService=Fiber(1),
    #        OnlineSecurity=No(0), OnlineBackup=No(0), DeviceProtection=No(0),
    #        TechSupport=No(0), StreamingTV=Yes(1), StreamingMovies=No(0) → 3
    assert result["total_services"].iloc[0] == 3
    # Row 2: no phone, no internet → all zeros
    assert result["total_services"].iloc[2] == 0
    # Row 1: PhoneService=Yes(1), MultipleLines=Yes(1), InternetService=DSL(1),
    #        OnlineSecurity=Yes(1), OnlineBackup=Yes(1), TechSupport=Yes(1) → 6
    assert result["total_services"].iloc[1] == 6


def test_has_premium_services(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    assert result["has_premium_services"].iloc[0] == 1   # StreamingTV=Yes
    assert result["has_premium_services"].iloc[1] == 0   # neither streaming
    assert result["has_premium_services"].iloc[2] == 0   # no internet service
    assert result["has_premium_services"].iloc[3] == 0   # neither streaming


def test_monthly_charges_per_service_positive(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    # denominator is total_services + 1, so always ≥ 1 → always positive
    assert (result["monthly_charges_per_service"] > 0).all()


def test_tenure_x_contract_interaction(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    # Row 0: tenure=1, contract_risk=2 → 2
    assert result["tenure_x_contract"].iloc[0] == 2
    # Row 2: tenure=60, contract_risk=0 → 0
    assert result["tenure_x_contract"].iloc[2] == 0
    # Row 1: tenure=24, contract_risk=1 → 24
    assert result["tenure_x_contract"].iloc[1] == 24


def test_services_intensity_zero_tenure(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    # tenure=0: denominator is (0+1)=1, so intensity = n_services * 12
    row3 = result.iloc[3]
    expected = float(row3["total_services"]) * 12.0
    assert row3["services_intensity"] == pytest.approx(expected)
