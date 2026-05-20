"""Feature engineering for the IBM Telco Customer Churn dataset.

All new features are motivated by EDA findings documented in
notebooks/02_eda_churn_drivers.ipynb:
  - Contract type is the single strongest churn driver (Cramér's V ≈ 0.40).
  - Tenure shows a non-linear churn decay; cohort bins capture the shape better
    than raw tenure alone.
  - Service add-ons (security, tech support) correlate strongly with retention.
  - Electronic check holders churn at 2× the rate of automatic-payment customers.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_SERVICE_BINARY_COLS: tuple[str, ...] = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

_AUTOPAY_METHODS: frozenset[str] = frozenset(
    {"Bank transfer (automatic)", "Credit card (automatic)"}
)

_CONTRACT_RISK: dict[str, int] = {
    "Month-to-month": 2,
    "One year": 1,
    "Two year": 0,
}

_PAYMENT_RISK: dict[str, int] = {
    "Electronic check": 2,
    "Mailed check": 1,
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 0,
}

# Ordinal encoding used by tenure_bucket (preserves monotone churn signal).
_TENURE_LABELS: list[str] = ["new", "developing", "established", "loyal"]
_TENURE_BINS: list[int] = [0, 12, 24, 48, 72]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _count_services(df: pd.DataFrame) -> pd.Series:
    """Sum active service subscriptions per customer.

    Counts nine possible subscriptions: the eight binary service columns
    (where 'Yes' = active) plus InternetService (where anything other than
    'No' is active).
    """
    binary_flags = (df[list(_SERVICE_BINARY_COLS)] == "Yes").astype(int)
    has_internet = (df["InternetService"] != "No").astype(int)
    return binary_flags.sum(axis=1) + has_internet


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to *df* and return the augmented DataFrame.

    The nine new columns are appended; original columns are preserved unchanged.
    The input DataFrame is not mutated.

    Engineered columns
    ------------------
    tenure_bucket : Categorical["new" | "developing" | "established" | "loyal"]
        Ordinal cohort bin that captures the non-linear churn decay curve found
        in EDA.  Bins: 0–12 m (new), 13–24 m (developing), 25–48 m
        (established), 49–72 m (loyal).
    total_services : int
        Count of active service subscriptions (0–9).  Deeper product
        relationships correlate with retention.
    services_intensity : float
        ``total_services × 12 / (tenure + 1)`` — services adopted per year of
        tenure.  High values flag customers who signed up for many services very
        quickly, a pattern associated with promotional-offer churn.
    contract_risk_score : int [0, 1, 2]
        Ordinal contract risk: Two year=0, One year=1, Month-to-month=2.
    payment_risk : int [0, 1, 2]
        Ordinal payment risk: autopay=0, Mailed check=1, Electronic check=2.
    tenure_x_contract : int
        Interaction of tenure × contract_risk_score.  New month-to-month
        customers get a low value despite high risk; established month-to-month
        customers get a high value — both informative for the model.
    monthly_charges_per_service : float
        MonthlyCharges / (total_services + 1).  A rough price-efficiency signal;
        customers paying a lot per service have fewer bundling incentives to stay.
    has_premium_services : int [0, 1]
        1 if the customer subscribes to StreamingTV or StreamingMovies. EDA
        shows premium streaming customers overlap heavily with high-churn fiber
        cohort.
    autopay_flag : int [0, 1]
        1 for automatic payment methods (bank transfer or credit card).  Autopay
        customers churn at roughly half the rate of electronic-check holders.

    Args:
        df: Telco DataFrame with the original 21 columns. Must contain all
            columns referenced above.

    Returns:
        New DataFrame (copy of *df*) with 9 additional columns.

    Raises:
        KeyError: If a required source column is absent from *df*.
    """
    out = df.copy()

    # tenure_bucket — non-linear churn decay captured as an ordered Categorical
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=_TENURE_BINS,
        labels=_TENURE_LABELS,
        include_lowest=True,
        right=True,
    )

    # total_services — depth of product relationship
    n_services: pd.Series = _count_services(out)
    out["total_services"] = n_services

    # services_intensity — rapid service adoption signals promo-churn risk
    out["services_intensity"] = (n_services * 12.0) / (out["tenure"] + 1)

    # contract_risk_score — ordinal contract commitment level (high = riskier)
    out["contract_risk_score"] = out["Contract"].map(_CONTRACT_RISK)

    # payment_risk — ordinal payment friction (high = more likely to churn)
    out["payment_risk"] = out["PaymentMethod"].map(_PAYMENT_RISK)

    # tenure_x_contract — interaction term for tree ensembles that struggle
    # with multiplicative interactions at shallow depth
    out["tenure_x_contract"] = out["tenure"] * out["contract_risk_score"]

    # monthly_charges_per_service — per-subscription price signal
    out["monthly_charges_per_service"] = out["MonthlyCharges"] / (n_services + 1)

    # has_premium_services — streaming subscriber flag (high-churn fiber overlap)
    out["has_premium_services"] = (
        (out["StreamingTV"] == "Yes") | (out["StreamingMovies"] == "Yes")
    ).astype(int)

    # autopay_flag — automatic payment → lower churn risk
    out["autopay_flag"] = out["PaymentMethod"].isin(_AUTOPAY_METHODS).astype(int)

    return out
