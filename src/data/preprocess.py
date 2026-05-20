"""sklearn-compatible preprocessing pipeline for the Telco churn dataset."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column groups — imported by pipeline.py and test_preprocess.py
# ---------------------------------------------------------------------------

NUMERIC_COLS: list[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "total_services",
    "services_intensity",
    "contract_risk_score",
    "payment_risk",
    "tenure_x_contract",
    "monthly_charges_per_service",
    "has_premium_services",
    "autopay_flag",
]

# Low-cardinality nominal columns — all one-hot encoded.
CATEGORICAL_COLS: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Columns with a meaningful order — encoded as integers preserving rank.
ORDINAL_COLS: list[str] = ["tenure_bucket"]
ORDINAL_CATEGORIES: list[list[str]] = [["new", "developing", "established", "loyal"]]


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the Telco churn feature matrix.

    Transformations applied:
    - Numeric: median imputation (handles TotalCharges blanks) → StandardScaler.
    - Categorical: most-frequent imputation → OneHotEncoder (drop binary
      redundancy with ``drop='if_binary'``; unknown categories silently zeroed).
    - Ordinal: most-frequent imputation → OrdinalEncoder with explicit
      category ordering (new < developing < established < loyal).

    The ColumnTransformer uses ``remainder='drop'`` so customerID and any
    other non-feature columns are silently excluded if present.

    Returns:
        Unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
            ("ord", ordinal_pipe, ORDINAL_COLS),
        ],
        remainder="drop",
    )
