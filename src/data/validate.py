"""Validate a raw Telco churn DataFrame against the expected schema."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.schema import (
    BINARY_YES_NO,
    CHURN_VALUES,
    CONTRACT_VALUES,
    EXPECTED_COLUMNS,
    GENDER_VALUES,
    INTERNET_ADDON_VALUES,
    INTERNET_SERVICE_VALUES,
    MULTIPLE_LINES_VALUES,
    PAYMENT_METHOD_VALUES,
)

# Map each categorical column to its allowed value set.
_ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "gender": GENDER_VALUES,
    "Partner": BINARY_YES_NO,
    "Dependents": BINARY_YES_NO,
    "PhoneService": BINARY_YES_NO,
    "MultipleLines": MULTIPLE_LINES_VALUES,
    "InternetService": INTERNET_SERVICE_VALUES,
    "OnlineSecurity": INTERNET_ADDON_VALUES,
    "OnlineBackup": INTERNET_ADDON_VALUES,
    "DeviceProtection": INTERNET_ADDON_VALUES,
    "TechSupport": INTERNET_ADDON_VALUES,
    "StreamingTV": INTERNET_ADDON_VALUES,
    "StreamingMovies": INTERNET_ADDON_VALUES,
    "Contract": CONTRACT_VALUES,
    "PaperlessBilling": BINARY_YES_NO,
    "PaymentMethod": PAYMENT_METHOD_VALUES,
    "Churn": CHURN_VALUES,
}

# Columns that must not contain nulls in the raw data.
_NON_NULLABLE: tuple[str, ...] = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "Churn",
)


@dataclass
class ValidationResult:
    """Holds errors and warnings from a single validation run."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def raise_on_error(self) -> None:
        """Raise ValueError if any errors were found."""
        if not self.is_valid:
            raise ValueError("Dataset validation failed:\n" + str(self))

    def __str__(self) -> str:
        lines = [f"ERROR:   {e}" for e in self.errors]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines) if lines else "OK — no issues found."


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Check schema, dtypes, nulls, ranges, and categoricals of raw Telco data.

    Designed to be called immediately after ``load_raw()``.  Errors indicate
    data that will cause downstream failures; warnings flag known quirks that
    are handled explicitly in preprocessing.

    Args:
        df: DataFrame returned by ``load_raw()``.

    Returns:
        :class:`ValidationResult` with ``.is_valid``, ``.errors``, and
        ``.warnings`` populated.
    """
    result = ValidationResult()

    # ------------------------------------------------------------------
    # 1. Column presence
    # ------------------------------------------------------------------
    expected_cols = set(EXPECTED_COLUMNS)
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    if missing:
        result.errors.append(f"Missing columns: {sorted(missing)}")
    if extra:
        result.warnings.append(f"Unexpected extra columns (will be ignored): {sorted(extra)}")

    if missing:
        # Further checks are meaningless if required columns are absent.
        return result

    # ------------------------------------------------------------------
    # 2. Row count sanity
    # ------------------------------------------------------------------
    if len(df) < 100:
        result.warnings.append(
            f"Only {len(df)} rows present — full dataset should have ~7,043."
        )

    # ------------------------------------------------------------------
    # 3. Duplicate primary keys
    # ------------------------------------------------------------------
    n_dupes = int(df["customerID"].duplicated().sum())
    if n_dupes > 0:
        result.errors.append(f"Found {n_dupes} duplicate customerID values.")

    # ------------------------------------------------------------------
    # 4. Null checks
    # ------------------------------------------------------------------
    for col in _NON_NULLABLE:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            result.errors.append(f"'{col}' has {n_null} null value(s).")

    # TotalCharges: blank strings are expected for zero-tenure customers.
    n_blank_tc = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
    if n_blank_tc > 0:
        result.warnings.append(
            f"TotalCharges has {n_blank_tc} blank-string row(s) "
            "(expected for new customers with tenure=0)."
        )

    # ------------------------------------------------------------------
    # 5. Dtype checks
    # ------------------------------------------------------------------
    for col in ("tenure", "SeniorCitizen"):
        if not pd.api.types.is_integer_dtype(df[col]):
            result.errors.append(
                f"'{col}' expected integer dtype, got {df[col].dtype}."
            )

    if not pd.api.types.is_float_dtype(df["MonthlyCharges"]):
        result.errors.append(
            f"'MonthlyCharges' expected float dtype, got {df['MonthlyCharges'].dtype}."
        )

    # ------------------------------------------------------------------
    # 6. Value range checks
    # ------------------------------------------------------------------
    invalid_sc = df.loc[~df["SeniorCitizen"].isin((0, 1)), "SeniorCitizen"].unique()
    if len(invalid_sc) > 0:
        result.errors.append(f"SeniorCitizen contains values outside {{0,1}}: {list(invalid_sc)}.")

    if int((df["tenure"] < 0).sum()) > 0:
        result.errors.append(f"tenure has {int((df['tenure'] < 0).sum())} negative value(s).")

    if int((df["MonthlyCharges"] < 0).sum()) > 0:
        result.errors.append(
            f"MonthlyCharges has {int((df['MonthlyCharges'] < 0).sum())} negative value(s)."
        )

    # ------------------------------------------------------------------
    # 7. Categorical allowed-value checks
    # ------------------------------------------------------------------
    for col, allowed in _ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        actual_vals = set(df[col].dropna().unique())
        unexpected = actual_vals - allowed
        if unexpected:
            result.errors.append(
                f"'{col}' contains unexpected value(s): {sorted(str(v) for v in unexpected)}."
            )

    return result
