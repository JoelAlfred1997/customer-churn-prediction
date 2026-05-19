"""Pydantic schema for the raw IBM Telco Customer Churn dataset."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

# ---------------------------------------------------------------------------
# Allowed categorical values (sourced from IBM dataset documentation)
# ---------------------------------------------------------------------------

BINARY_YES_NO: frozenset[str] = frozenset({"Yes", "No"})
GENDER_VALUES: frozenset[str] = frozenset({"Male", "Female"})
MULTIPLE_LINES_VALUES: frozenset[str] = frozenset({"Yes", "No", "No phone service"})
INTERNET_SERVICE_VALUES: frozenset[str] = frozenset({"DSL", "Fiber optic", "No"})
INTERNET_ADDON_VALUES: frozenset[str] = frozenset({"Yes", "No", "No internet service"})
CONTRACT_VALUES: frozenset[str] = frozenset({"Month-to-month", "One year", "Two year"})
PAYMENT_METHOD_VALUES: frozenset[str] = frozenset(
    {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    }
)
CHURN_VALUES: frozenset[str] = frozenset({"Yes", "No"})

EXPECTED_COLUMNS: tuple[str, ...] = (
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
    "TotalCharges",
    "Churn",
)


class TelcoCustomerRaw(BaseModel):
    """Single-row schema for the raw Telco churn CSV.

    TotalCharges is typed as str because ~11 rows in the source file contain
    blank strings (new customers with zero tenure).  Coercion to float happens
    downstream in the preprocessing step.
    """

    model_config = ConfigDict(strict=False, populate_by_name=True)

    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str
    Churn: str

    @field_validator("gender")
    @classmethod
    def _gender(cls, v: str) -> str:
        if v not in GENDER_VALUES:
            raise ValueError(f"gender must be one of {GENDER_VALUES!r}, got {v!r}")
        return v

    @field_validator("SeniorCitizen")
    @classmethod
    def _senior_citizen(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"SeniorCitizen must be 0 or 1, got {v!r}")
        return v

    @field_validator("Contract")
    @classmethod
    def _contract(cls, v: str) -> str:
        if v not in CONTRACT_VALUES:
            raise ValueError(f"Contract must be one of {CONTRACT_VALUES!r}, got {v!r}")
        return v

    @field_validator("Churn")
    @classmethod
    def _churn(cls, v: str) -> str:
        if v not in CHURN_VALUES:
            raise ValueError(f"Churn must be one of {CHURN_VALUES!r}, got {v!r}")
        return v
