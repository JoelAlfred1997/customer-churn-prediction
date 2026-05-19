"""Data ingestion and validation."""

from src.data.load import download_telco_data, load_raw
from src.data.validate import ValidationResult, validate_dataframe

__all__ = [
    "download_telco_data",
    "load_raw",
    "validate_dataframe",
    "ValidationResult",
]
