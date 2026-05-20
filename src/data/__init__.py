"""Data ingestion, validation, and preprocessing."""

from src.data.load import download_telco_data, load_raw
from src.data.pipeline import PreparedData, prepare_data
from src.data.preprocess import build_preprocessor
from src.data.split import stratified_split
from src.data.validate import ValidationResult, validate_dataframe

__all__ = [
    "download_telco_data",
    "load_raw",
    "validate_dataframe",
    "ValidationResult",
    "build_preprocessor",
    "stratified_split",
    "prepare_data",
    "PreparedData",
]
