"""End-to-end data preparation pipeline: load → engineer → split → preprocess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data.load import load_raw
from src.data.preprocess import build_preprocessor
from src.data.split import stratified_split
from src.features.build_features import engineer_features


class PreparedData(TypedDict):
    """Dict returned by :func:`prepare_data`."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list[str]


def prepare_data(
    raw_path: Path | str,
    *,
    target: str = "Churn",
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    processed_dir: Path | str | None = None,
) -> PreparedData:
    """Load, engineer, split, and preprocess the Telco churn dataset.

    The preprocessor is fit exclusively on training data to prevent leakage.
    Optionally persists the processed arrays and feature name list to
    *processed_dir* as ``splits.npz`` and ``feature_names.json``.

    Args:
        raw_path: Path to the raw Telco CSV file.
        target: Name of the target column (binary Yes/No string).
        val_size: Fraction of total rows for the validation set.
        test_size: Fraction of total rows for the test set.
        seed: Random seed for reproducibility.
        processed_dir: If provided, processed arrays are saved here.

    Returns:
        :class:`PreparedData` dict with arrays, series, preprocessor, and
        feature names.
    """
    df = load_raw(raw_path)

    # Convert blank-string TotalCharges to float; NaN rows are handled by the
    # median imputer inside the preprocessing pipeline.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = engineer_features(df)

    # OrdinalEncoder requires string input; pandas Categorical must be cast.
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)

    # Binary-encode the target before splitting so stratification is on ints.
    y_binary = (df[target] == "Yes").astype(int).rename(target)
    df = df.drop(columns=["customerID", target])
    df_with_target = df.assign(**{target: y_binary})

    train_df, val_df, test_df = stratified_split(
        df_with_target,
        target=target,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    y_train: pd.Series = train_df.pop(target)
    y_val: pd.Series = val_df.pop(target)
    y_test: pd.Series = test_df.pop(target)

    preprocessor = build_preprocessor()
    X_train: np.ndarray = preprocessor.fit_transform(train_df)
    X_val: np.ndarray = preprocessor.transform(val_df)
    X_test: np.ndarray = preprocessor.transform(test_df)

    feature_names: list[str] = preprocessor.get_feature_names_out().tolist()

    if processed_dir is not None:
        out_dir = Path(processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "splits.npz",
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train.to_numpy(),
            y_val=y_val.to_numpy(),
            y_test=y_test.to_numpy(),
        )
        (out_dir / "feature_names.json").write_text(
            json.dumps(feature_names, indent=2)
        )

    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
