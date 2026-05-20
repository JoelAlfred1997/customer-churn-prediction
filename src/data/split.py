"""Stratified train / validation / test split for the Telco churn dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    target: str,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train, validation, and test sets preserving class balance.

    The test set is carved first (``test_size`` fraction of total rows).
    The validation set is then carved from the remainder so that its size
    relative to the full dataset equals ``val_size``.

    Args:
        df: DataFrame that includes the target column.
        target: Name of the binary target column used for stratification.
        val_size: Fraction of *total* rows reserved for the validation set.
        test_size: Fraction of *total* rows reserved for the test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_df, val_df, test_df)``.

    Raises:
        ValueError: If fractions are out of range or their sum exceeds 1.
    """
    if not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("val_size and test_size must each be strictly between 0 and 1.")
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must be less than 1."
        )

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=seed,
    )

    # Compute the fraction relative to the train_val pool that gives the
    # desired absolute validation size.
    relative_val = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[target],
        random_state=seed,
    )

    return train, val, test
