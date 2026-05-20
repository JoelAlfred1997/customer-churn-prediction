"""EDA helper utilities for churn rate analysis and statistical feature ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def churn_rate_by_group(
    df: pd.DataFrame,
    group_col: str,
    *,
    target_col: str = "Churn",
    positive_label: str = "Yes",
) -> pd.DataFrame:
    """Compute churn rate, absolute churner count, and group size for each level of *group_col*.

    Args:
        df: DataFrame containing both *group_col* and *target_col*.
        group_col: Column to group by (categorical, string, or ordered Categorical).
        target_col: Binary target column.
        positive_label: Value in *target_col* that represents the positive class.

    Returns:
        DataFrame indexed by group level with columns ``count``, ``churners``,
        ``churn_rate``.  Sorted descending by ``churn_rate``.
    """
    grouped = df.groupby(group_col, observed=True)[target_col].agg(
        count="count",
        churners=lambda s: (s == positive_label).sum(),
    )
    grouped["churn_rate"] = grouped["churners"] / grouped["count"]
    return grouped.sort_values("churn_rate", ascending=False)


def chi_square_table(
    df: pd.DataFrame,
    cat_cols: list[str],
    *,
    target_col: str = "Churn",
) -> pd.DataFrame:
    """Run Pearson chi-square test of independence for each column in *cat_cols* vs *target_col*.

    Cramér's V is the effect-size statistic; it is bounded [0, 1] and comparable
    across features with different numbers of categories.

    Args:
        df: DataFrame with feature and target columns.
        cat_cols: Categorical feature columns to test.
        target_col: Binary target column.

    Returns:
        DataFrame indexed by feature name with columns ``chi2``, ``p_value``,
        ``cramers_v``, ``significant``.  Sorted descending by ``cramers_v``.
    """
    rows: list[dict] = []
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        n = int(contingency.to_numpy().sum())
        k = min(contingency.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0
        rows.append(
            {
                "feature": col,
                "chi2": round(chi2, 2),
                "p_value": p_value,
                "cramers_v": round(cramers_v, 4),
                "significant": p_value < 0.05,
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values("cramers_v", ascending=False)
    )


def compute_mutual_info(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "Churn",
    positive_label: str = "Yes",
    random_state: int = 42,
) -> pd.Series:
    """Compute mutual information between each feature in *feature_cols* and the churn target.

    Categorical and Categorical-dtype columns are label-encoded before calling
    ``sklearn.feature_selection.mutual_info_classif``.  The ``discrete_features``
    flag is set per-column so sklearn treats numeric and categorical features
    correctly.

    Args:
        df: DataFrame with feature and target columns.
        feature_cols: Columns to rank.  May be a mix of numeric and categorical.
        target_col: Binary target column.
        positive_label: Value in *target_col* treated as class 1.
        random_state: Random seed for reproducibility.

    Returns:
        Series mapping feature name → MI score, sorted descending.
    """
    work = df[feature_cols].copy()
    discrete_mask: list[bool] = []

    for col in feature_cols:
        is_categorical = work[col].dtype == object or isinstance(
            work[col].dtype, pd.CategoricalDtype
        )
        # Low-cardinality integers (e.g. binary flags like SeniorCitizen) are
        # discrete; the KNN-based MI estimator is only appropriate for continuous.
        is_discrete_int = (
            np.issubdtype(work[col].dtype, np.integer) and work[col].nunique() <= 20
        )
        if is_categorical:
            work[col] = LabelEncoder().fit_transform(work[col].astype(str))
            discrete_mask.append(True)
        elif is_discrete_int:
            discrete_mask.append(True)
        else:
            discrete_mask.append(False)

    y = (df[target_col] == positive_label).astype(int)
    scores = mutual_info_classif(
        work.to_numpy(),
        y.to_numpy(),
        discrete_features=discrete_mask,
        random_state=random_state,
    )
    return pd.Series(scores, index=feature_cols, name="mutual_info").sort_values(
        ascending=False
    )
