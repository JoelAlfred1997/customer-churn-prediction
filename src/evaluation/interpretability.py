"""SHAP-based interpretability utilities for churn prediction models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.utils.plotting import save_fig, set_plot_style


def clean_feature_names(names: list[str]) -> list[str]:
    """Strip ColumnTransformer prefix (``num__``, ``cat__``, ``ord__``)."""
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def build_explainer(
    model: Any,
    X_background: np.ndarray | None = None,
    *,
    feature_names: list[str] | None = None,
) -> Any:
    """Build a SHAP explainer for *model*.

    Uses :class:`shap.TreeExplainer` for XGBoost and LightGBM (fast, exact
    Shapley values). Falls back to :class:`shap.KernelExplainer` for other
    model types, which requires a background dataset.

    Args:
        model: Fitted sklearn-compatible estimator.
        X_background: Background data for KernelExplainer; ignored for tree
            models. Typically 100–200 rows sampled from the training set.
        feature_names: Optional feature names stored on the explainer for
            downstream use in :func:`get_shap_values`.

    Returns:
        Fitted SHAP explainer (TreeExplainer or KernelExplainer).

    Raises:
        ValueError: If a non-tree model is passed without *X_background*.
    """
    underlying = model
    if hasattr(model, "named_steps"):
        underlying = list(model.named_steps.values())[-1]

    _tree_types = {"XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor"}
    is_tree = type(underlying).__name__ in _tree_types

    if is_tree:
        explainer = shap.TreeExplainer(underlying)
    else:
        if X_background is None:
            raise ValueError(
                "X_background is required for non-tree models (KernelExplainer)."
            )
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],  # noqa: E731
            X_background,
        )

    if feature_names is not None:
        explainer.feature_names = feature_names

    return explainer


def get_shap_values(
    explainer: Any,
    X: np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> shap.Explanation:
    """Compute SHAP values for *X* and wrap them in a :class:`shap.Explanation`.

    Handles binary-classification TreeExplainers that may return a list of
    per-class arrays; always extracts values for the positive class (churn=1).

    Args:
        explainer: Fitted SHAP explainer (TreeExplainer or KernelExplainer).
        X: Feature matrix to explain, shape (n_samples, n_features).
        feature_names: Column names for plots; falls back to the attribute set
            on the explainer by :func:`build_explainer`.

    Returns:
        :class:`shap.Explanation` with ``.values`` of shape
        (n_samples, n_features) representing SHAP contributions toward churn.
    """
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        values = raw[1] if len(raw) == 2 else raw[0]
    else:
        values = raw

    ev = explainer.expected_value
    expected = float(
        ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) >= 2 else ev
    )

    names = feature_names or getattr(explainer, "feature_names", None)

    return shap.Explanation(
        values=values,
        base_values=np.full(len(values), expected),
        data=X,
        feature_names=names,
    )


def top_features_by_importance(
    shap_explanation: shap.Explanation,
    *,
    n: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-n features sorted by mean absolute SHAP value.

    Args:
        shap_explanation: Explanation with ``.values`` and ``.feature_names``.
        n: Number of features to return.

    Returns:
        List of ``(feature_name, mean_abs_shap)`` tuples, sorted descending.
    """
    mean_abs = np.abs(shap_explanation.values).mean(axis=0)
    names = shap_explanation.feature_names or [f"f{i}" for i in range(len(mean_abs))]
    ranked = sorted(zip(names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:n]


def plot_summary(
    shap_explanation: shap.Explanation,
    *,
    max_display: int = 20,
    title: str = "SHAP Feature Impact (Beeswarm)",
    figures_dir: Path | str | None = None,
    fname: str = "shap_summary_beeswarm",
) -> plt.Figure:
    """Beeswarm summary plot showing global feature impact distribution.

    Each dot is one customer; the x-axis is the SHAP value (impact on
    log-odds of churn). Colour encodes the raw feature value (red=high,
    blue=low), revealing whether a feature increases or decreases churn risk.

    Args:
        shap_explanation: Explanation with values, data, and feature_names.
        max_display: Maximum number of features shown.
        title: Figure suptitle.
        figures_dir: Directory to save the figure; skipped if None.
        fname: Filename stem (no extension).

    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    shap.summary_plot(
        shap_explanation.values,
        shap_explanation.data,
        feature_names=shap_explanation.feature_names,
        max_display=max_display,
        plot_size=(10, max(5, min(max_display, 20) * 0.40)),
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(title, fontsize=13, y=1.01)

    if figures_dir is not None:
        save_fig(fig, fname, figures_dir)

    return fig


def plot_importance_bar(
    shap_explanation: shap.Explanation,
    *,
    max_display: int = 20,
    title: str = "Mean |SHAP| — Global Feature Importance",
    figures_dir: Path | str | None = None,
    fname: str = "shap_importance_bar",
) -> plt.Figure:
    """Bar chart of mean absolute SHAP values (global importance ranking).

    Args:
        shap_explanation: Explanation with values and feature_names.
        max_display: Maximum features shown.
        title: Figure suptitle.
        figures_dir: Save directory; skipped if None.
        fname: Filename stem.

    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    shap.summary_plot(
        shap_explanation.values,
        shap_explanation.data,
        feature_names=shap_explanation.feature_names,
        max_display=max_display,
        plot_type="bar",
        plot_size=(9, max(4, min(max_display, 20) * 0.37)),
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(title, fontsize=13, y=1.01)

    if figures_dir is not None:
        save_fig(fig, fname, figures_dir)

    return fig


def plot_dependence(
    shap_explanation: shap.Explanation,
    feature: str | int,
    *,
    interaction_feature: str | int | None = "auto",
    title: str | None = None,
    figures_dir: Path | str | None = None,
    fname: str | None = None,
) -> plt.Axes:
    """SHAP dependence plot for a single feature.

    Shows how the SHAP value for *feature* varies with its magnitude, coloured
    by the feature with the strongest interaction effect (auto-detected by
    default). Reveals monotone and non-linear relationships, and interactions.

    Args:
        shap_explanation: Explanation with values, data, and feature_names.
        feature: Feature name (str) or column index (int) to plot.
        interaction_feature: Feature to colour by; ``"auto"`` lets SHAP pick
            the strongest interaction, ``None`` colours by the feature itself.
        title: Axes title; auto-generated from the feature name if None.
        figures_dir: Save directory; skipped if None.
        fname: Filename stem; auto-generated if None.

    Returns:
        Matplotlib axes.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    shap.dependence_plot(
        feature,
        shap_explanation.values,
        shap_explanation.data,
        feature_names=shap_explanation.feature_names,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )

    names = shap_explanation.feature_names
    if isinstance(feature, str):
        feat_label = feature
    elif names:
        feat_label = names[feature]
    else:
        feat_label = f"feature {feature}"

    ax.set_title(title or f"SHAP Dependence — {feat_label}")
    fig.tight_layout()

    if figures_dir is not None:
        slug = feat_label.replace(" ", "_").replace("/", "_").replace(".", "_")
        save_fig(fig, fname or f"shap_dependence_{slug}", figures_dir)

    return ax


def explain_customer(
    shap_explanation: shap.Explanation,
    customer_idx: int,
    *,
    label: str = "customer",
    max_display: int = 15,
    figures_dir: Path | str | None = None,
    fname: str | None = None,
) -> plt.Figure:
    """Waterfall plot for a single customer's SHAP values.

    Shows how each feature pushes the churn prediction above or below the
    background base rate, making the individual prediction fully transparent.
    The leftmost bar starts at the expected model output (base value) and each
    feature contribution is stacked until the final prediction.

    Args:
        shap_explanation: Full-sample Explanation; sliced internally by index.
        customer_idx: Row index within *shap_explanation* to explain.
        label: Descriptive archetype label for the title (e.g. ``"high-risk"``).
        max_display: Maximum number of features shown in the waterfall.
        figures_dir: Save directory; skipped if None.
        fname: Filename stem; auto-generated if None.

    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    sv_single = shap_explanation[customer_idx]
    shap.plots.waterfall(sv_single, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Local Explanation — {label}", fontsize=13, y=1.02)

    if figures_dir is not None:
        slug = label.lower().replace(" ", "_")
        save_fig(fig, fname or f"shap_local_{slug}", figures_dir)

    return fig


def save_explainer(explainer: Any, path: Path | str) -> Path:
    """Persist *explainer* to *path* using pickle.

    Args:
        explainer: Fitted SHAP explainer.
        path: Destination file path; parent directories created if absent.

    Returns:
        Resolved absolute path of the saved file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        pickle.dump(explainer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return dest.resolve()


def load_explainer(path: Path | str) -> Any:
    """Load a pickled SHAP explainer from *path*.

    Args:
        path: Path to the pickled explainer file.

    Returns:
        Deserialized SHAP explainer.
    """
    with Path(path).open("rb") as fh:
        return pickle.load(fh)
