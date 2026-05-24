"""Model evaluation, threshold optimisation, and reporting."""

from src.evaluation.interpretability import (
    build_explainer,
    clean_feature_names,
    explain_customer,
    get_shap_values,
    load_explainer,
    plot_dependence,
    plot_importance_bar,
    plot_summary,
    save_explainer,
    top_features_by_importance,
)
from src.evaluation.metrics import classification_report_extended, compare_models
from src.evaluation.plots import (
    plot_calibration,
    plot_confusion_matrix,
    plot_decision_curve,
    plot_lift_gains,
    plot_pr,
    plot_roc,
)
from src.evaluation.threshold import cost_curve, optimal_threshold, threshold_sensitivity

__all__ = [
    "build_explainer",
    "clean_feature_names",
    "explain_customer",
    "get_shap_values",
    "load_explainer",
    "plot_dependence",
    "plot_importance_bar",
    "plot_summary",
    "save_explainer",
    "top_features_by_importance",
    "classification_report_extended",
    "compare_models",
    "plot_calibration",
    "plot_confusion_matrix",
    "plot_decision_curve",
    "plot_lift_gains",
    "plot_pr",
    "plot_roc",
    "cost_curve",
    "optimal_threshold",
    "threshold_sensitivity",
]
