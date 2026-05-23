"""Model evaluation, threshold optimisation, and reporting."""

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
