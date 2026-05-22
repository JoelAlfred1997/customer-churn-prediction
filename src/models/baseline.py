"""Baseline model training: Logistic Regression and Decision Tree."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


def _val_metrics(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "f1": float(f1_score(y_val, preds, zero_division=0)),
        "recall": float(recall_score(y_val, preds, zero_division=0)),
        "precision": float(precision_score(y_val, preds, zero_division=0)),
        "average_precision": float(average_precision_score(y_val, proba)),
    }


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 42,
    log_to_mlflow: bool = False,
) -> tuple[LogisticRegression, dict[str, float]]:
    """Fit an L2-regularised Logistic Regression with balanced class weights.

    Args:
        X_train: Preprocessed training features.
        y_train: Binary training labels.
        X_val: Preprocessed validation features.
        y_val: Binary validation labels.
        seed: Random seed for reproducibility.
        log_to_mlflow: Log params and metrics to the active MLflow run.

    Returns:
        Trained model and dict of validation metrics.
    """
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    metrics = _val_metrics(model, X_val, y_val)

    if log_to_mlflow:
        mlflow.log_params({"C": 1.0, "solver": "lbfgs", "class_weight": "balanced"})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

    logger.info(
        "LR  val AUC=%.4f  F1=%.4f  Recall=%.4f",
        metrics["roc_auc"], metrics["f1"], metrics["recall"],
    )
    return model, metrics


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 42,
    log_to_mlflow: bool = False,
) -> tuple[DecisionTreeClassifier, dict[str, float]]:
    """Fit a shallow Decision Tree with balanced class weights.

    Args:
        X_train: Preprocessed training features.
        y_train: Binary training labels.
        X_val: Preprocessed validation features.
        y_val: Binary validation labels.
        seed: Random seed for reproducibility.
        log_to_mlflow: Log params and metrics to the active MLflow run.

    Returns:
        Trained model and dict of validation metrics.
    """
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        criterion="gini",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    metrics = _val_metrics(model, X_val, y_val)

    if log_to_mlflow:
        mlflow.log_params({"max_depth": 5, "min_samples_leaf": 20, "class_weight": "balanced"})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

    logger.info(
        "DT  val AUC=%.4f  F1=%.4f  Recall=%.4f",
        metrics["roc_auc"], metrics["f1"], metrics["recall"],
    )
    return model, metrics
