"""XGBoost and LightGBM training functions with early stopping."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    seed: int = 42,
    log_to_mlflow: bool = False,
) -> tuple[Any, dict[str, float]]:
    """Train XGBoost with early stopping on the validation set.

    Args:
        X_train: Preprocessed training features.
        y_train: Binary training labels.
        X_val: Preprocessed validation features.
        y_val: Binary validation labels.
        params: XGBClassifier constructor kwargs (overrides defaults).
        early_stopping_rounds: Patience rounds for early stopping.
        seed: Random seed.
        log_to_mlflow: Log params and metrics to the active MLflow run.

    Returns:
        Fitted XGBClassifier and dict of validation metrics.
    """
    from xgboost import XGBClassifier

    defaults: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if params:
        defaults.update(params)

    model = XGBClassifier(
        **defaults,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    metrics = _val_metrics(model, X_val, y_val)

    if log_to_mlflow:
        mlflow.log_params({**defaults, "early_stopping_rounds": early_stopping_rounds})
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

    logger.info(
        "XGB val AUC=%.4f  F1=%.4f  best_iter=%d",
        metrics["roc_auc"], metrics["f1"], model.best_iteration,
    )
    return model, metrics


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    seed: int = 42,
    log_to_mlflow: bool = False,
) -> tuple[Any, dict[str, float]]:
    """Train LightGBM with early stopping on the validation set.

    Args:
        X_train: Preprocessed training features.
        y_train: Binary training labels.
        X_val: Preprocessed validation features.
        y_val: Binary validation labels.
        params: LGBMClassifier constructor kwargs (overrides defaults).
        early_stopping_rounds: Patience rounds for early stopping.
        seed: Random seed.
        log_to_mlflow: Log params and metrics to the active MLflow run.

    Returns:
        Fitted LGBMClassifier and dict of validation metrics.
    """
    import lightgbm as lgb

    defaults: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if params:
        defaults.update(params)

    model = lgb.LGBMClassifier(
        **defaults,
        random_state=seed,
        verbose=-1,
    )
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(-1),
    ]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    metrics = _val_metrics(model, X_val, y_val)

    if log_to_mlflow:
        mlflow.log_params({**defaults, "early_stopping_rounds": early_stopping_rounds})
        mlflow.log_metric("best_iteration", model.best_iteration_)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

    logger.info(
        "LGB val AUC=%.4f  F1=%.4f  best_iter=%d",
        metrics["roc_auc"], metrics["f1"], model.best_iteration_,
    )
    return model, metrics
