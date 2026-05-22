"""Optuna-based hyperparameter optimisation for XGBoost."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_trials: int = 50,
    study_name: str = "xgb-churn",
    storage: str | None = None,
    seed: int = 42,
    show_progress_bar: bool = False,
) -> Any:
    """Run Optuna TPE search over XGBoost hyperparameters.

    Uses a MedianPruner with a 5-trial warm-up to discard unpromising trials
    early. The study is resumable when *storage* points to a SQLite database.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        n_trials: Number of Optuna trials to run.
        study_name: Name of the Optuna study.
        storage: SQLite or other Optuna storage URI; in-memory if None.
        seed: Random seed for the sampler.
        show_progress_bar: Display tqdm progress bar.

    Returns:
        Completed ``optuna.Study`` object.
    """
    import optuna
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }
        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False,
        )
        proba = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, proba))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    logger.info("Best trial #%d  AUC=%.5f", study.best_trial.number, study.best_value)
    return study


def best_params_as_xgb_kwargs(study: Any) -> dict[str, Any]:
    """Extract best-trial parameters formatted for XGBClassifier.

    Args:
        study: Completed Optuna study returned by :func:`optimize_xgboost`.

    Returns:
        Dict suitable for passing directly to ``XGBClassifier(**kwargs)``.
    """
    return dict(study.best_params)
