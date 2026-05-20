"""CLI entry point for training baseline models with MLflow tracking.

Usage
-----
    python scripts/train_baselines.py
    python scripts/train_baselines.py --config configs/config.yaml --n-splits 5
    python scripts/train_baselines.py --no-mlflow   # skip MLflow logging
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import prepare_data
from src.models.baseline import train_decision_tree, train_logistic
from src.models.cv import cv_summary, stratified_cv_score
from src.utils.mlflow_utils import init_mlflow, log_metrics_and_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LR and DT baseline models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging (useful for quick debugging runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    seed: int = cfg["random_seed"]
    raw_path = REPO_ROOT / cfg["paths"]["raw_data"]
    processed_dir = REPO_ROOT / cfg["paths"]["processed_data"]
    models_dir = REPO_ROOT / cfg["paths"]["models"]
    tracking_uri = str(REPO_ROOT / cfg["paths"]["mlruns"])
    experiment_name: str = cfg["mlflow"]["experiment_name"]

    models_dir.mkdir(parents=True, exist_ok=True)
    use_mlflow = not args.no_mlflow

    logger.info("Loading and preparing data from %s", raw_path)
    data = prepare_data(
        raw_path,
        val_size=cfg["data"]["val_size"],
        test_size=cfg["data"]["test_size"],
        seed=seed,
        processed_dir=processed_dir,
    )

    X_train = data["X_train"]
    y_train = data["y_train"].to_numpy()
    X_val = data["X_val"]
    y_val = data["y_val"].to_numpy()

    # Combine train + val for more robust CV estimates
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])

    if use_mlflow:
        init_mlflow(experiment_name, tracking_uri=tracking_uri)

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression

    lr_estimator = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=seed
    )

    logger.info("Running %d-fold CV for Logistic Regression", args.n_splits)
    lr_cv = stratified_cv_score(lr_estimator, X_cv, y_cv, n_splits=args.n_splits, seed=seed)
    lr_summary = cv_summary(lr_cv)

    logger.info("LR CV results:\n%s", lr_cv.to_string(index=False))
    logger.info("LR CV summary:\n%s", lr_summary.to_string(index=False))

    run_name_lr = "logistic_regression_baseline"
    with mlflow.start_run(run_name=run_name_lr) if use_mlflow else _null_context():
        model_lr, val_metrics_lr = train_logistic(
            X_train, y_train, X_val, y_val, seed=seed, log_to_mlflow=use_mlflow
        )
        if use_mlflow:
            cv_metrics = {
                f"cv_{col}": float(lr_cv[col].mean())
                for col in lr_cv.columns
                if col != "fold"
            }
            cv_std = {
                f"cv_{col}_std": float(lr_cv[col].std())
                for col in lr_cv.columns
                if col != "fold"
            }
            log_metrics_and_artifacts(
                {**cv_metrics, **cv_std},
                tags={"model": "logistic_regression", "n_cv_splits": str(args.n_splits)},
            )

    # Save model
    lr_path = models_dir / "logistic_regression_baseline.pkl"
    with lr_path.open("wb") as fh:
        pickle.dump(model_lr, fh)
    logger.info("Saved LR model to %s", lr_path)

    # ------------------------------------------------------------------
    # Decision Tree
    # ------------------------------------------------------------------
    from sklearn.tree import DecisionTreeClassifier

    dt_estimator = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        criterion="gini",
        random_state=seed,
    )

    logger.info("Running %d-fold CV for Decision Tree", args.n_splits)
    dt_cv = stratified_cv_score(dt_estimator, X_cv, y_cv, n_splits=args.n_splits, seed=seed)
    dt_summary = cv_summary(dt_cv)

    logger.info("DT CV results:\n%s", dt_cv.to_string(index=False))
    logger.info("DT CV summary:\n%s", dt_summary.to_string(index=False))

    run_name_dt = "decision_tree_baseline"
    with mlflow.start_run(run_name=run_name_dt) if use_mlflow else _null_context():
        model_dt, val_metrics_dt = train_decision_tree(
            X_train, y_train, X_val, y_val, seed=seed, log_to_mlflow=use_mlflow
        )
        if use_mlflow:
            cv_metrics = {
                f"cv_{col}": float(dt_cv[col].mean())
                for col in dt_cv.columns
                if col != "fold"
            }
            cv_std = {
                f"cv_{col}_std": float(dt_cv[col].std())
                for col in dt_cv.columns
                if col != "fold"
            }
            log_metrics_and_artifacts(
                {**cv_metrics, **cv_std},
                tags={"model": "decision_tree", "n_cv_splits": str(args.n_splits)},
            )

    # Save model
    dt_path = models_dir / "decision_tree_baseline.pkl"
    with dt_path.open("wb") as fh:
        pickle.dump(model_dt, fh)
    logger.info("Saved DT model to %s", dt_path)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    import pandas as pd

    comparison = pd.DataFrame({
        "model": ["LogisticRegression", "DecisionTree"],
        "cv_roc_auc_mean": [lr_cv["roc_auc"].mean(), dt_cv["roc_auc"].mean()],
        "cv_roc_auc_std": [lr_cv["roc_auc"].std(), dt_cv["roc_auc"].std()],
        "cv_f1_mean": [lr_cv["f1"].mean(), dt_cv["f1"].mean()],
        "cv_f1_std": [lr_cv["f1"].std(), dt_cv["f1"].std()],
        "val_roc_auc": [val_metrics_lr["roc_auc"], val_metrics_dt["roc_auc"]],
        "val_f1": [val_metrics_lr["f1"], val_metrics_dt["f1"]],
    })
    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print("=" * 70 + "\n")

    # Save comparison to JSON for downstream reference
    results_path = REPO_ROOT / "reports" / "baseline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "logistic_regression": {
                    "cv": lr_cv.to_dict(orient="records"),
                    "val": val_metrics_lr,
                },
                "decision_tree": {
                    "cv": dt_cv.to_dict(orient="records"),
                    "val": val_metrics_dt,
                },
            },
            indent=2,
        )
    )
    logger.info("Results saved to %s", results_path)


class _null_context:
    """Minimal no-op context manager used when MLflow logging is disabled."""

    def __enter__(self) -> "_null_context":
        return self

    def __exit__(self, *_: object) -> None:
        pass


if __name__ == "__main__":
    main()
