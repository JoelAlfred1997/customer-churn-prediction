"""CLI entry point for training XGBoost and LightGBM models with MLflow tracking.

Usage
-----
    python scripts/train_gbm.py
    python scripts/train_gbm.py --config configs/config.yaml --n-splits 5
    python scripts/train_gbm.py --no-mlflow
    python scripts/train_gbm.py --model xgboost   # train one model only
    python scripts/train_gbm.py --model lightgbm
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import prepare_data
from src.models.cv import cv_summary, stratified_cv_score
from src.models.gbm import train_lightgbm, train_xgboost
from src.utils.mlflow_utils import init_mlflow, log_metrics_and_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost and LightGBM churn models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "config.yaml",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "both"],
        default="both",
        help="Which model(s) to train.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging.",
    )
    return parser.parse_args()


@contextmanager
def _maybe_run(run_name: str, use_mlflow: bool) -> Iterator[None]:
    if use_mlflow:
        with mlflow.start_run(run_name=run_name):
            yield
    else:
        yield


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

    X_cv = np.vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])

    if use_mlflow:
        init_mlflow(experiment_name, tracking_uri=tracking_uri)

    results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    if args.model in ("xgboost", "both"):
        from xgboost import XGBClassifier

        xgb_hp = cfg.get("xgboost", {})
        xgb_cv_proto = XGBClassifier(
            n_estimators=xgb_hp.get("n_estimators", 300),
            learning_rate=xgb_hp.get("learning_rate", 0.05),
            max_depth=xgb_hp.get("max_depth", 6),
            subsample=xgb_hp.get("subsample", 0.8),
            colsample_bytree=xgb_hp.get("colsample_bytree", 0.8),
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )

        logger.info("Running %d-fold CV for XGBoost", args.n_splits)
        xgb_cv = stratified_cv_score(xgb_cv_proto, X_cv, y_cv, n_splits=args.n_splits, seed=seed)
        logger.info("XGB CV:\n%s", xgb_cv.to_string(index=False))
        logger.info("XGB CV summary:\n%s", cv_summary(xgb_cv).to_string(index=False))

        with _maybe_run("xgboost_v1", use_mlflow):
            model_xgb, val_metrics_xgb = train_xgboost(
                X_train,
                y_train,
                X_val,
                y_val,
                params={k: v for k, v in xgb_hp.items() if k not in ("use_label_encoder", "eval_metric")},
                early_stopping_rounds=args.early_stopping_rounds,
                seed=seed,
                log_to_mlflow=use_mlflow,
            )
            if use_mlflow:
                cv_means = {f"cv_{c}_mean": float(xgb_cv[c].mean()) for c in xgb_cv.columns if c != "fold"}
                cv_stds = {f"cv_{c}_std": float(xgb_cv[c].std()) for c in xgb_cv.columns if c != "fold"}
                log_metrics_and_artifacts(
                    {**cv_means, **cv_stds},
                    tags={"model": "xgboost", "n_cv_splits": str(args.n_splits)},
                )

        xgb_path = models_dir / "xgb_v1.pkl"
        xgb_path.write_bytes(pickle.dumps(model_xgb))
        logger.info("Saved XGBoost model to %s", xgb_path)

        results["xgboost"] = {"cv": xgb_cv.to_dict(orient="records"), "val": val_metrics_xgb}

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    if args.model in ("lightgbm", "both"):
        import lightgbm as lgb

        lgb_hp = cfg.get("lightgbm", {})
        lgb_cv_proto = lgb.LGBMClassifier(
            n_estimators=lgb_hp.get("n_estimators", 300),
            learning_rate=lgb_hp.get("learning_rate", 0.05),
            num_leaves=lgb_hp.get("num_leaves", 31),
            subsample=lgb_hp.get("subsample", 0.8),
            colsample_bytree=lgb_hp.get("colsample_bytree", 0.8),
            random_state=seed,
            verbose=-1,
        )

        logger.info("Running %d-fold CV for LightGBM", args.n_splits)
        lgb_cv = stratified_cv_score(lgb_cv_proto, X_cv, y_cv, n_splits=args.n_splits, seed=seed)
        logger.info("LGB CV:\n%s", lgb_cv.to_string(index=False))
        logger.info("LGB CV summary:\n%s", cv_summary(lgb_cv).to_string(index=False))

        with _maybe_run("lightgbm_v1", use_mlflow):
            model_lgb, val_metrics_lgb = train_lightgbm(
                X_train,
                y_train,
                X_val,
                y_val,
                params=lgb_hp,
                early_stopping_rounds=args.early_stopping_rounds,
                seed=seed,
                log_to_mlflow=use_mlflow,
            )
            if use_mlflow:
                cv_means = {f"cv_{c}_mean": float(lgb_cv[c].mean()) for c in lgb_cv.columns if c != "fold"}
                cv_stds = {f"cv_{c}_std": float(lgb_cv[c].std()) for c in lgb_cv.columns if c != "fold"}
                log_metrics_and_artifacts(
                    {**cv_means, **cv_stds},
                    tags={"model": "lightgbm", "n_cv_splits": str(args.n_splits)},
                )

        lgb_path = models_dir / "lgbm_v1.pkl"
        lgb_path.write_bytes(pickle.dumps(model_lgb))
        logger.info("Saved LightGBM model to %s", lgb_path)

        results["lightgbm"] = {"cv": lgb_cv.to_dict(orient="records"), "val": val_metrics_lgb}

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    import pandas as pd

    rows = []
    for name, res in results.items():
        cv_df = pd.DataFrame(res["cv"])
        rows.append(
            {
                "model": name,
                "cv_roc_auc": f"{cv_df['roc_auc'].mean():.4f} ± {cv_df['roc_auc'].std():.4f}",
                "cv_f1": f"{cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}",
                "val_roc_auc": f"{res['val']['roc_auc']:.4f}",
                "val_f1": f"{res['val']['f1']:.4f}",
            }
        )

    if rows:
        comparison = pd.DataFrame(rows)
        print("\n" + "=" * 70)
        print("GBM MODEL COMPARISON")
        print("=" * 70)
        print(comparison.to_string(index=False))
        print("=" * 70 + "\n")

    results_path = REPO_ROOT / "reports" / "gbm_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
