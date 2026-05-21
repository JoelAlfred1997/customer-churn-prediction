"""CLI for Bayesian hyperparameter optimisation of XGBoost with Optuna.

Usage
-----
    python scripts/tune_xgboost.py
    python scripts/tune_xgboost.py --n-trials 100 --study-name xgb-churn-v2
    python scripts/tune_xgboost.py --storage sqlite:///models/optuna.db
    python scripts/tune_xgboost.py --no-mlflow
"""

from __future__ import annotations

import argparse
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
from src.models.tuning import best_params_as_xgb_kwargs, optimize_xgboost
from src.utils.mlflow_utils import init_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise XGBoost hyperparameters with Optuna."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "config.yaml",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides configs/config.yaml).",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="xgb-churn",
        help="Name of the Optuna study.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "SQLAlchemy URI for study persistence, e.g. "
            "sqlite:///models/optuna.db. "
            "Omit for in-memory (no resume support)."
        ),
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show tqdm progress bar.",
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

    optuna_cfg = cfg.get("optuna", {})
    n_trials = args.n_trials if args.n_trials is not None else optuna_cfg.get("n_trials", 50)

    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing data from %s", raw_path)
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

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        init_mlflow(experiment_name, tracking_uri=tracking_uri)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    logger.info("Running %d Optuna trials (study='%s')", n_trials, args.study_name)
    study = optimize_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=n_trials,
        study_name=args.study_name,
        storage=args.storage,
        seed=seed,
        show_progress_bar=args.progress_bar,
    )

    best = study.best_trial
    best_xgb_kwargs = best_params_as_xgb_kwargs(study)
    logger.info("Best val AUC = %.5f (trial #%d)", best.value, best.number)
    logger.info("Best XGBClassifier kwargs: %s", best_xgb_kwargs)

    # ------------------------------------------------------------------
    # Persist best params to YAML
    # ------------------------------------------------------------------
    best_params_path = REPO_ROOT / "configs" / "best_params.yaml"
    best_params_record = {
        "study_name": args.study_name,
        "n_trials_run": len(study.trials),
        "best_trial": best.number,
        "best_value": float(best.value),
        "params": {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                   for k, v in best_xgb_kwargs.items()},
    }
    best_params_path.write_text(yaml.dump(best_params_record, default_flow_style=False))
    logger.info("Best params written to %s", best_params_path)

    # ------------------------------------------------------------------
    # Retrain on train+val with best params; evaluate on test
    # ------------------------------------------------------------------
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, f1_score

    X_test = data["X_test"]
    y_test = data["y_test"].to_numpy()

    logger.info("Retraining tuned XGBoost on train+val (%d rows)", X_cv.shape[0])
    tuned_model = XGBClassifier(
        **best_xgb_kwargs,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    # Final model: train on combined train+val, no early stopping
    tuned_model.fit(X_cv, y_cv, verbose=False)

    test_proba = tuned_model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, test_proba)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    logger.info("Tuned model test AUC=%.4f  F1=%.4f", test_auc, test_f1)

    # ------------------------------------------------------------------
    # Save tuned model
    # ------------------------------------------------------------------
    model_path = models_dir / "xgb_tuned.pkl"
    model_path.write_bytes(pickle.dumps(tuned_model))
    logger.info("Tuned model saved to %s", model_path)

    # ------------------------------------------------------------------
    # MLflow — log study summary as a single run
    # ------------------------------------------------------------------
    if use_mlflow:
        with mlflow.start_run(run_name=f"xgb_tuned_{args.study_name}"):
            mlflow.set_tags({"model": "xgboost_tuned", "study": args.study_name})
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_params(
                {k: (int(v) if isinstance(v, np.integer) else v)
                 for k, v in best_xgb_kwargs.items()}
            )
            mlflow.log_metrics(
                {
                    "best_val_auc": float(best.value),
                    "test_auc": test_auc,
                    "test_f1": test_f1,
                }
            )
            mlflow.log_artifact(str(best_params_path))

    print(f"\nBest val AUC : {best.value:.5f}")
    print(f"Test AUC     : {test_auc:.4f}")
    print(f"Test F1      : {test_f1:.4f}")
    print(f"Model saved  : {model_path}")
    print(f"Params saved : {best_params_path}")


if __name__ == "__main__":
    main()
