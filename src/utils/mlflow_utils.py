"""MLflow helpers for experiment initialisation and metric logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def init_mlflow(
    experiment_name: str,
    tracking_uri: str | Path = "mlruns/",
) -> str:
    """Configure MLflow tracking URI and set the active experiment.

    Creates the experiment if it does not already exist.

    Args:
        experiment_name: Human-readable label for the experiment.
        tracking_uri: Local directory or remote URI for the MLflow store.

    Returns:
        MLflow experiment ID string.
    """
    mlflow.set_tracking_uri(str(tracking_uri))
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info("Created MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Reusing MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_metrics_and_artifacts(
    metrics: dict[str, float],
    artifacts: dict[str, str | Path] | None = None,
    tags: dict[str, Any] | None = None,
) -> None:
    """Log metrics, file artifacts, and tags to the active MLflow run.

    Must be called inside an active ``mlflow.start_run()`` context.

    Args:
        metrics: Mapping of metric name to scalar value.
        artifacts: Mapping of descriptive label to local file path.
            The label is not passed to MLflow; the file is uploaded as-is.
        tags: Additional string key-value pairs attached to the run.
    """
    mlflow.log_metrics(metrics)
    if tags:
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
    if artifacts:
        for _, path in artifacts.items():
            mlflow.log_artifact(str(path))
