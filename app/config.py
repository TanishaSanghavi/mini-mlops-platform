import os
from pathlib import Path

MLFLOW_EXPERIMENT_NAME = "fraud-detection-training"
MODEL_NAME = "fraud-detection-model"

_LOCAL_TRACKING_URI = "sqlite:///mlflow.db"
_DOCKER_TRACKING_URI = "http://mlflow:5000"


def _is_running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def _resolve_tracking_uri() -> str:
    if uri := os.getenv("MLFLOW_TRACKING_URI"):
        return uri
    if _is_running_in_docker():
        return _DOCKER_TRACKING_URI
    return _LOCAL_TRACKING_URI


MLFLOW_TRACKING_URI = _resolve_tracking_uri()
