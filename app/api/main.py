import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from app import config

REGISTERED_MODEL_URI = f"models:/{config.MODEL_NAME}/Staging"
LOCAL_MODEL_PATH = Path("models/fraud_model.pkl")
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

_model: Any = None
_feature_columns: list[str] = FEATURE_COLUMNS
_model_load_error: str | None = None


class PredictionRequest(BaseModel):
    """Feature payload matching training columns (excluding Class)."""

    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


def _resolve_feature_columns(model: Any) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return FEATURE_COLUMNS.copy()


def _load_from_registry() -> Any:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(REGISTERED_MODEL_URI)


def _load_from_local() -> Any:
    if not LOCAL_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Local model not found at {LOCAL_MODEL_PATH.resolve()}")
    with LOCAL_MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def _load_model() -> Any | None:
    global _model_load_error
    try:
        model = _load_from_registry()
        print("Loaded model from MLflow registry")
        _model_load_error = None
        return model
    except Exception as registry_exc:
        try:
            model = _load_from_local()
            print("Loaded local fallback model")
            _model_load_error = None
            return model
        except Exception as local_exc:
            _model_load_error = (
                f"MLflow registry failed: {registry_exc}. "
                f"Local fallback failed: {local_exc}"
            )
            return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _feature_columns
    _model = _load_model()
    if _model is not None:
        _feature_columns = _resolve_feature_columns(_model)
    yield


app = FastAPI(title="Mini MLOps Platform", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, int | str]:
    if _model is None:
        detail = (
            f"Unable to load model from MLflow registry ({REGISTERED_MODEL_URI}) "
            f"or local fallback ({LOCAL_MODEL_PATH}). Train and register a model first."
        )
        if _model_load_error:
            detail = f"{detail} Error: {_model_load_error}"
        raise HTTPException(status_code=503, detail=detail)

    try:
        features = request.model_dump()
        frame = pd.DataFrame([features], columns=_feature_columns)
        prediction = int(_model.predict(frame)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return {
        "prediction": prediction,
        "label": "fraud" if prediction == 1 else "not_fraud",
    }
