import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from app import config
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/fraud_model.pkl")
TARGET_COLUMN = "Class"
STAGING_STAGE = "Staging"
MODEL_TYPE = "LogisticRegression"
CLASS_WEIGHT = "balanced"
MAX_ITER = 200


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def run_training() -> None:
    """Train a fraud detection model on processed data and log with MLflow."""
    if not TRAIN_PATH.is_file():
        print(f"Error: train data not found at {TRAIN_PATH.resolve()}")
        print("Run the ingestion pipeline first: python -m app.pipelines.ingest")
        return
    if not TEST_PATH.is_file():
        print(f"Error: test data not found at {TEST_PATH.resolve()}")
        print("Run the ingestion pipeline first: python -m app.pipelines.ingest")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = _split_features_target(train_df)
    X_test, y_test = _split_features_target(test_df)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    model = LogisticRegression(
        class_weight=CLASS_WEIGHT,
        max_iter=MAX_ITER,
        random_state=42,
    )

    with mlflow.start_run(run_name="fraud-logreg-balanced") as run:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("class_weight", CLASS_WEIGHT)
        mlflow.log_param("max_iter", MAX_ITER)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.MODEL_NAME,
        )

        client = MlflowClient()
        model_version = model_info.registered_model_version
        client.transition_model_version_stage(
            name=config.MODEL_NAME,
            version=model_version,
            stage=STAGING_STAGE,
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_PATH.open("wb") as f:
            pickle.dump(model, f)

        print("Metrics summary:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"Model saved to: {MODEL_PATH.resolve()}")
        print(f"MLflow run id: {run.info.run_id}")
        print(f"Registered model name: {config.MODEL_NAME}")
        print(f"Model version: {model_version}")
        print(f"Model stage: {STAGING_STAGE}")


if __name__ == "__main__":
    run_training()
