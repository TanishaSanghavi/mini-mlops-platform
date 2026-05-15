# Mini MLOps Platform

A minimal project scaffold for a mini MLOps platform using FastAPI, scikit-learn, and MLflow.

## Project Structure

- `app/api/main.py`: FastAPI application entry point
- `app/pipelines/train.py`: Model training pipeline with MLflow logging
- `app/pipelines/ingest.py`: Data ingestion pipeline placeholder
- `app/models/model.py`: Model-related configuration
- `app/services/mlflow_service.py`: MLflow service helper
- `app/utils/logger.py`: Logging utility
- `data/raw/`: Raw datasets
- `data/processed/`: Processed datasets
- `experiments/`: Local MLflow tracking artifacts

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start API:
   - `uvicorn app.api.main:app --reload`
4. Run training pipeline:
   - `python app/pipelines/train.py`
