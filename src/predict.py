import logging
import math
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.pyfunc
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "credit-risk-best-model"
MODEL_VERSION = 1
RANDOM_STATE = 42

COLS_TO_DROP = [
    "TransactionId", "BatchId", "AccountId", "SubscriptionId",
    "CustomerId", "CurrencyCode", "TransactionStartTime",
    "FirstTransaction", "LastTransaction",
]


def load_model(
    model_name: str = MODEL_NAME,
    version: int = MODEL_VERSION,
    experiment_name: str = "credit-risk-model",
):
    # 1. Try loading from the Model Registry
    try:
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Trying Model Registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded from Model Registry.")
        return model
    except Exception as e:
        logger.warning(f"Model Registry load failed: {e}")

    # 2. Fall back to the latest run in the experiment
    try:
        logger.info(f"Trying latest run from experiment: {experiment_name}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError("No runs found in experiment")

        run_id = runs.iloc[0]["run_id"]
        logger.info(f"Latest run: {run_id}")

        # Try to find a model artifact in the run
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        model_artifact = None
        for artifact in artifacts:
            if artifact.path.endswith("_model") or artifact.path == "model":
                model_artifact = artifact.path
                break

        if model_artifact is None:
            raise ValueError("No model artifact found in latest run")

        model_uri = f"runs:/{run_id}/{model_artifact}"
        logger.info(f"Loading from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded from latest MLflow run.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from runs: {e}")

    raise RuntimeError(
        "Could not load model. Run 'python -m src.train' first to train "
        "and register the model."
    )


def load_pipeline(pipeline_path: Optional[str] = None):
    if pipeline_path is None:
        project_root = Path(__file__).resolve().parent.parent
        pipeline_path = project_root / "models" / "pipeline.joblib"
    logger.info(f"Loading pipeline from: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    return pipeline


def transform_raw_input(
    raw_df: pd.DataFrame, pipeline
) -> pd.DataFrame:
    """Transform raw transaction DataFrame through the fitted pipeline."""
    transformed = pipeline.transform(raw_df)
    return transformed


def predict_risk(model, features: pd.DataFrame) -> np.ndarray:
    try:
        proba = model.predict_proba(features)[:, 1]
    except Exception:
        prediction = model.predict(features)
        proba = prediction.astype(float)
    return proba


def probability_to_credit_score(probability: float) -> int:
    if probability <= 0 or probability >= 1:
        probability = max(0.001, min(0.999, probability))

    log_odds = math.log(probability / (1 - probability))

    min_score = 300
    max_score = 850
    midpoint = (min_score + max_score) / 2
    scaling_factor = 50

    score = midpoint - (log_odds * scaling_factor)
    score = max(min_score, min(max_score, score))
    return int(round(score))


def recommend_loan(amounts: pd.Series, credit_score: int, risk_prob: float):
    avg_amount = amounts.mean()

    if credit_score >= 700 and risk_prob < 0.3:
        loan_multiplier = 2.0
        duration_months = 12
    elif credit_score >= 600 and risk_prob < 0.5:
        loan_multiplier = 1.5
        duration_months = 6
    elif credit_score >= 500:
        loan_multiplier = 1.0
        duration_months = 3
    else:
        loan_multiplier = 0.5
        duration_months = 2

    max_loan = avg_amount * loan_multiplier
    return round(max_loan, 2), duration_months


def predict_single(
    model,
    features: pd.DataFrame,
    transaction_amounts: pd.Series = None,
):
    risk_prob = predict_risk(model, features)[0]
    credit_score = probability_to_credit_score(risk_prob)

    if transaction_amounts is not None and len(transaction_amounts) > 0:
        loan_amount, loan_duration = recommend_loan(
            transaction_amounts, credit_score, risk_prob
        )
    else:
        loan_amount, loan_duration = None, None

    return {
        "risk_probability": round(float(risk_prob), 4),
        "credit_score": credit_score,
        "recommended_loan_amount": loan_amount,
        "recommended_loan_duration_months": loan_duration,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "processed_data.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        target_col = "is_high_risk"

        if target_col in df.columns:
            df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
            feature_cols = [c for c in df.columns if c != target_col]
        else:
            feature_cols = list(df.columns)

        model = load_model()
        sample = df[feature_cols].head(5)

        amounts_col = (
            "TotalTransactionAmount"
            if "TotalTransactionAmount" in df.columns
            else None
        )
        results = predict_single(
            model,
            sample,
            transaction_amounts=df[amounts_col].head(5)
            if amounts_col
            else None,
        )
        logger.info(f"Sample prediction: {results}")
    else:
        logger.warning(f"Processed data not found at {data_path}")
