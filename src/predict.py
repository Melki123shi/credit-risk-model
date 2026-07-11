import logging
import math
import numpy as np
import pandas as pd
import mlflow.pyfunc
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "credit-risk-best-model"
MODEL_VERSION = 1


def load_model(model_name: str = MODEL_NAME, version: int = MODEL_VERSION):
    model_uri = f"models:/{model_name}/{version}"
    logger.info(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict_risk(model, features: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(features)[:, 1]
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
    std_amount = amounts.std() if len(amounts) > 1 else 0

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
        feature_cols = [c for c in df.columns if c != target_col]

        model = load_model()
        sample = df[feature_cols].head(5)

        results = predict_single(
            model,
            sample,
            transaction_amounts=df["TotalTransactionAmount"].head(5)
            if "TotalTransactionAmount" in df.columns
            else None,
        )
        logger.info(f"Sample prediction: {results}")
    else:
        logger.warning(f"Processed data not found at {data_path}")
