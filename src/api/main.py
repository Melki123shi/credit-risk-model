import logging
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Real-time credit risk probability scoring for BNPL customers",
    version="1.0.0",
)

MODEL_NAME = "credit-risk-best-model"
MODEL_VERSION = 1
model = None


def load_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded: {MODEL_NAME} v{MODEL_VERSION}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
    )


def probability_to_credit_score(probability: float) -> int:
    import math
    probability = max(0.001, min(0.999, probability))
    log_odds = math.log(probability / (1 - probability))
    score = 425 - (log_odds * 50)
    return max(300, min(850, int(round(score))))


def get_risk_category(score: int) -> str:
    if score >= 700:
        return "Low"
    elif score >= 500:
        return "Medium"
    else:
        return "High"


def recommend_loan(avg_amount: float, credit_score: int, risk_prob: float):
    if credit_score >= 700 and risk_prob < 0.3:
        return round(avg_amount * 2.0, 2), 12
    elif credit_score >= 600 and risk_prob < 0.5:
        return round(avg_amount * 1.5, 2), 6
    elif credit_score >= 500:
        return round(avg_amount * 1.0, 2), 3
    else:
        return round(avg_amount * 0.5, 2), 2


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    customer_id = request.transactions[0].customer_id
    amounts = [t.amount for t in request.transactions]
    categories = [t.product_category for t in request.transactions]
    channels = [t.channel_id for t in request.transactions]
    providers = [t.provider_id for t in request.transactions]
    products = [t.product_id for t in request.transactions]

    feature_dict = {
        "TotalTransactionAmount": sum(amounts),
        "AvgTransactionAmount": sum(amounts) / len(amounts),
        "TransactionCount": len(amounts),
        "StdTransactionAmount": pd.Series(amounts).std() if len(amounts) > 1 else 0,
        "Recency": 0,
        "CustomerTenureDays": 0,
        "TransactionHour": 12,
        "TransactionDay": 15,
        "TransactionMonth": 1,
        "TransactionYear": 2019,
        "TransactionWeekday": 2,
        "CountryCode": 256,
        "PricingStrategy": request.transactions[0].pricing_strategy,
        "Value": sum(abs(t.value) for t in request.transactions),
        "FraudResult": 0,
        "ProductCategory": categories[0],
        "ChannelId": channels[0],
        "ProviderId": providers[0],
        "ProductId": products[0],
        "CurrencyCode": "UGX",
        "BatchId": "Batch_1",
        "AccountId": request.transactions[0].account_id,
        "SubscriptionId": request.transactions[0].subscription_id,
    }

    features_df = pd.DataFrame([feature_dict])

    try:
        prediction = model.predict(features_df)
        proba = model.predict_proba(features_df)[:, 1]
        risk_prob = float(proba[0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    credit_score = probability_to_credit_score(risk_prob)
    risk_category = get_risk_category(credit_score)
    loan_amount, loan_duration = recommend_loan(
        sum(amounts) / len(amounts), credit_score, risk_prob
    )

    return PredictionResponse(
        customer_id=customer_id,
        risk_probability=round(risk_prob, 4),
        credit_score=credit_score,
        risk_category=risk_category,
        recommended_loan_amount=loan_amount,
        recommended_loan_duration_months=loan_duration,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
