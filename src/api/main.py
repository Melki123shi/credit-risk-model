import logging
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException

import src.data_processing  # noqa: F401
from src.data_processing import (
    MissingValueImputer,
    FeatureEngineerTransformer,
    DataFrameOneHotEncoder,
    FeatureScaler,
    OutlierCapper,
    WoEFeatureTransformer,
)
from src.api.pydantic_models import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

# Register custom transformers under __main__ so joblib can unpickle them
_main = sys.modules["__main__"]
_main.MissingValueImputer = MissingValueImputer
_main.FeatureEngineerTransformer = FeatureEngineerTransformer
_main.DataFrameOneHotEncoder = DataFrameOneHotEncoder
_main.FeatureScaler = FeatureScaler
_main.OutlierCapper = OutlierCapper
_main.WoEFeatureTransformer = WoEFeatureTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "credit-risk-best-model"
MODEL_VERSION = 1
EXPERIMENT_NAME = "credit-risk-model"

model = None
pipeline = None


def _load_model():
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        logger.info(f"Trying Model Registry: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.warning(f"Model Registry failed: {e}")

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError("No runs found")
        run_id = runs.iloc[0]["run_id"]
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        model_artifact = None
        for artifact in artifacts:
            if artifact.path.endswith("_model") or artifact.path == "model":
                model_artifact = artifact.path
                break
        if model_artifact is None:
            raise ValueError("No model artifact found")
        model_uri = f"runs:/{run_id}/{model_artifact}"
        logger.info(f"Loading from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    return None


def load_artifacts():
    global model, pipeline

    model = _load_model()
    if model is not None:
        logger.info("Model loaded successfully.")
    else:
        logger.error("Model not available. Run 'python -m src.train' first.")

    try:
        pipeline_path = (
            Path(__file__).resolve().parent.parent.parent / "models" / "pipeline.joblib"
        )
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from: {pipeline_path}")
    except Exception as e:
        logger.warning(f"Pipeline not available: {e}")
        pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Real-time credit risk probability scoring for BNPL customers",
    version="1.0.0",
    lifespan=lifespan,
)


def probability_to_credit_score(probability: float) -> int:
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


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    customer_id = request.transactions[0].customer_id
    amounts = [t.amount for t in request.transactions]

    if pipeline is not None:
        raw_df = pd.DataFrame([
            {
                "TransactionId": t.transaction_id,
                "BatchId": "Batch_1",
                "AccountId": t.account_id,
                "SubscriptionId": t.subscription_id,
                "CustomerId": t.customer_id,
                "CurrencyCode": "UGX",
                "CountryCode": 256,
                "ProviderId": t.provider_id,
                "ProductId": t.product_id,
                "ProductCategory": t.product_category,
                "ChannelId": t.channel_id,
                "Amount": t.amount,
                "Value": abs(t.value),
                "TransactionStartTime": pd.to_datetime(
                    t.transaction_start_time, utc=True
                ),
                "PricingStrategy": t.pricing_strategy,
                "FraudResult": 0,
            }
            for t in request.transactions
        ])
        try:
            raw_df["is_high_risk"] = 0
            features_df = pipeline.transform(raw_df)
            cols_to_drop = [
                "is_high_risk", "TransactionId", "BatchId", "AccountId",
                "SubscriptionId", "CustomerId", "CurrencyCode",
                "TransactionStartTime", "FirstTransaction", "LastTransaction",
            ]
            cols_to_drop = [c for c in cols_to_drop if c in features_df.columns]
            features_df = features_df.drop(columns=cols_to_drop)
        except Exception as e:
            logger.error(f"Pipeline transform error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature transformation failed: {str(e)}",
            )
    else:
        feature_dict = {
            "TotalTransactionAmount": sum(abs(t.value) for t in request.transactions),
            "AvgTransactionAmount": sum(abs(t.value) for t in request.transactions)
            / len(request.transactions),
            "TransactionCount": len(request.transactions),
            "StdTransactionAmount": pd.Series(
                [abs(t.value) for t in request.transactions]
            ).std()
            if len(request.transactions) > 1
            else 0,
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
            "ProductCategory": request.transactions[0].product_category,
            "ChannelId": request.transactions[0].channel_id,
            "ProviderId": request.transactions[0].provider_id,
            "ProductId": request.transactions[0].product_id,
        }
        features_df = pd.DataFrame([feature_dict])

    try:
        proba = model.predict_proba(features_df)
        risk_prob = float(proba[0, 1])
    except Exception:
        try:
            risk_prob = float(model.predict(features_df)[0])
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {str(e)}"
            )

    credit_score = probability_to_credit_score(risk_prob)
    risk_category = get_risk_category(credit_score)
    avg_amount = sum(amounts) / len(amounts) if amounts else 0
    loan_amount, loan_duration = recommend_loan(
        avg_amount, credit_score, risk_prob
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
