from pydantic import BaseModel, Field
from typing import Optional, List


class TransactionInput(BaseModel):
    transaction_id: str = Field(
        ...,
        description="Unique transaction identifier",
    )
    account_id: str = Field(
        ...,
        description="Customer account identifier",
    )
    subscription_id: str = Field(
        ...,
        description="Customer subscription identifier",
    )
    customer_id: str = Field(
        ...,
        description="Customer unique identifier",
    )
    provider_id: str = Field(
        ...,
        description="Source provider of the item",
    )
    product_id: str = Field(
        ...,
        description="Item being bought",
    )
    product_category: str = Field(
        ...,
        description="Product category",
    )
    channel_id: str = Field(
        ...,
        description="Customer channel",
    )
    amount: float = Field(
        ...,
        description="Transaction amount",
    )
    value: int = Field(
        ...,
        description="Absolute value of the amount",
    )
    transaction_start_time: str = Field(
        ...,
        description="Transaction timestamp (ISO format)",
    )
    pricing_strategy: int = Field(..., description="Pricing strategy category")


class PredictionRequest(BaseModel):
    transactions: List[TransactionInput] = Field(
        ..., description="List of customer transactions"
    )


class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float = Field(
        ...,
        description="Probability of default (0-1)",
    )
    credit_score: int = Field(
        ...,
        description="Credit score (300-850)",
    )
    risk_category: str = Field(
        ...,
        description="Risk category: Low, Medium, High",
    )
    recommended_loan_amount: Optional[float] = Field(
        None,
        description="Recommended max loan amount",
    )
    recommended_loan_duration_months: Optional[int] = Field(
        None,
        description="Recommended loan duration in months",
    )


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
