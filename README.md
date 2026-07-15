# Credit Risk Probability Model for Alternative Data

**An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model using Transaction Data**

---

## Overview

**Bati Bank** is partnering with a leading eCommerce platform to launch a **Buy-Now-Pay-Later (BNPL)** service. To enable responsible lending, we developed a **Credit Risk Probability Model** that uses alternative data (transaction behavior) to predict the likelihood of customer default.

This project demonstrates a complete **MLOps lifecycle** for building a production-grade credit scoring system using **RFM + Statistical + Machine Learning** techniques, aligned with **Basel II** principles and international best practices.

---

## Business Need

Traditional credit scoring relies on credit bureau data, which excludes millions of people (especially in emerging markets). This project leverages **behavioral transaction data** to create an inclusive, alternative credit scoring solution.

**Key Objectives:**
1. Define a **proxy default variable** using RFM analysis (since no direct default label exists).
2. Engineer predictive features from transaction history.
3. Build and compare interpretable vs. high-performance models.
4. Develop a **credit score** from default probability.
5. Deploy a real-time **REST API** for loan origination.
6. Ensure **regulatory compliance**, transparency, and auditability.

---

## Credit Scoring Business Understanding

### 1. Influence of Basel II Accord

The **Basel II Capital Accord** emphasizes three pillars:
- **Minimum Capital Requirements** (based on accurate PD - Probability of Default)
- **Supervisory Review**
- **Market Discipline** (transparency)

It requires banks to use **well-documented, validated, and interpretable** models for credit risk. This project prioritizes:
- Feature interpretability (WoE, IV, Logistic Regression coefficients)
- Model documentation and governance
- Back-testing readiness
- Clear separation of development and production environments

### 2. Why a Proxy Variable is Necessary

The Xente dataset contains no explicit default label. Following the **RFMS methodology** (Huang et al., 2018), we engineer a proxy target using customer behavior:

- **Recency (R)**: Time since last transaction
- **Frequency (F)**: Number of transactions
- **Monetary (M)**: Total spending
- **Standard Deviation (S)**: Volatility of spending

**Proxy Definition:**
Customers are clustered into 3 groups using K-Means on RFM features. The cluster with the **highest recency, lowest frequency, and lowest monetary value** is labeled as **high risk (bad = 1)**. This approach is grounded in marketing and credit risk literature where behavioral patterns strongly correlate with repayment ability.

**Business Risks of Proxy:**
- Potential misclassification (false positives/negatives)
- Need for continuous monitoring and periodic recalibration
- Must be validated against actual default data when available

### 3. Trade-offs: Interpretable vs. High-Performance Models

| Aspect                    | Logistic Regression (WoE)       | Gradient Boosting (XGBoost/LightGBM) |
|--------------------------|----------------------------------|--------------------------------------|
| **Interpretability**     | Excellent (regulatory friendly) | Low (black-box)                     |
| **Performance**          | Good                             | Superior                            |
| **Regulatory Acceptance**| High                             | Requires explainability techniques (SHAP, LIME) |
| **Implementation Speed** | Fast                             | Moderate                            |
| **Use Case**             | Scorecard, baseline              | Production ensemble                 |

**Our Approach**: Use **Logistic Regression with Weight of Evidence** as the primary model for regulatory compliance, while benchmarking against boosting models.

---

## Data Source

**Xente eCommerce Transaction Dataset** (Kaggle)

- **~95,000 transactions**
- **~3,700 unique customers**
- Rich behavioral signals: timestamps, amounts, categories, channels, etc.

**Key Fields Used:**
- `TransactionStartTime`, `Amount`, `Value`, `ProductCategory`, `ChannelId`, `CustomerId`, etc.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/
│   ├── raw/                       # Raw data (credit_data.csv)
│   └── processed/                 # Processed data for training
├── notebooks/
│   ├── eda.ipynb                  # Exploratory analysis
│   ├── feature_engineering.ipynb  # Feature engineering exploration
│   └── predict.ipynb              # Prediction & scoring demo
├── models/
│   └── pipeline.joblib            # Fitted sklearn pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data loading utilities
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── preprocessing.py           # EDA preprocessing utilities
│   ├── train.py                   # Model training with MLflow
│   ├── predict.py                 # Inference utilities
│   └── api/
│       ├── __init__.py
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### 1. Feature Engineering (RFMS + Behavioral)
- Aggregated RFM features per customer
- Transaction frequency by category/channel
- Spending volatility (std, CV)
- Recency trends
- Pricing strategy interaction
- Temporal features (hour, day of week, tenure)
- Weight of Evidence (WoE) encoding for categoricals
- Information Value (IV) for feature selection

### 2. Target Variable Construction
Used unsupervised + rule-based approach:
1. Calculate RFM metrics per customer
2. Scale features with StandardScaler
3. Cluster into 3 groups with K-Means (random_state=42)
4. Identify the least-engaged cluster (high R, low F, low M)
5. Label as `is_high_risk = 1`, others as `0`

### 3. Modeling Pipeline
- **Exploratory Data Analysis** (notebooks/eda.ipynb)
- **WoE / IV** feature selection
- **Scikit-learn Pipelines** for reproducible transforms
- Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
- Experiment tracking with **MLflow**
- Best model registered in MLflow Model Registry

### 4. Credit Score Development
Converted predicted probabilities to a **300-850 scale** (FICO-style):

```python
log_odds = log(probability / (1 - probability))
credit_score = 425 - (log_odds * 50)
credit_score = max(300, min(850, credit_score))
```

### 5. Loan Recommendation Model
Rule-based logic based on credit score and risk probability:

| Risk Level      | Loan Multiplier | Duration |
|-----------------|----------------|----------|
| Low (score >= 700) | 2x avg transaction | 12 months |
| Medium (score >= 600) | 1.5x | 6 months |
| Moderate (score >= 500) | 1.0x | 3 months |
| High (score < 500) | 0.5x | 2 months |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for deployment)
- MLflow (for experiment tracking)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/credit-risk-model.git
cd credit-risk-model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Processing

```bash
# Generate processed data (run from project root)
python -m src.data_processing
```

This will:
1. Load raw data from `data/raw/data.csv`
2. Create proxy target via RFM + KMeans
3. Run the full feature pipeline
4. Save to `data/processed/processed_data.csv`

### Model Training

```bash
# Train all models with MLflow tracking
python -m src.train
```

This will:
1. Load processed data
2. Train 5 models (LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM)
3. Log all experiments to MLflow
4. Register the best model as `credit-risk-best-model`

View experiments: `mlflow ui`

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| LogisticRegression | 0.9985 | 0.9937 | 0.9932 | 0.9935 | 1.0000 |
| RandomForest | 0.9998 | 1.0000 | 0.9986 | 0.9993 | 1.0000 |
| GradientBoosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

> **Note:** Near-perfect scores are expected since the proxy target is derived from the same RFM features used for prediction. This is a known limitation of proxy-based approaches — the target is not independent of the features. In production, the model should be validated against actual default data when available.

### Running Tests

```bash
pytest tests/ -v
```

### API Deployment

```bash
# Run locally
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or with Docker Compose
docker-compose up --build
```

API docs: `http://localhost:8000/docs`

### API Endpoints

**Health Check:**
```bash
GET /health
```

**Predict Credit Risk:**
```bash
POST /predict
Content-Type: application/json

{
  "transactions": [
    {
      "transaction_id": "T123",
      "account_id": "A456",
      "subscription_id": "S789",
      "customer_id": "Customer_123",
      "provider_id": "ProviderId_1",
      "product_id": "ProductId_10",
      "product_category": "Financial Services",
      "channel_id": "ChannelId_4",
      "amount": 5000.0,
      "value": 5000,
      "transaction_start_time": "2019-01-01T12:00:00Z",
      "pricing_strategy": 1
    }
  ]
}
```

**Response:**
```json
{
  "customer_id": "Customer_123",
  "risk_probability": 0.2341,
  "credit_score": 712,
  "risk_category": "Low",
  "recommended_loan_amount": 10000.0,
  "recommended_loan_duration_months": 12
}
```

---

## CI/CD Pipeline

GitHub Actions workflow triggers on push to `main` and `task-*` branches:
1. **Lint**: flake8 code style checks
2. **Test**: pytest unit tests

Both must pass for the build to succeed.

---

## Technologies Used

- **Python 3.11** - Core language
- **pandas / numpy** - Data manipulation
- **scikit-learn** - ML pipelines, preprocessing, models
- **XGBoost / LightGBM** - Gradient boosting models
- **MLflow** - Experiment tracking & model registry
- **FastAPI + Pydantic** - REST API
- **Docker + Docker Compose** - Containerization
- **GitHub Actions** - CI/CD
- **pytest** - Unit testing
- **flake8** - Code linting
- **matplotlib / seaborn** - Visualization

---

## Regulatory & Best Practices Alignment

This solution aligns with:
- World Bank Credit Scoring Guidelines (2019)
- Basel II Capital Accord
- RFMS Method for transaction data (Statistica Sinica, 2018)
- Principles of fairness, transparency, and explainability

---

## Future Improvements

- Integration with telco and utility data
- Time-series modeling (LSTM/Transformers)
- Active learning & human-in-the-loop feedback
- Bias detection and fairness metrics
- Champion-challenger model framework
- SHAP-based model explainability for regulatory audits
- Automated model retraining pipeline
