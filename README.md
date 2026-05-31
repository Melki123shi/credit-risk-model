# Credit Risk Probability Model for Alternative Data

**An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model using Transaction Data**

---

## Overview

**Bati Bank** is partnering with a leading eCommerce platform to launch a **Buy-Now-Pay-Later (BNPL)** service. To enable responsible lending, we developed a **Credit Risk Probability Model** that uses alternative data (transaction behavior) to predict the likelihood of customer default.

This project demonstrates a complete **MLOps lifecycle** for building a production-grade credit scoring system using **RFM + Statistical + Machine Learning** techniques, aligned with **Basel II** principles and international best practices.

---

## Business Need

Traditional credit scoring relies on credit bureau data, which excludes millions of people (especially in emerging markets like Ethiopia). This project leverages **behavioral transaction data** to create an inclusive, alternative credit scoring solution.

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
Customers in the **lowest 20% quantile** of RFM score **or** showing high volatility + low frequency are labeled as **high risk (bad = 1)**. This approach is grounded in marketing and credit risk literature where behavioral patterns strongly correlate with repayment ability.

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

## Methodology

### 1. Feature Engineering (RFMS + Behavioral)
- Aggregated RFM features per customer
- Transaction frequency by category/channel
- Spending volatility (std, CV)
- Recency trends
- Pricing strategy interaction
- Temporal features (hour, day of week, tenure)

### 2. Target Variable Construction
Used unsupervised + rule-based approach combining RFM scoring and behavioral red flags.

### 3. Modeling Pipeline
- **Exploratory Data Analysis**
- **WoE / IV** feature selection
- **Scikit-learn Pipelines** + ColumnTransformer
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Hyperparameter tuning with Optuna
- Experiment tracking with **MLflow**

### 4. Credit Score Development
Converted predicted probabilities to a **300–850 scale** (similar to FICO-style):

```python
credit_score = 850 - (log_odds * scaling_factor + offset)
```
### 5. Loan Recommendation Model
Simple rule-based + regression model to suggest:

Optimal loan amount (based on average transaction value)
Recommended duration


#### **Project Structure**

Bashcredit-risk-model/
├── data/                    # Raw & processed data
├── notebooks/               # EDA and experimentation
├── src/
│   ├── data_processing.py
│   ├── features/
│   ├── models/
│   └── api/                 # FastAPI service
├── tests/
├── .github/workflows/       # CI/CD
├── Dockerfile
├── docker-compose.yml
├── MLproject
├── requirements.txt
└── README.md

## Deployment

- FastAPI REST API for real-time scoring
- Docker containerization
- GitHub Actions CI/CD pipeline
- Model serving via MLflow or direct API

API Endpoint Example:
```json
httpPOST /predict
{
  "customer_id": "Customer_123",
  "recent_transactions": [...]
}
```

## Technologies Used

- Python, pandas, scikit-learn, XGBoost, LightGBM
- MLflow (experiment tracking & registry)
- FastAPI + Pydantic
- Docker + Docker Compose
- GitHub Actions
- Optuna, SHAP, matplotlib/seaborn


Regulatory & Best Practices Alignment
This solution aligns with:

- World Bank Credit Scoring Guidelines (2019)
- Basel II Capital Accord
- RFMS Method for transaction data (Statistica Sinica, 2018)
- Principles of fairness, transparency, and explainability


### Future Improvements

- Integration with telco and utility data
- Time-series modeling (LSTM/Transformers)
- Active learning & human-in-the-loop feedback
- Bias detection and fairness metrics
- Champion-challenger model framework


