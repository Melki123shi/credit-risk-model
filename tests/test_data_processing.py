import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_processing import (  # noqa: E402
    engineer_features,
    compute_woe_iv,
    _iv_label,
    calculate_rfm,
    assign_high_risk_label,
    MissingValueImputer,
    FeatureEngineerTransformer,
    OutlierCapper,
)


@pytest.fixture
def sample_transactions():
    dates = pd.date_range("2018-11-01", periods=100, freq="h", tz="UTC")
    return pd.DataFrame({
        "TransactionId": [f"T{i}" for i in range(100)],
        "BatchId": [f"B{i // 10}" for i in range(100)],
        "AccountId": [f"A{i % 5}" for i in range(100)],
        "SubscriptionId": [f"S{i % 5}" for i in range(100)],
        "CustomerId": [f"C{i % 3}" for i in range(100)],
        "CurrencyCode": ["UGX"] * 100,
        "CountryCode": [256] * 100,
        "ProviderId": [f"P{i % 4}" for i in range(100)],
        "ProductId": [f"Pr{i % 6}" for i in range(100)],
        "ProductCategory": [
            "Financial Services", "Airtime", "Utility Bill",
            "Data Bundle", "TV", "Movie",
        ] * 16 + ["Financial Services"] * 4,
        "ChannelId": ["ChannelId_4", "ChannelId_3", "ChannelId_2"] * 33 + ["ChannelId_4"],
        "Amount": np.random.uniform(-1000, 5000, 100).round(2),
        "Value": np.random.randint(100, 5000, 100),
        "TransactionStartTime": dates,
        "PricingStrategy": [1, 2, 3, 4] * 25,
        "FraudResult": [0] * 95 + [1] * 5,
    })


class TestEngineerFeatures:
    def test_returns_expected_columns(self, sample_transactions):
        result = engineer_features(sample_transactions)

        expected_new_cols = [
            "TransactionHour", "TransactionDay", "TransactionMonth",
            "TransactionYear", "TransactionWeekday",
            "TotalTransactionAmount", "AvgTransactionAmount",
            "TransactionCount", "StdTransactionAmount",
            "Recency", "CustomerTenureDays",
        ]
        for col in expected_new_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_preserves_original_columns(self, sample_transactions):
        original_cols = set(sample_transactions.columns)
        result = engineer_features(sample_transactions)
        for col in original_cols:
            assert col in result.columns, f"Original column lost: {col}"

    def test_customer_aggregates_correct(self, sample_transactions):
        result = engineer_features(sample_transactions)
        cust_c0 = result[result["CustomerId"] == "C0"]
        assert cust_c0["TransactionCount"].nunique() == 1
        assert cust_c0["TotalTransactionAmount"].iloc[0] == pytest.approx(
            sample_transactions[
                sample_transactions["CustomerId"] == "C0"
            ]["Value"].sum(),
            rel=1e-6,
        )


class TestWoEIV:
    def test_compute_woe_iv_returns_tuple(self, sample_transactions):
        sample_transactions["target"] = (
            sample_transactions["FraudResult"].astype(int)
        )
        stats, iv = compute_woe_iv(
            sample_transactions, "ProductCategory", "target"
        )
        assert isinstance(stats, pd.DataFrame)
        assert isinstance(iv, float)
        assert "woe" in stats.columns
        assert "iv_component" in stats.columns

    def test_iv_label_mapping(self):
        assert _iv_label(0.01) == "Useless"
        assert _iv_label(0.05) == "Weak"
        assert _iv_label(0.15) == "Medium"
        assert _iv_label(0.35) == "Strong"
        assert _iv_label(0.6) == "Very Strong"


class TestCalculateRFM:
    def test_rfm_output_columns(self, sample_transactions):
        rfm = calculate_rfm(sample_transactions)
        assert "CustomerId" in rfm.columns
        assert "Recency" in rfm.columns
        assert "Frequency" in rfm.columns
        assert "Monetary" in rfm.columns
        assert len(rfm) == sample_transactions["CustomerId"].nunique()

    def test_rfm_values(self, sample_transactions):
        rfm = calculate_rfm(sample_transactions)
        cust_c0 = rfm[rfm["CustomerId"] == "C0"]
        expected_count = (sample_transactions["CustomerId"] == "C0").sum()
        assert cust_c0["Frequency"].values[0] == expected_count


class TestAssignHighRisk:
    def test_assigns_binary_target(self, sample_transactions):
        rfm = calculate_rfm(sample_transactions)
        result = assign_high_risk_label(rfm, n_clusters=3, random_state=42)
        assert "is_high_risk" in result.columns
        assert set(result["is_high_risk"].unique()).issubset({0, 1})


class TestMissingValueImputer:
    def test_imputes_numerical_median(self):
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": ["x", "y", "x", "y"]})
        imputer = MissingValueImputer()
        result = imputer.fit_transform(df)
        assert not result.isnull().any().any()
        assert result["a"].iloc[2] == 2.0


class TestFeatureEngineerTransformer:
    def test_sklearn_transformer_interface(self, sample_transactions):
        transformer = FeatureEngineerTransformer()
        transformer.fit(sample_transactions)
        result = transformer.transform(sample_transactions)
        assert "TransactionHour" in result.columns
        assert "TotalTransactionAmount" in result.columns


class TestOutlierCapper:
    def test_caps_outliers(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
        capper = OutlierCapper(numerical_cols=["a"])
        result = capper.fit_transform(df)
        assert result["a"].max() < 100
        assert result["a"].min() >= 1
