import logging
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Embedded WoE/IV utilities into data_processing module")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering pipeline."""
    df = df.copy()

    # 1. Temporal Features
    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year
    df["TransactionWeekday"] = df["TransactionStartTime"].dt.weekday

    # 2. Customer-Level Aggregate Features (RFM will be added later)
    customer_agg = (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionAmount=("Amount", "sum"),
            AvgTransactionAmount=("Amount", "mean"),
            TransactionCount=("TransactionId", "count"),
            StdTransactionAmount=("Amount", "std"),
            FirstTransaction=("TransactionStartTime", "min"),
            LastTransaction=("TransactionStartTime", "max"),
        )
        .reset_index()
    )

    now_utc = pd.Timestamp.now(tz="UTC")
    customer_agg["Recency"] = (now_utc - customer_agg["LastTransaction"]).dt.days
    customer_agg["CustomerTenureDays"] = (
        customer_agg["LastTransaction"] - customer_agg["FirstTransaction"]
    ).dt.days

    df = df.merge(customer_agg, on="CustomerId", how="left")

    # 3. Handle missing values in aggregates
    df["StdTransactionAmount"] = df["StdTransactionAmount"].fillna(0)

    return df


class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self, numerical_strategy="median", categorical_strategy="most_frequent"
    ):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy

    def fit(self, X, y=None):
        X = X.copy()

        self.num_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=["number"]).columns.tolist()

        self.num_imputer_ = SimpleImputer(strategy=self.numerical_strategy)
        self.cat_imputer_ = SimpleImputer(strategy=self.categorical_strategy)

        if self.num_cols_:
            self.num_imputer_.fit(X[self.num_cols_])

        if self.cat_cols_:
            self.cat_imputer_.fit(X[self.cat_cols_])

        return self

    def transform(self, X):
        X = X.copy()

        if self.num_cols_:
            X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])

        if self.cat_cols_:
            X[self.cat_cols_] = self.cat_imputer_.transform(X[self.cat_cols_])

        return X


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X.copy())


def split_data(
    df: pd.DataFrame,
    target_col: str = "is_high_risk",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Parameters:
    - df: Full processed DataFrame with target column
    - target_col: Name of the target variable
    - test_size: Proportion of data for test set (default 20%)
    - val_size: Proportion of data for validation set (default 10% of total)
    - random_state: Random seed for reproducibility
    - stratify: Whether to stratify splits based on target
        (recommended for imbalanced data)

    Returns:
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: Train + Temp (val + test)
    stratify_param = y if stratify else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=stratify_param,
    )

    # Second split: Validation vs Test
    # val_ratio = val_size / (val_size + test_size)  # e.g., 0.1 / 0.3 = 1/3

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        stratify=y_temp if stratify else None,
    )

    print("Split completed:")
    n = len(df)
    print(f"   Train: {X_train.shape[0]:,} records ({X_train.shape[0] / n:.1%})")
    print(f"   Val:   {X_val.shape[0]:,} records ({X_val.shape[0] / n:.1%})")
    print(f"   Test:  {X_test.shape[0]:,} records ({X_test.shape[0] / n:.1%})")
    train_dist = y_train.value_counts(normalize=True).round(3).to_dict()
    print(f"   Target distribution (Train): {train_dist}")

    return X_train, X_val, X_test, y_train, y_val, y_test


class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_cols: List[str],
        sparse_output: bool = False,
        handle_unknown: str = "ignore",
    ):
        self.categorical_cols = categorical_cols
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown

    def fit(self, X: pd.DataFrame, y=None):
        allowed = self.categorical_cols or X.columns
        present_cols = [col for col in X.columns if col in allowed]
        self.categorical_cols_ = [
            col
            for col in present_cols
            if (
                not pd.api.types.is_numeric_dtype(X[col])
                and col != self.categorical_cols
            )
        ]
        self.encoder_ = OneHotEncoder(
            sparse_output=self.sparse_output,
            handle_unknown=self.handle_unknown,
        )
        self.encoder_.fit(X[self.categorical_cols_])
        self.feature_names_ = self.encoder_.get_feature_names_out(
            self.categorical_cols_
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded = self.encoder_.transform(X[self.categorical_cols_])
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names_, index=X.index)
        X = X.drop(columns=self.categorical_cols_)
        X = pd.concat([X, encoded_df], axis=1)
        return X


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to scale numerical features.
    Supports Standardization (StandardScaler) and Normalization (MinMaxScaler).
    """

    def __init__(
        self,
        numerical_cols: List[str],
        method: str = "standard",  # "standard" or "minmax"
        target_col: str = "is_high_risk",
    ):
        self.numerical_cols = numerical_cols
        self.method = method.lower()
        self.target_col = target_col

        if self.method not in ["standard", "minmax"]:
            raise ValueError("method must be 'standard' or 'minmax'")

    def fit(self, X: pd.DataFrame, y=None):
        # Determine numerical columns
        if self.numerical_cols is None:
            self.numerical_cols_ = [
                col
                for col in X.columns
                if (pd.api.types.is_numeric_dtype(X[col]) and col != self.target_col)
            ]
        else:
            self.numerical_cols_ = [
                col for col in self.numerical_cols if col in X.columns
            ]

        # Choose scaler
        if self.method == "standard":
            self.scaler_ = StandardScaler()
        else:
            self.scaler_ = MinMaxScaler()

        if self.numerical_cols_:
            self.scaler_.fit(X[self.numerical_cols_])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.numerical_cols_:
            scaled = self.scaler_.transform(X[self.numerical_cols_])
            X[self.numerical_cols_] = scaled
        return X


# ----------------------------- HELPER FUNCTION -----------------------------
def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: List[str],
    method: str = "standard",  # "standard" or "minmax"
    target_col: str = "is_high_risk",
) -> tuple:
    """
    Scale numerical features consistently across train/val/test.
    """
    scaler_transformer = FeatureScaler(
        numerical_cols=num_cols, method=method, target_col=target_col
    )

    scaler_transformer.fit(X_train)

    X_train_scaled = scaler_transformer.transform(X_train)
    X_val_scaled = scaler_transformer.transform(X_val)
    X_test_scaled = scaler_transformer.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler_transformer.scaler_,
    )


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols: List[str], iqr_multiplier: float = 1.5):
        self.numerical_cols = numerical_cols
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.cols_ = self.numerical_cols or [
            c for c in self.numerical_cols if c in X.columns
        ]
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.cols_:
            if col not in X.columns:
                continue
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.iqr_multiplier * iqr
            self.upper_bounds_[col] = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols_:
            if col not in X.columns or col not in self.lower_bounds_:
                continue
            X[col] = X[col].clip(
                lower=self.lower_bounds_[col], upper=self.upper_bounds_[col]
            )
        return X


def cap_outliers(
    df: pd.DataFrame, numerical_cols: List[str] = None, iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    # Cap outliers at IQR fences (Winsorization) using OutlierCapper."""
    capper = OutlierCapper(numerical_cols=numerical_cols, iqr_multiplier=iqr_multiplier)
    return capper.fit_transform(df)


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, float]:
    """Compute WoE and IV for a single feature."""
    total_events = (df[target] == 1).sum()
    total_non_events = (df[target] == 0).sum()

    stats = (
        df.groupby(feature, observed=False)[target]
        .agg(
            events=lambda x: (x == 1).sum(),
            non_events=lambda x: (x == 0).sum(),
        )
        .reset_index()
    )

    stats["dist_events"] = (stats["events"] + epsilon) / (total_events + epsilon)
    stats["dist_non_events"] = (stats["non_events"] + epsilon) / (
        total_non_events + epsilon
    )
    stats["woe"] = np.log(stats["dist_events"] / stats["dist_non_events"])
    dist_diff = stats["dist_events"] - stats["dist_non_events"]
    stats["iv_component"] = dist_diff * stats["woe"]

    iv_total = stats["iv_component"].sum()
    return stats, iv_total


def _iv_label(iv: float) -> str:
    if iv < 0.02:
        return "Useless"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    elif iv < 0.5:
        return "Strong"
    else:
        return "Very Strong"


def compute_all_iv(
    df: pd.DataFrame,
    features: List[str],
    categorical_cols,
    numerical_cols,
    target: str,
) -> pd.DataFrame:
    """Compute IV for multiple features."""
    if features is None:
        features = categorical_cols + numerical_cols

    results = []
    for feat in features:
        if feat not in df.columns or feat == target:
            continue

        temp_df = df[[feat, target]].copy()

        # Bin numerical features
        if feat in numerical_cols:
            try:
                temp_df[feat] = pd.qcut(df[feat], q=10, duplicates="drop")
            except Exception:
                temp_df[feat] = pd.cut(df[feat], bins=5)

        _, iv_value = compute_woe_iv(temp_df, feat, target)
        results.append({"feature": feat, "iv": iv_value})

    iv_df = (
        pd.DataFrame(results).sort_values("iv", ascending=False).reset_index(drop=True)
    )
    iv_df["predictive_power"] = iv_df["iv"].apply(_iv_label)
    return iv_df


class WoEFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer to apply Weight of Evidence encoding."""

    def __init__(self, target_col: str, categorical_cols: List[str]):
        self.categorical_cols = categorical_cols
        self.target_col = target_col

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if y is not None:
            X[self.target_col] = y

        # Determine categorical columns
        if self.categorical_cols is None:
            self.categorical_cols_ = [
                col
                for col in X.columns
                if (
                    col != self.target_col and not pd.api.types.is_numeric_dtype(X[col])
                )
            ]
        else:
            self.categorical_cols_ = [
                col for col in self.categorical_cols if col in X.columns
            ]

        self.woe_maps_: Dict[str, Dict] = {}

        for col in self.categorical_cols_:
            if col not in X.columns:
                continue
            subset = X[[col, self.target_col]]
            woe_df, _ = compute_woe_iv(subset, col, self.target_col)
            self.woe_maps_[col] = dict(zip(woe_df[col], woe_df["woe"]))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.woe_maps_.items():
            if col in X.columns:
                X[f"{col}_woe"] = X[col].map(mapping).fillna(0.0)
        return X


def calculate_rfm(
    df: pd.DataFrame,
    customer_id_col: str = "CustomerId",
    transaction_date_col: str = "TransactionStartTime",
    amount_col: str = "Amount",
    snapshot_date: str = None,
) -> pd.DataFrame:
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[transaction_date_col]):
        df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])

    # FIXED: Handle snapshot_date with timezone
    if snapshot_date is None:
        snapshot_date = pd.Timestamp.now(tz="UTC")
    else:
        snapshot_date = pd.to_datetime(snapshot_date)
        if snapshot_date.tz is None:
            snapshot_date = snapshot_date.tz_localize("UTC")

    rfm = (
        df.groupby(customer_id_col)
        .agg(
            Recency=(transaction_date_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(customer_id_col, "count"),
            Monetary=(amount_col, "sum"),
        )
        .reset_index()
    )

    # Additional metrics
    rfm["AvgMonetary"] = df.groupby(customer_id_col)[amount_col].mean().values
    rfm["FirstTransaction"] = (
        df.groupby(customer_id_col)[transaction_date_col].min().values
    )
    rfm["LastTransaction"] = (
        df.groupby(customer_id_col)[transaction_date_col].max().values
    )

    return rfm


def create_rfm_features(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Calculate RFM features and return a DataFrame with CustomerId and RFM metrics.
    """
    rfm = calculate_rfm(
        df,
        customer_id_col="CustomerId",
        transaction_date_col="TransactionStartTime",
        amount_col="Amount",
        snapshot_date=snapshot_date,
    )
    return rfm


def visualize_and_cluster_kmeans(
    X_scaled: np.ndarray,
    original_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fits KMeans, plots the clusters across RFM dimensions, and returns cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Temporary dataframe for plotting
    plot_df = original_df.copy()
    plot_df["Cluster"] = cluster_labels

    # Set style
    sns.set_theme(style="whitegrid")

    # Create a 1x2 visualization (Distribution and Scatters)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Scatter plot: Recency vs Monetary
    sns.scatterplot(
        data=plot_df,
        x="Recency",
        y="Monetary",
        hue="Cluster",
        palette="viridis",
        ax=axes[0],
    )
    axes[0].set_title("Clusters: Recency vs Monetary")

    # 2. Boxplot: Monetary distribution by Cluster
    sns.boxplot(data=plot_df, x="Cluster", y="Monetary", palette="viridis", ax=axes[1])
    axes[1].set_title("Monetary Value Distribution by Cluster")

    plt.tight_layout()
    plt.show()

    # Also plot the pairplot for a more detailed view
    sns.pairplot(
        plot_df,
        vars=["Recency", "Frequency", "Monetary"],
        hue="Cluster",
        palette="viridis",
    )
    plt.suptitle("RFM Cluster Pairplot", y=1.02)
    plt.show()

    return cluster_labels


def assign_high_risk_label(
    rfm_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42
) -> pd.DataFrame:
    """
    Use KMeans to cluster customers based on RFM and assign is_high_risk label.
    """
    rfm = rfm_df.copy()

    # Select features for clustering
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].copy()

    # Handle any potential NaNs (though calculate_rfm should avoid them)
    X = X.fillna(X.median())

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    # Analyze clusters to identify high-risk (least engaged)
    cluster_summary = (
        rfm.groupby("Cluster")
        .agg(
            {
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean",
                "CustomerId": "count",
            }
        )
        .round(2)
    )

    print("Cluster Summary (for high-risk identification):")
    print(cluster_summary)
    visualize_and_cluster_kmeans(
        X_scaled, rfm, n_clusters=n_clusters, random_state=random_state
    )

    # High-risk cluster = highest Recency + lowest Frequency + lowest Monetary
    # Usually the cluster with highest Recency and lowest Frequency/Monetary
    risk_cluster = cluster_summary.sort_values(
        by=["Recency", "Frequency", "Monetary"], ascending=[False, True, True]
    ).index[0]

    print(f"High-risk cluster identified: {risk_cluster}")

    # Assign target
    rfm["is_high_risk"] = (rfm["Cluster"] == risk_cluster).astype(int)

    # Keep only needed columns
    rfm = rfm[["CustomerId", "Recency", "Frequency", "Monetary", "is_high_risk"]]

    return rfm


def create_proxy_target(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Main function to create is_high_risk target using RFM + KMeans.
    Returns the original DataFrame with the new target column added.
    """
    df = df.copy()

    # 1. Calculate RFM
    rfm_df = create_rfm_features(df, snapshot_date=snapshot_date)

    # 2. Assign high-risk label
    rfm_with_target = assign_high_risk_label(rfm_df)

    # 3. Merge back to original DataFrame
    df = df.merge(
        rfm_with_target[["CustomerId", "is_high_risk"]], on="CustomerId", how="left"
    )

    print(
        f"Proxy target created: {df['is_high_risk'].value_counts(normalize=True).round(3)}"
    )
    print(f"High-risk customers: {df['is_high_risk'].sum():,} out of {len(df):,}")

    return df


def build_feature_pipeline(
    target_col: str = "is_high_risk",
    scaling_method: str = "standard",
    categorical_cols: List[str] = None,
    numerical_cols: List[str] = None,
):
    """
    Returns a fitted sklearn Pipeline capable of transforming
    raw transaction data into model-ready features.
    """

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineerTransformer()),
            (
                "missing_values",
                MissingValueImputer(
                    numerical_strategy="median", categorical_strategy="most_frequent"
                ),
            ),
            ("outlier_capping", OutlierCapper(numerical_cols=numerical_cols)),
            (
                "woe_encoding",
                WoEFeatureTransformer(
                    categorical_cols=categorical_cols, target_col=target_col
                ),
            ),
            (
                "one_hot_encoding",
                DataFrameOneHotEncoder(categorical_cols=categorical_cols),
            ),
            (
                "scaling",
                FeatureScaler(
                    numerical_cols=numerical_cols,
                    method=scaling_method,
                    target_col=target_col,
                ),
            ),
        ]
    )

    return pipeline
