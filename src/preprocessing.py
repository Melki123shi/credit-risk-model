"""Basic preprocessing utilities for exploratory data analysis.

Provides a data_summary function to compute central tendency, dispersion,
and shape statistics for pandas Series or DataFrame.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os

sns.set_theme(style="whitegrid")


def df_summary(df: pd.DataFrame) -> None:
    """
    Prints a summary of a DataFrame including dimensions,
    data types, and a preview of the data.
    """
    print("--- DATASET OVERVIEW ---")

    rows, cols = df.shape
    print(f"Dimensions: {rows} rows, {cols} columns\n")

    print("--- Column Data Types ---")
    print(df.dtypes)

    print("\n--- Column Information ---")
    print(df.info())

    print("\n---- Data Summary Statistics ----\n")
    print(df.describe())

    print("\n--- Missing Values Summary ---\n")
    print(df.isnull().sum())

    print("\n--- Data Preview (First 5 Rows) ---")
    print(df.head())


def analyze_distribution(df):
    """
    Calculates and displays central tendency, dispersion,
    and shape of the dataset's distribution.
    """
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        return "No numerical columns found in the dataset."

    # 1. Central Tendency
    stats = pd.DataFrame()
    stats["Mean"] = numeric_df.mean()
    stats["Median"] = numeric_df.median()
    # Mode can return multiple values; we take the first one
    stats["Mode"] = numeric_df.mode().iloc[0]

    # 2. Dispersion (Spread)
    stats["Std Dev"] = numeric_df.std()
    stats["Variance"] = numeric_df.var()
    stats["Range"] = numeric_df.max() - numeric_df.min()
    # Interquartile Range (IQR)
    stats["IQR"] = numeric_df.quantile(0.75) - numeric_df.quantile(0.25)

    # 3. Shape of Distribution
    stats["Skewness"] = numeric_df.skew()
    stats["Kurtosis"] = numeric_df.kurt()

    print("--- DISTRIBUTION ANALYSIS ---")
    return stats.transpose()


def validate_datatypes(df):
    """
    Validates if the DataFrame columns match the expected data types.
    Returns a status report.
    """
    # Define our expectations
    expected_types = {
        "TransactionId": "str",  # or 'string'
        "BatchId": "str",
        "AccountId": "str",
        "SubscriptionId": "str",
        "CustomerId": "str",
        "CurrencyCode": "category",
        "CountryCode": "category",
        "ProviderId": "category",
        "ProductId": "category",
        "ProductCategory": "category",
        "ChannelId": "category",
        "Amount": "float64",
        "Value": "int64",
        "TransactionStartTime": "datetime64[us, UTC]",
        "PricingStrategy": "category",
        "FraudResult": "bool",
    }

    results = []

    for col in df.columns:
        actual = str(df[col].dtype)
        expected = expected_types.get(col, "Unknown")

        is_correct = actual == expected

        results.append(
            {
                "Column": col,
                "Actual Type": actual,
                "Expected Type": expected,
                "Status": "✅ Correct" if is_correct else "❌ Incorrect",
            }
        )

    return pd.DataFrame(results)


def get_categorical_cols(df: pd.DataFrame) -> list:
    """Returns a list of categorical columns in the DataFrame."""
    return df.select_dtypes(include=["category", "bool"]).columns.tolist()


def get_numerical_cols(df: pd.DataFrame) -> list:
    """Returns a list of numerical columns in the DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def plot_numerical_distributions(
    df: pd.DataFrame,
    numerical_cols: list,
    figsize: tuple = (16, 12),
    bins: int = 30,
    apply_log=False,
) -> None:
    """Visualize distributions of all numerical features.

    Creates a grid of histograms with KDE overlays for each numerical column.
    Helps identify patterns, skewness, and potential outliers.

    Args:
        data: DataFrame containing numerical columns to visualize
        numerical_cols: List of numerical column names to visualize
        figsize: Figure size as (width, height). Default is (16, 12)
        bins: Number of bins for histograms. Default is 30

    Example:
        >>> plot_numerical_distributions(df, get_numerical_cols(df))
    """

    if not numerical_cols:
        print("No numerical columns found in the DataFrame")
        return

    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        series = df[col].dropna()
        if apply_log:
            series = np.log1p(series)
            title = f"{col} (log1p)"
            xlabel = f"log1p({col})"
        else:
            title = col
            xlabel = col

        axes[i].hist(series, bins=bins)
        axes[i].set_title(title)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel("Frequency")

    for j in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_numerical_boxplots(
    df: pd.DataFrame, numerical_cols: list, figsize: tuple = (14, 8)
) -> None:
    """Create boxplots for all numerical features to identify outliers.

    Args:
        df: DataFrame containing numerical columns to visualize
        numerical_cols: List of numerical column names to visualize
        figsize: Figure size as (width, height). Default is (14, 8)

    Example:
        >>> plot_numerical_boxplots(df, get_numerical_cols(df))
    """
    if not numerical_cols:
        print("No numerical columns found in the DataFrame")
        return

    fig, ax = plt.subplots(figsize=figsize)
    df[numerical_cols].boxplot(ax=ax)
    ax.set_title("Boxplots of Numerical Features")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: list,
    figsize: tuple = (16, 12),
    max_categories: int = 20,
) -> None:
    """Visualize distributions of all categorical features.

    Creates bar charts for each categorical column to show category frequency
    and variability. High-cardinality columns are truncated to the most common
    categories.

    Args:
        df: DataFrame containing categorical columns to visualize
        categorical_cols: List of categorical column names to visualize
        figsize: Figure size as (width, height). Default is (16, 12)
        max_categories: Maximum number of categories to display per column.

    Example:
        >>> plot_categorical_distributions(df, get_categorical_cols(df))
    """
    if not categorical_cols:
        print("No categorical columns found in the DataFrame")
        return

    if not categorical_cols:
        print("No categorical columns found in the DataFrame")
        return

    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        vc = df[col].astype(
            "object").value_counts(
                dropna=False
                ).head(max_categories)

        sns.barplot(x=vc.index.astype(str),
                    y=vc.values, ax=ax,
                    hue=vc.index.astype(str),
                    palette="viridis",
                    legend=False)

        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def correlation_analysis(
    df: pd.DataFrame,
    numerical_cols: list,
    method: str = "pearson",
    figsize: tuple = (12, 10),
) -> pd.DataFrame:
    """Compute and visualize correlation between numerical features.

    Calculates pairwise correlations for all numerical columns and displays
    them as a heatmap. Helps identify relationships and multicollinearity.

    Args:
        df: DataFrame containing numerical columns
        numerical_cols: List of numerical column names to analyze
        method: Correlation method - "pearson", "spearman", or
            "kendall". Default is "pearson"
        figsize: Figure size as (width, height). Default is (12, 10)

    Returns:
        Correlation matrix as a DataFrame

    Example:
        >>> corr_matrix = correlation_analysis(df)
    """
    if not numerical_cols:
        print("No numerical columns found in the DataFrame")
        return pd.DataFrame()

    if len(numerical_cols) < 2:
        print("At least 2 numerical columns required for correlation analysis")
        return pd.DataFrame()

    corr_matrix = df[numerical_cols].corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title(f"Correlation Matrix ({method.capitalize()} Correlation)")
    plt.tight_layout()
    plt.show()

    return corr_matrix


def handle_missing_values(df,
                          num_strategy='median',
                          cat_strategy='most_frequent',
                          return_report=True):
    """
    Automatically detects and imputes missing values in a DataFrame.
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame with missing values.
    num_strategy : str, default='median'
        Strategy for numerical columns:
        'mean', 'median', 'most_frequent', or 'constant'
    cat_strategy : str, default='most_frequent'
        Strategy for categorical columns: 'most_frequent', 'constant'
    return_report : bool, default=True
        Whether to return a report of missing values before and after.

    Returns:
    --------
    df_imputed : pandas DataFrame
        DataFrame with missing values imputed.
    report : dict (optional)
        Summary report of missing values.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df_imputed = df.copy()

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(
        include=['int64', 'float64']
        ).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'string', 'category', 'bool']
    ).columns.tolist()

    missing_before = df.isnull().sum()
    if missing_before.sum() == 0:
        print("No missing values detected in the DataFrame.")
        return df_imputed, {
            'missing_before': missing_before,
            'missing_percent_before': (
                missing_before / len(df) * 100
            ).round(2),
            'missing_after': 0,
            'total_missing_filled': 0
        } if return_report else df_imputed

    missing_percent_before = (missing_before / len(df) * 100).round(2)

    report = {
        'missing_before': missing_before[missing_before > 0],
        'missing_percent_before': (
            missing_percent_before[missing_percent_before > 0]
        )
    }

    # Impute Numerical Columns
    if numerical_cols:
        num_imputer = SimpleImputer(strategy=num_strategy)
        df_imputed[numerical_cols] = num_imputer.fit_transform(
            df_imputed[numerical_cols]
        )

    # Impute Categorical Columns
    if categorical_cols:
        if cat_strategy not in {'most_frequent', 'constant'}:
            raise ValueError(
                "cat_strategy must be either 'most_frequent' or 'constant'"
            )

        for col in categorical_cols:
            series = df_imputed[col]

            if not series.isna().any():
                continue

            if cat_strategy == 'most_frequent':
                mode_values = series.mode(dropna=True)
                if not mode_values.empty:
                    fill_value = mode_values.iloc[0]
                elif pd.api.types.is_bool_dtype(series):
                    fill_value = False
                else:
                    fill_value = 'Missing'
            else:  # cat_strategy == 'constant'
                fill_value = (
                    False if pd.api.types.is_bool_dtype(series)
                    else 'Missing'
                )

            if isinstance(series.dtype, pd.CategoricalDtype):
                if fill_value not in series.cat.categories:
                    series = series.cat.add_categories([fill_value])

            df_imputed[col] = series.fillna(fill_value)

    # Final Report
    missing_after = df_imputed.isnull().sum().sum()

    report['missing_after'] = missing_after
    report['total_missing_filled'] = missing_before.sum()

    print("✅ Imputation Completed!")
    print(f"   Total missing values filled: {report['total_missing_filled']}")
    print(f"   Remaining missing values: {report['missing_after']}")

    if return_report:
        return df_imputed, report
    else:
        return df_imputed


def outlier_detection_boxplot(
    df: pd.DataFrame, figsize: tuple = (14, 8)
) -> pd.DataFrame:
    """Detect potential outliers in numerical columns using box plots.

    Args:
        df: DataFrame to inspect
        figsize: Figure size as (width, height). Default is (14, 8)

    Returns:
        DataFrame with outlier counts and bounds for each numerical column
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_cols:
        print("No numerical columns found in the DataFrame")
        return pd.DataFrame()

    summary_rows = []
    for col in numerical_cols:
        series = df[col].dropna()
        if series.empty:
            summary_rows.append(
                {
                    "column": col,
                    "outlier_count": 0,
                    "outlier_percent": 0.0,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                }
            )
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())

        summary_rows.append(
            {
                "column": col,
                "outlier_count": outlier_count,
                "outlier_percent": float(outlier_count / len(series) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
        )

    fig, ax = plt.subplots(figsize=figsize)
    df[numerical_cols].boxplot(ax=ax)
    ax.set_title("Boxplots for Outlier Detection")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(summary_rows)


def save_preprocessed_data(df: pd.DataFrame, filename: str) -> None:
    """Saves the preprocessed DataFrame to a CSV file.

    Creates the directory if it doesn't exist.

    Args:
        df: DataFrame to save
        filename: Path to the output CSV file (e.g.,
            'data/preprocessed_data.csv')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    df.to_csv(filename, index=False)
    print(f"✅ Preprocessed data saved to {filename}")
