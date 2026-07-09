
def split_data(
    df: pd.DataFrame,
    target_col: str = "is_high_risk",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
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
    print(f"  Train: {X_train.shape[0]:,} records({X_train.shape[0] / n:.1%})")
    print(f"  Val:  {X_val.shape[0]:,} records({X_val.shape[0] / n:.1%})")
    print(f"  Test:  {X_test.shape[0]:,} records({X_test.shape[0] / n:.1%})")
    train_dist = y_train.value_counts(normalize=True).round(3).to_dict()
    print(f"  Target distribution (Train): {train_dist}")

    return X_train, X_val, X_test, y_train, y_val, y_test