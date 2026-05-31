import pandas as pd


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(
            file_path,
            dtype={
                "TransactionId": "str",
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
                "PricingStrategy": "category",
                "FraudResult": "bool",
            },
            parse_dates=["TransactionStartTime"],
        )

        df["Amount"] = df["Amount"].astype("float64")
        df["Value"] = df["Value"].astype("int64")
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
