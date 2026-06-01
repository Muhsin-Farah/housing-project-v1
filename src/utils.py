import pandas as pd


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for a DataFrame."""
    return df.describe()


def save_dataframe(df: pd.DataFrame, path: str):
    """Save a DataFrame to CSV."""
    df.to_csv(path, index=False)
