import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning on housing data."""
    df = df.copy()
    df = df.dropna()
    return df


def get_features_and_target(df: pd.DataFrame, target_column: str):
    """Split data into features and target."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split features and target into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
