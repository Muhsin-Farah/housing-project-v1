import pandas as pd
from src.preprocessing import clean_data, get_features_and_target


def test_clean_data_removes_na():
    df = pd.DataFrame({"a": [1, None, 3], "price": [10, 20, None]})
    cleaned = clean_data(df)
    assert cleaned.isna().sum().sum() == 0


def test_get_features_and_target_splits_columns():
    df = pd.DataFrame({"feature": [1, 2], "price": [10, 20]})
    X, y = get_features_and_target(df, "price")
    assert "price" not in X.columns
    assert y.name == "price"
