import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessing import clean_data, get_features_and_target, split_data
from src.data_loader import load_csv


def train_linear_model(csv_path: str, target_column: str = "price"):
    df = load_csv(csv_path)
    df = clean_data(df)
    X, y = get_features_and_target(df, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mse, r2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a housing regression model.")
    parser.add_argument("csv_path", help="Path to the housing CSV dataset.")
    parser.add_argument("--target", default="price", help="Target column name.")
    args = parser.parse_args()

    model, mse, r2 = train_linear_model(args.csv_path, args.target)
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
