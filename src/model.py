"""Train regression models on London housing data."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_training_data(csv_path='./data/london_properties_adjusted.csv'):
    """Load cleaned data with CSI and inflation adjustments."""
    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    print(f"✓ Loaded {len(df):,} records")
    return df


def prepare_features(df):
    """Prepare features for model training, encode categorical variables."""
    df = df.copy()
    
    # Use log of CSI-adjusted price as target
    y = df['log_price_adjusted']
    
    # Select features
    features = ['property_age', 'distance_to_centre', 'csi', 'year']
    
    # Add encoded categorical features
    categorical = ['property_type', 'tenure_type', 'district']
    
    X = df[features].copy()
    
    # Encode categorical variables
    encoders = {}
    for col in categorical:
        le = LabelEncoder()
        X[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Drop rows with missing values
    X = X.dropna()
    y = y[X.index]
    
    print(f"✓ Features prepared: {list(X.columns)}")
    print(f"  {len(X):,} records with no missing values")
    
    return X, y, encoders


def train_models(X, y, test_size=0.2, random_state=42):
    """Train multiple regression models."""
    print(f"\nTraining models on {len(X):,} records...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    results = {}
    
    # Linear Regression
    print("\n1. Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    results['Linear Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'mse': mean_squared_error(y_test, y_pred_lr),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'r2': r2_score(y_test, y_pred_lr)
    }
    
    # Random Forest
    print("2. Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'mse': mean_squared_error(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'r2': r2_score(y_test, y_pred_rf)
    }
    
    return results, X_test, y_test


def print_results(results, y_test):
    """Print model performance metrics."""
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE (log price): {metrics['rmse']:.4f}")
        print(f"  MAE (log price): {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        
        # Convert back from log scale for interpretation
        actual_prices = np.exp(y_test)
        predicted_prices = np.exp(metrics['predictions'])
        actual_rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        print(f"  RMSE (actual prices): £{actual_rmse:,.0f}")
    
    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("TRAINING HOUSING PRICE MODELS")
    print("=" * 80)
    
    # Load data
    df = load_training_data()
    
    # Prepare features
    X, y, encoders = prepare_features(df)
    
    # Train models
    results, X_test, y_test = train_models(X, y)
    
    # Print results
    print_results(results, y_test)
    
    # Feature importance (Random Forest)
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(feature_importance.to_string(index=False))
    
    return results


if __name__ == "__main__":
    main()
