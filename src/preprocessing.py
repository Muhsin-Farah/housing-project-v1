import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_cpi(cpi_path: str = "data/cpi.csv") -> dict:
    """
    Load Consumer Price Index (CPI) values from CSV.
    
    Args:
        cpi_path: Path to CPI CSV file with 'year' and 'cpi_value' columns
        
    Returns:
        Dictionary mapping year to CPI value
    """
    cpi_df = pd.read_csv(cpi_path)
    return dict(zip(cpi_df['year'], cpi_df['cpi_value']))


def adjust_prices_for_inflation(df: pd.DataFrame, year_column: str, price_column: str, 
                                 base_year: int = 2023, cpi_path: str = "data/cpi.csv") -> pd.DataFrame:
    """
    Adjust historical housing prices to base year dollars using CPI.
    
    This normalizes prices across different years for fair ML model training.
    All prices are adjusted to a single year's value, eliminating inflation bias.
    
    Args:
        df: DataFrame with year and price columns
        year_column: Name of the year column
        price_column: Name of the price column to adjust
        base_year: Target year for price adjustment (default: 2023)
        cpi_path: Path to CPI CSV file
        
    Returns:
        DataFrame with new 'adjusted_price' column containing inflation-adjusted prices
    """
    df = df.copy()
    cpi = load_cpi(cpi_path)
    base_cpi = cpi[base_year]
    
    df['adjusted_price'] = df.apply(
        lambda row: row[price_column] * (base_cpi / cpi.get(row[year_column], cpi[base_year])),
        axis=1
    )
    return df


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
