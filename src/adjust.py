"""Apply Cost-to-Sales Index (CSI), inflation adjustments, and feature engineering."""

import pandas as pd
import numpy as np
from pathlib import Path


# Approximate distances from Charing Cross to London postcode areas (km)
POSTCODE_DISTANCES = {
    'E': 8, 'EC': 2, 'SE': 6, 'S': 10, 'SW': 8, 'W': 5, 'WC': 3,
    'N': 7, 'NE': 12, 'NW': 8, 'EN': 20, 'IG': 15, 'RM': 18, 'DA': 20,
    'BR': 20, 'CR': 18, 'TW': 15, 'KT': 18, 'SM': 20, 'HA': 15, 'UB': 18,
    'SL': 30, 'ME': 45
}


def load_cpi(path='./data/cpi.csv'):
    """Load CPI data for inflation adjustment."""
    try:
        cpi_df = pd.read_csv(path)
        return dict(zip(cpi_df['year'], cpi_df['cpi_value']))
    except:
        print("⚠ CPI file not found - skipping inflation adjustment")
        return {}


def extract_postcode_area(postcode):
    """Extract postcode area (first 1-2 letters) from full postcode."""
    if pd.isna(postcode):
        return None
    postcode_str = str(postcode).strip().upper()
    # Extract letters only from start
    for i, char in enumerate(postcode_str):
        if not char.isalpha():
            return postcode_str[:i]
    return postcode_str


def calculate_distance_to_london(df):
    """Calculate approximate distance to Charing Cross from postcode."""
    df['postcode_area'] = df['postcode'].apply(extract_postcode_area)
    df['distance_to_centre'] = df['postcode_area'].map(POSTCODE_DISTANCES)
    df['distance_to_centre'] = df['distance_to_centre'].fillna(25)  # Default 25km for unknown
    return df


def calculate_age(df):
    """Calculate property age from year built (assume current year is 2026)."""
    current_year = 2026
    df['property_age'] = current_year - df['year']
    return df


def apply_csi(df):
    """Calculate CSI by property type."""
    property_medians = df.groupby('property_type')['price'].median()
    overall_median = df['price'].median()
    df['csi'] = df['property_type'].map(lambda x: property_medians.get(x, overall_median) / overall_median)
    return df


def adjust_inflation(df, cpi_dict, base_year=2023):
    """Adjust prices for inflation to base year."""
    if not cpi_dict:
        df['price_adjusted'] = df['price']
        return df
    
    base_cpi = cpi_dict.get(base_year)
    if not base_cpi:
        df['price_adjusted'] = df['price']
        return df
    
    df['price_adjusted'] = df.apply(
        lambda row: row['price'] * (base_cpi / cpi_dict.get(row['year'], base_cpi)), 
        axis=1
    )
    return df


def add_log_price(df):
    """Add log-transformed price columns."""
    df['log_price'] = np.log(df['price'])
    df['log_price_adjusted'] = np.log(df['price_adjusted'])
    return df


def main():
    print("=" * 80)
    print("FEATURE ENGINEERING & ADJUSTMENTS")
    print("=" * 80)
    
    df = pd.read_csv('./data/london_properties_cleaned.csv', parse_dates=['date'])
    print(f"Loaded {len(df):,} records")
    
    # Calculate new features
    df = calculate_age(df)
    df = calculate_distance_to_london(df)
    
    # Apply adjustments
    cpi = load_cpi()
    df = apply_csi(df)
    df = adjust_inflation(df, cpi)
    df['price_csi_adjusted'] = df['price_adjusted'] * df['csi']
    
    # Add log prices
    df = add_log_price(df)
    
    output = './data/london_properties_adjusted.csv'
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    
    print(f"\n✓ Saved to {output}")
    print(f"New columns added:")
    print(f"  • property_age: {df['property_age'].min()}-{df['property_age'].max()} years")
    print(f"  • distance_to_centre: {df['distance_to_centre'].min()}-{df['distance_to_centre'].max()} km")
    print(f"  • log_price: ln(original price)")
    print(f"  • log_price_adjusted: ln(adjusted price)")
    print(f"\nPrice statistics:")
    print(f"  Original: £{df['price'].mean():,.0f}")
    print(f"  Adjusted: £{df['price_csi_adjusted'].mean():,.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
