"""Load and clean London housing data."""

import pandas as pd
from pathlib import Path


def load_london_properties(csv_path='./data/pp-complete.csv'):
    """Load only London properties from pp-complete.csv using chunking."""
    print("Loading London properties...")
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=50000):
        london = chunk[chunk['LONDON'] == 'LONDON']
        if len(london) > 0:
            chunks.append(london)
    df = pd.concat(chunks, ignore_index=True)
    print(f"✓ Loaded {len(df):,} properties")
    return df


def clean_data(df):
    """Convert types and validate data."""
    # Convert types
    df['date'] = pd.to_datetime(df.iloc[:, 2], errors='coerce')
    df['price'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df['year'] = df['date'].dt.year
    
    # Remove invalid rows
    df = df.dropna(subset=['price', 'date'])
    df = df[(df['price'] >= 10000) & (df['price'] <= 10000000)]
    df = df.drop_duplicates(subset=[df.columns[0]])
    
    return df


def clean_columns(df):
    """Rename and consolidate columns."""
    col_map = {
        df.columns[0]: 'transaction_id',
        df.columns[1]: 'price_raw',
        df.columns[2]: 'date_raw',
        df.columns[3]: 'postcode',
        df.columns[4]: 'property_type',
        df.columns[5]: 'new_build',
        df.columns[6]: 'tenure_type',
        df.columns[7]: 'house_number',
        df.columns[8]: 'flat_number',
        df.columns[9]: 'street',
        'LONDON': 'city',
        df.columns[12]: 'district'
    }
    df = df.rename(columns=col_map)
    
    # Consolidate address
    def make_address(row):
        parts = [str(row['house_number']).strip() if pd.notna(row['house_number']) else '']
        if pd.notna(row['flat_number']):
            parts.append(str(row['flat_number']).strip())
        if pd.notna(row['street']):
            parts.append(str(row['street']).strip())
        return ', '.join([p for p in parts if p])
    
    df['address'] = df.apply(make_address, axis=1)
    
    # Keep only essential columns
    df = df[['transaction_id', 'price', 'date', 'year', 'postcode', 'district', 
             'address', 'property_type', 'new_build', 'tenure_type']]
    
    return df


def main():
    print("=" * 80)
    print("CLEANING LONDON HOUSING DATA")
    print("=" * 80)
    
    df = load_london_properties()
    df = clean_data(df)
    df = clean_columns(df)
    
    output = './data/london_properties_cleaned.csv'
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    
    print(f"\n✓ Saved {len(df):,} records to {output}")
    print(f"Columns: {list(df.columns)}")
    print(f"Price: £{df['price'].min():,.0f} - £{df['price'].max():,.0f} (avg: £{df['price'].mean():,.0f})")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
