"""Load adjusted data into SQLite."""

import pandas as pd
import sqlite3
from pathlib import Path


def create_db(db_path='./data/housing.db'):
    """Create SQLite database with optimized schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS properties")
    
    cursor.execute("""
    CREATE TABLE properties (
        transaction_id TEXT PRIMARY KEY,
        price INTEGER,
        date TEXT,
        year INTEGER,
        postcode TEXT,
        district TEXT,
        address TEXT,
        property_type TEXT,
        new_build TEXT,
        tenure_type TEXT,
        property_age INTEGER,
        postcode_area TEXT,
        distance_to_centre REAL,
        csi REAL,
        price_adjusted REAL,
        price_csi_adjusted REAL,
        log_price REAL,
        log_price_adjusted REAL
    )
    """)
    
    # Create indexes
    for idx_col in ['date', 'postcode', 'district', 'year', 'price', 'property_age', 'distance_to_centre']:
        cursor.execute(f"CREATE INDEX idx_{idx_col} ON properties({idx_col})")
    
    conn.commit()
    print(f"✓ Created database: {db_path}")
    return conn


def insert_data(df, conn):
    """Insert data in batches."""
    cursor = conn.cursor()
    cols = ', '.join(df.columns)
    placeholders = ', '.join(['?' for _ in df.columns])
    sql = f"INSERT INTO properties ({cols}) VALUES ({placeholders})"
    
    for i in range(0, len(df), 5000):
        batch = df.iloc[i:i+5000]
        cursor.executemany(sql, [tuple(row) for row in batch.values])
        print(f"  Inserted {min(i+5000, len(df)):,}/{len(df):,}", end='\r')
    
    conn.commit()
    print(f"\n✓ Inserted {len(df):,} records")


def main():
    print("=" * 80)
    print("LOADING DATA INTO SQLITE")
    print("=" * 80)
    
    df = pd.read_csv('./data/london_properties_adjusted.csv')
    print(f"Loaded {len(df):,} records")
    
    conn = create_db()
    insert_data(df, conn)
    
    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM properties")
    count = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(price), AVG(price), MAX(price) FROM properties")
    min_p, avg_p, max_p = cursor.fetchone()
    
    print(f"\n✓ Database verified:")
    print(f"  Records: {count:,}")
    print(f"  Price range: £{min_p:,.0f} - £{max_p:,.0f} (avg: £{avg_p:,.0f})")
    print("=" * 80)
    
    conn.close()


if __name__ == "__main__":
    main()
