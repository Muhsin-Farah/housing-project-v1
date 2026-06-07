import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('./data/pp-complete.csv')

print("=" * 80)
print("DATA OVERVIEW")
print("=" * 80)
print(f"\nShape: {df.shape} (rows, columns)")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nMemory Usage:\n{df.memory_usage(deep=True)}")
print("=" * 80)
