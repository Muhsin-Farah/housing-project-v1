# Housing Project

A Python ML project for London housing data analysis using pandas and SQLite.

## Pipeline Overview

```
Raw Data (pp-complete.csv)
    ↓
[1] Clean (src/clean.py)
    • Load London properties only
    • Rename & consolidate columns
    • Validate prices & dates
    ↓
london_properties_cleaned.csv
    ↓
[2] Adjust (src/adjust.py)
    • Calculate CSI by property type
    • Adjust for inflation (2023 base)
    ↓
london_properties_adjusted.csv
    ↓
[3] Load (src/load.py)
    • Create SQLite database
    • Batch insert with indexes
    ↓
housing.db
```

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Individual Steps

```bash
python src/clean.py      # Step 1: Clean data
python src/adjust.py     # Step 2: Apply CSI & inflation
python src/load.py       # Step 3: Load to SQLite
```

## Project Structure

```
data/
  ├── pp-complete.csv                    # Raw UK Price Paid data
  ├── london_properties_cleaned.csv       # After step 1
  ├── london_properties_adjusted.csv      # After step 2
  ├── housing.db                         # SQLite database
  └── cpi.csv                            # Optional: CPI data
src/
  ├── clean.py                           # Load & clean
  ├── adjust.py                          # CSI & inflation
  ├── load.py                            # SQLite loading
  ├── model.py                           # Model training
  ├── visualization.py                   # Charts
  └── database.py                        # DB utilities
load_london_data.py                       # Standalone utility
```

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib

