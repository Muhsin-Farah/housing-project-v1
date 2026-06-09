# Housing Project
An end-to-end Machine Learning pipeline designed to analyze and predict residential property prices in Greater London. This project leverages historical data from the UK Land Registry to uncover key drivers of property valuation, specifically accounting for inflationary trends and geographic proximity to city centers.

Tech Stack
Data Manipulation: Pandas, NumPy

Statistical Analysis: SciPy

Machine Learning: Scikit-Learn (Random Forest Regressor)

Data Storage: SQLite

Visualization: Matplotlib

Methodology
1. Data Processing & ETL
Data Acquisition: Ingested raw Land Registry CSV data, performing initial filtering to isolate Greater London transactions.

Economic Adjustment: Applied the Consumer Price Index (CPI) to normalize historical pricing for inflation, ensuring an "apples-to-apples" comparison over time.

Feature Engineering: Calculated geographical proximity to central transit hubs (e.g., Charing Cross) to quantify the "location premium."

Data Cleaning: Handled missing values and anomalies using Pandas to ensure a robust dataset before storage in a local SQLite database.

2. Feature Transformation
Logarithmic Scaling: Implemented np.log transformation on target price variables. This effectively normalized the distribution, reduced the impact of extreme outliers/skewness, and improved the model's predictive stability.

3. Predictive Modeling
Model Selection: Utilized a Random Forest Regressor to handle non-linear relationships between property features and final sale prices.

Analysis: Evaluated feature importance to determine which variables—such as distance to the city center and other structural factors—most significantly influence market volatility and pricing.

Key Takeaways
Demonstrated an ability to manage the full data lifecycle: from raw CSV ingestion and SQL storage to feature engineering and predictive modeling.

Applied domain knowledge (CPI adjustments) to ensure statistical accuracy in economic datasets.

Improved model robustness by implementing advanced data transformations to mitigate the impact of outliers.

Note : London housing prices are very skewed and different districts vary in Price despite housing sizes being similar which is why some of my
features may confuse you since they account for location a lot.


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

