# Housing Project

A Python machine learning project for housing data analysis using pandas, NumPy, scikit-learn, and SQLite.

## Goals
- Load and preprocess housing data with pandas
- Clean and adjust for inflation using CPI
- Store processed data in SQLite database
- Build regression models with scikit-learn
- Visualize results with matplotlib

## Data Processing Pipeline

1. **Raw Data** (`data/sample_housing.csv`): Original housing data
2. **CPI Reference** (`data/cpi.csv`): Consumer Price Index for inflation adjustment
3. **Preprocessing** (`src/preprocessing.py`): 
   - Cleans missing values and duplicates
   - Adjusts prices to base year (2023) using CPI
   - Handles feature engineering and scaling
4. **Stored Data** (`housing.db`): Final cleaned, adjusted data for ML model
5. **Model Training** (`src/model.py`): Trains regression models on processed data
6. **Visualization** (`src/visualization.py`): Charts and performance metrics

### Why CPI Adjustment?
Housing prices vary by year due to inflation. Adjusting all prices to a base year 
(2023 dollars) ensures fair comparison and better model training without inflation bias.

## Project Structure
```
data/
  ├── sample_housing.csv    # Raw housing data
  └── cpi.csv              # Consumer Price Index data
src/
  ├── data_loader.py       # CSV loading utilities
  ├── preprocessing.py     # Data cleaning and CPI adjustment
  ├── database.py          # SQLite connection and operations
  ├── model.py             # ML model training
  ├── visualization.py     # Matplotlib visualizations
  └── sqlite_example.py    # Example usage script
tests/
  └── test_example.py      # Unit tests
notebooks/
  └── README.md            # Jupyter notebook directory
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Process and store data in SQLite:
```bash
python src/sqlite_example.py --database housing.db --csv data/sample_housing.csv --table housing_data
```

### Train model:
```bash
python src/model.py
```

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- jupyter (for notebook exploration)

## Notes
- SQLite database is created automatically on first run
- CPI values are used to normalize prices to 2023 dollars
- All preprocessing is applied before storing to SQLite for efficiency

