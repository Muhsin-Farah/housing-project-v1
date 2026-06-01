<<<<<<< HEAD
# housing-project-v1
A machine learning project using Python Numpy,SciPy,Pandas and SQL - This a prelude to my next project which will be done with pytorch
=======
# Housing Project

A Python project for housing data analysis and machine learning.

## Goals
- Load and preprocess housing data
- Build regression models with scikit-learn
- Store and query datasets with MySQL
- Use pandas and NumPy for data manipulation

## Structure
- `data/` - raw and processed datasets
- `notebooks/` - exploratory analysis notebooks
- `src/` - reusable Python modules
- `tests/` - unit tests

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/model.py data/sample_housing.csv
```

## Sample MySQL usage
```bash
python src/mysql_example.py --host localhost --user root --password yourpass --database housing
```

This sample script loads `data/sample_housing.csv` into MySQL and reads it back using the MySQL connector.
>>>>>>> 5379249 (Initial commit)
