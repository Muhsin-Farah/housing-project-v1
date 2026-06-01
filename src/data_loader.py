import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load housing data from a CSV file."""
    return pd.read_csv(path)


def load_sql_table(conn, table_name: str) -> pd.DataFrame:
    """Load a table from a MySQL connection into a DataFrame."""
    query = f"SELECT * FROM `{table_name}`"
    return pd.read_sql(query, conn)
