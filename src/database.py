import sqlite3
import pandas as pd


def create_sqlite_connection(database: str = "housing.db"):
    """Create a SQLite database connection."""
    return sqlite3.connect(database)


def load_sql_table(conn, table_name: str) -> pd.DataFrame:
    """Load a table from SQLite into a DataFrame."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(rows, columns=columns)


def save_df_to_sql(df, table_name: str, conn, if_exists: str = "replace"):
    """Save a pandas DataFrame to a SQLite table."""
    cursor = conn.cursor()
    if if_exists == "replace":
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    columns = [f"{col} TEXT" for col in df.columns]
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})")

    placeholders = ", ".join(["?"] * len(df.columns))
    column_names = ", ".join(df.columns)
    insert_sql = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

    values = [tuple(x) for x in df.to_numpy()]
    cursor.executemany(insert_sql, values)
    conn.commit()
    cursor.close()
