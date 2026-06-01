import mysql.connector
import pandas as pd


def create_mysql_connection(host: str = "localhost", user: str = "root", password: str = "", database: str = "housing"):
    """Create a MySQL database connection."""
    return mysql.connector.connect(host=host, user=user, password=password, database=database)


def load_sql_table(conn, table_name: str) -> pd.DataFrame:
    """Load a table from MySQL into a DataFrame."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"SELECT * FROM `{table_name}`")
    rows = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(rows)


def save_df_to_sql(df, table_name: str, conn, if_exists: str = "replace"):
    """Save a pandas DataFrame to a MySQL table."""
    cursor = conn.cursor()
    if if_exists == "replace":
        cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")

    columns = [f"`{col}` TEXT" for col in df.columns]
    cursor.execute(f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(columns)})")

    placeholders = ", ".join(["%s"] * len(df.columns))
    column_names = ", ".join([f"`{col}`" for col in df.columns])
    insert_sql = f"INSERT INTO `{table_name}` ({column_names}) VALUES ({placeholders})"

    values = [tuple(x) for x in df.to_numpy()]
    cursor.executemany(insert_sql, values)
    conn.commit()
    cursor.close()
