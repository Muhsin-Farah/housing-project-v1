import argparse
from src.database import create_sqlite_connection, save_df_to_sql, load_sql_table
from src.data_loader import load_csv


def main(database: str, csv_path: str, table_name: str):
    conn = create_sqlite_connection(database=database)
    df = load_csv(csv_path)
    save_df_to_sql(df, table_name, conn, if_exists="replace")

    loaded_df = load_sql_table(conn, table_name)
    print("Loaded data from SQLite:")
    print(loaded_df.head())
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample SQLite usage for housing data.")
    parser.add_argument("--database", default="housing.db", help="SQLite database file")
    parser.add_argument("--csv", default="data/sample_housing.csv", help="Path to sample housing CSV")
    parser.add_argument("--table", default="housing_data", help="SQLite table name")
    args = parser.parse_args()
    main(args.database, args.csv, args.table)
