import pandas as pd
import sqlite3

def create_connection():
    conn = sqlite3.connect('platinum_challenge.db')
    return conn

def create_table(conn):
    # Create analisis_result table if not exists
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS analisis_result (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_text TEXT,
        clean_text TEXT,
        sentiment TEXT
    );
    """
    conn.execute(create_table_sql)
    print("Table analisis_result created successfully.")

def show_analisis_result(conn):
    # Show cleansing result
    print("Showing cleansing result...")
    df = pd.read_sql_query("SELECT * FROM analisis_result", conn)
    return df.T.to_dict()

def insert_result_to_db(conn, raw_text, cleansing_text, analisis_text):
    # Insert result to database
    print("Inserting result to database...")
    df = pd.DataFrame({'raw_text': [raw_text], 'clean_text': [cleansing_text], "Sentiment": [analisis_text]})
    df.to_sql('analisis_result', conn, if_exists='append', index=False)
    print("Inserting result to database success!")

def insert_upload_result_to_db(conn, clean_df):
    # Insert result to database
    print("Inserting result to database...")
    clean_df.to_sql('analisis_result', conn, if_exists='append', index=False)
    print("Inserting result to database success!")