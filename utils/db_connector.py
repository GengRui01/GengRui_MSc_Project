import pymysql
import pandas as pd
from sqlalchemy import create_engine
import os
import subprocess

# Create database connection
def get_connection():
    engine = create_engine("mysql+pymysql://root:123456@localhost:3306/gengrui_msc")
    return engine.connect()

# Fetch student data
def fetch_student_data():
    conn = get_connection()
    query = "SELECT * FROM student_behavior;"
    df = pd.read_sql(query, conn)

    # If table is empty, auto-generate data
    if df.empty:
        print("[INFO] No data found in database, generating synthetic data...")
        subprocess.run(["python", "utils/data_generator.py"], check=True)
        # Reconnect and fetch again after generating data
        conn = get_connection()
        df = pd.read_sql(query, conn)
        print("[INFO] Synthetic data loaded successfully.")

    conn.close()
    return df