# db.py
from dotenv import load_dotenv
import os
from mysql.connector import pooling, Error

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", ""),
    "database": os.getenv("DB_NAME", "image_classification"),
}

_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=4,
    **DB_CONFIG
)

def get_conn():
    return _pool.get_connection()

def save_prediction(filename: str, prediction: str, confidence: float) -> int:
    """Insert a prediction and return inserted id"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        sql = "INSERT INTO image_records (filename, prediction, confidence, timestamp) VALUES (%s, %s, %s, NOW())"
        cur.execute(sql, (filename, prediction, float(confidence)))
        conn.commit()
        return cur.lastrowid
    finally:
        cur.close()
        conn.close()

def fetch_recent(limit: int = 100):
    """Return recent rows as list of dicts"""
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            "SELECT id, filename, prediction, confidence, timestamp FROM image_records ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        return cur.fetchall()  # list of dicts
    finally:
        cur.close()
        conn.close()
