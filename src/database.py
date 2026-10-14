"""SQLite storage for prediction history."""
from pathlib import Path
import sqlite3
from datetime import datetime

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "review_history.db"


def connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        review_text TEXT NOT NULL,
        predicted_label INTEGER NOT NULL,
        predicted_category TEXT NOT NULL,
        model_name TEXT
    )""")
    conn.commit()
    return conn


def save_prediction(review_text, label, category, model_name="Unknown"):
    with connect() as conn:
        conn.execute("INSERT INTO predictions(created_at, review_text, predicted_label, predicted_category, model_name) VALUES (?, ?, ?, ?, ?)",
                     (datetime.now().astimezone().isoformat(timespec="seconds"), review_text,
                      int(label), category, model_name))


def get_history(limit=5000):
    with connect() as conn:
        return conn.execute("SELECT id, created_at, review_text, predicted_label, predicted_category, model_name FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
