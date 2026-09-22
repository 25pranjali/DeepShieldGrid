"""
init_db.py

Run this ONCE (or any time you want to reset the database) to create
database.db and fill it with:
  - a demo user account
  - a handful of sample sensor readings
  - a sample FDI incident with SHAP values attached

Usage:
    python init_db.py
"""

import os
import sqlite3
from werkzeug.security import generate_password_hash

from dataset_utils import load_clean_dataset, MODEL_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'admin'))
);

CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    row_index INTEGER,
    timestamp TEXT,
    source TEXT,
    actual_frequency_value REAL,
    fraction_of_second REAL,
    time_synchronized REAL,
    interarrival_time REAL,
    time_difference REAL,
    ground_truth_label TEXT,
    manipulated_fields TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source TEXT,
    attack_type TEXT,
    confidence REAL,
    risk_score REAL,
    risk_level TEXT,
    ground_truth_label TEXT,
    manipulated_fields TEXT
);

CREATE TABLE IF NOT EXISTS shap_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL,
    feature_name TEXT,
    shap_value REAL,
    FOREIGN KEY (incident_id) REFERENCES incidents(id)
);
"""


def create_database():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Tables created in {DB_PATH}")


def insert_demo_user(conn):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    if cur.fetchone():
        print("Demo user already exists, skipping.")
        return

    cur.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        ("admin", generate_password_hash("admin123"), "admin"),
    )
    cur.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        ("operator", generate_password_hash("operator123"), "user"),
    )
    conn.commit()
    print("Demo users created: admin/admin123 (role=admin), operator/operator123 (role=user)")


def insert_sample_readings(conn):
    """Insert a small number of real rows from the dataset so the
    dashboard has something to show before the simulation is started."""
    df = load_clean_dataset()
    sample = df.head(20)

    cur = conn.cursor()
    for _, row in sample.iterrows():
        cur.execute(
            """
            INSERT INTO readings (
                row_index, timestamp, source,
                actual_frequency_value, fraction_of_second, time_synchronized,
                interarrival_time, time_difference,
                ground_truth_label, manipulated_fields
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["row_index"]),
                "sample-load",
                "Simulated PMU Stream",
                row["Actual frequency value"],
                row["Fraction of second"],
                row["Time synchronized"],
                row["interarrival time"],
                row["time difference"],
                row["attack_type"],
                row["manipulated_fields"] if not str(row["manipulated_fields"]) == "nan" else None,
            ),
        )
    conn.commit()
    print(f"Inserted {len(sample)} sample readings from the dataset.")


def insert_sample_incident(conn):
    """Insert one example FDI incident (with SHAP values) so the Attack
    Detection and SHAP pages have something to display immediately,
    even before you run the live simulation."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM incidents LIMIT 1")
    if cur.fetchone():
        print("Sample incident already exists, skipping.")
        return

    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "Simulated PMU Stream", "FDI", 94.25,
            97.1, "Critical", "FDI", "Actual frequency value",
        ),
    )
    incident_id = cur.lastrowid

    # Example SHAP-style values, just for demo purposes until the real
    # simulation produces actual SHAP output.
    demo_shap = [
        ("Actual frequency value", 0.42),
        ("time difference", -0.31),
        ("interarrival time", 0.18),
        ("Fraction of second", 0.07),
        ("Time synchronized", -0.02),
    ]
    for feature_name, shap_value in demo_shap:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (incident_id, feature_name, shap_value),
        )

    conn.commit()
    print(f"Inserted sample incident (id={incident_id}) with SHAP values.")


if __name__ == "__main__":
    create_database()
    conn = sqlite3.connect(DB_PATH)
    insert_demo_user(conn)
    insert_sample_readings(conn)
    insert_sample_incident(conn)
    conn.close()
    print("\nDatabase setup complete ->", DB_PATH)
