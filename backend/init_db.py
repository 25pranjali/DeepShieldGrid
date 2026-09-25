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


DROP_SCHEMA = """
DROP TABLE IF EXISTS shap_values;
DROP TABLE IF EXISTS incidents;
DROP TABLE IF EXISTS readings;
DROP TABLE IF EXISTS users;
"""

def create_database():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DROP_SCHEMA)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Tables created in {DB_PATH}")


def insert_demo_user(conn):
    cur = conn.cursor()
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
    """Insert example incidents and normal events across all domains."""
    cur = conn.cursor()

    # 1. IEC 61850 Replay Incident (Medium Risk)
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "Substation Process Bus (IEC 61850 GOOSE)", "Replay", 88.0,
            64.6, "Medium", "Replay", "sqNum, stnum, sqDiff",
        ),
    )
    replay_id = cur.lastrowid
    for feat, shap_val in [
        ("sqDiff", 0.58),
        ("sqNum", 0.44),
        ("stDiff", 0.26),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (replay_id, feat, shap_val),
        )

    # 2. IEC 61850 Fault Incident (Medium Risk)
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "Substation Process Bus (IEC 61850 GOOSE)", "Fault", 90.0,
            65.2, "Medium", "Fault", "MU1CurrentAngleB, state_cb",
        ),
    )
    fault_id = cur.lastrowid
    for feat, shap_val in [
        ("MU1CurrentAngleB", 0.61),
        ("state_cb", 0.45),
        ("stnum", 0.19),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (fault_id, feat, shap_val),
        )

    # 3. IEC 61850 Packet Injection Incident (Critical Risk)
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "Substation Process Bus (IEC 61850 GOOSE)", "Injection", 94.0,
            88.8, "Critical", "Injection", "stnum, state_cb, any_relay",
        ),
    )
    inj_id = cur.lastrowid
    for feat, shap_val in [
        ("state_cb", 0.67),
        ("stnum", 0.52),
        ("any_relay", 0.38),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (inj_id, feat, shap_val),
        )

    # 4. IEC 61850 Masquerade Incident (Critical Risk)
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "Substation Process Bus (IEC 61850 GOOSE)", "Masquerade", 92.8,
            88.6, "Critical", "Masquerade", "stnum, sqNum, state_cb",
        ),
    )
    masq_id = cur.lastrowid
    for feat, shap_val in [
        ("stnum", 0.55),
        ("sqNum", 0.41),
        ("state_cb", 0.22),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (masq_id, feat, shap_val),
        )

    # 5. PMU FDI Incident (Critical Risk)
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "PMU Synchrophasor (IEEE C37.118)", "FDI", 97.8,
            89.6, "Critical", "FDI", "Actual frequency value",
        ),
    )
    fdi_id = cur.lastrowid
    for feat, shap_val in [
        ("Actual frequency value", 0.62),
        ("interarrival time", 0.24),
        ("time difference", -0.18),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (fdi_id, feat, shap_val),
        )

    # 6. PMU TSA Incident (Critical Risk)
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "PMU Synchrophasor (IEEE C37.118)", "TSA", 99.6,
            99.9, "Critical", "TSA", "time difference, interarrival time",
        ),
    )
    tsa_id = cur.lastrowid
    for feat, shap_val in [
        ("time difference", 0.72),
        ("interarrival time", 0.38),
        ("Actual frequency value", -0.05),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (tsa_id, feat, shap_val),
        )

    # 7. SCADA Telemetry Spoofing Incident (Medium Risk)
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "SCADA Telecontrol (IEC 60870-5-104)", "Telemetry Spoofing", 84.2,
            62.0, "Medium", "Telemetry Spoofing", "asduType, ioa",
        ),
    )
    spoof_id = cur.lastrowid
    for feat, shap_val in [
        ("asduType", 0.48),
        ("ioa", 0.35),
        ("cot", 0.18),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (spoof_id, feat, shap_val),
        )

    # 8. IEC 61850 Normal Baseline Telemetry
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "Substation Process Bus (IEC 61850 GOOSE)", "Normal", 98.4,
            0.2, "Normal", "Normal", "None (Nominal Substation State)",
        ),
    )
    norm1_id = cur.lastrowid
    for feat, shap_val in [
        ("sqNum", 0.02),
        ("stnum", -0.01),
        ("state_cb", 0.0),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (norm1_id, feat, shap_val),
        )

    # 9. PMU Synchrophasor Normal Baseline Telemetry
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "PMU Synchrophasor (IEEE C37.118)", "Normal", 99.2,
            0.1, "Normal", "Normal", "None (Nominal Frequency 60.0 Hz)",
        ),
    )
    norm2_id = cur.lastrowid
    for feat, shap_val in [
        ("Actual frequency value", 0.01),
        ("interarrival time", 0.01),
        ("time difference", -0.01),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (norm2_id, feat, shap_val),
        )

    # 10. SCADA Telecontrol Normal Baseline Telemetry
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "SCADA Telecontrol (IEC 60870-5-104)", "Normal", 97.9,
            0.3, "Normal", "Normal", "None (Nominal Telecontrol Polling)",
        ),
    )
    norm3_id = cur.lastrowid
    for feat, shap_val in [
        ("asduType", 0.02),
        ("cot", 0.01),
        ("ioa", 0.0),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (norm3_id, feat, shap_val),
        )

    # 11. Transmission Line Protection Normal Baseline Telemetry
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sample-load", "Power Transmission Protection (MSU/ORNL)", "Normal", 98.8,
            0.1, "Normal", "Natural", "None (Nominal Power Flow)",
        ),
    )
    norm4_id = cur.lastrowid
    for feat, shap_val in [
        ("R1-PA1:VH", 0.01),
        ("R1:F", 0.01),
        ("R1-PM1:V", 0.0),
    ]:
        cur.execute(
            "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
            (norm4_id, feat, shap_val),
        )

    conn.commit()
    print("Inserted sample events across Normal baseline (IEC 61850, PMU, SCADA, MSU) and Attacks (Replay, Fault, Injection, Masquerade, FDI, TSA, Spoofing).")


if __name__ == "__main__":
    create_database()
    conn = sqlite3.connect(DB_PATH)
    insert_demo_user(conn)
    insert_sample_readings(conn)
    insert_sample_incident(conn)
    conn.close()
    print("\nDatabase setup complete ->", DB_PATH)
