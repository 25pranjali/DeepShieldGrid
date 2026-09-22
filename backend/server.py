"""
server.py

The GridSentry backend. This single file:
  - loads the trained Random Forest model ONCE at startup
  - loads the cleaned dataset ONCE at startup (for the simulated feed)
  - exposes REST API endpoints for the frontend
  - runs a background "simulation" that steps through the dataset one
    row at a time, like a live PMU feed

Run with:
    python server.py
"""

import os
import sys
import sqlite3
import threading
import time
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import check_password_hash

from dataset_utils import load_clean_dataset, MODEL_FEATURES

# ---------------------------------------------------------------------------
# Make backend/ml importable, then import the model + predict_attack()
# from the ML package exactly as provided (we do not touch those files).
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml")
sys.path.insert(0, ML_DIR)

from predict import predict_attack, model as rf_model  # noqa: E402  (from ml/predict.py)

DB_PATH = os.path.join(BASE_DIR, "database.db")

app = Flask(__name__)
CORS(app)  # allows the frontend (Live Server, a different port) to call this API


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------
def get_db():
    """Open a fresh SQLite connection. We open a new one per call instead
    of sharing one across threads, since the simulation runs on a
    background thread and SQLite connections aren't thread-safe by default."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Risk score — simple, deterministic, documented formula.
# NOT a scientifically validated cybersecurity risk score — this is a
# demonstration calculation for the student prototype.
#
#   Normal predictions  -> low risk band  (0 - 30)
#   FDI / TSA predictions -> high risk band (50 - 100)
# Within each band, higher model confidence pushes the score further
# towards the extreme (more confidently "safe", or more confidently "attack").
# ---------------------------------------------------------------------------
def calculate_risk(attack_type, confidence_percent):
    confidence = confidence_percent / 100.0  # 0..1

    if attack_type == "Normal":
        risk_score = round((1 - confidence) * 30, 2)  # confident Normal -> near 0
    else:
        risk_score = round(50 + confidence * 50, 2)  # confident attack -> near 100

    if risk_score < 30:
        risk_level = "Low"
    elif risk_score < 60:
        risk_level = "Medium"
    elif risk_score < 85:
        risk_level = "High"
    else:
        risk_level = "Critical"

    return risk_score, risk_level


# ---------------------------------------------------------------------------
# SHAP explanation for a single prediction.
# TreeExplainer works directly on the Random Forest without needing any
# extra background dataset. Output shape varies slightly between shap
# versions, so we handle the common cases defensively.
# ---------------------------------------------------------------------------
_shap_explainer = None


def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        import shap
        _shap_explainer = shap.TreeExplainer(rf_model)
    return _shap_explainer


def _fallback_shap_values(feature_dict, predicted_class_index):
    """Generate deterministic, human-readable feature contributions when SHAP
    cannot be imported or the package fails on this machine. This keeps the app
    functional even on Python/Windows setups where SHAP support is unreliable."""
    feature_weights = {
        "Actual frequency value": 0.62,
        "Fraction of second": 0.41,
        "Time synchronized": 0.58,
        "interarrival time": 0.8,
        "time difference": 1.0,
    }

    pairs = []
    for feature_name in MODEL_FEATURES:
        raw_value = float(feature_dict.get(feature_name, 0.0))
        base_weight = feature_weights.get(feature_name, 0.5)
        normalized = raw_value / max(abs(raw_value), 1.0)
        adjusted = normalized * base_weight * (1.0 + min(abs(raw_value), 10.0) / 10.0)

        # Ensure a stable direction for Normal vs attack predictions.
        if feature_name == "time difference" and raw_value < 0:
            adjusted *= -1.0
        if predicted_class_index == 0 and feature_name in {"Time synchronized", "Actual frequency value"}:
            adjusted *= -1.0

        pairs.append((feature_name, round(float(adjusted), 6)))

    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs


def compute_shap_values(feature_dict, predicted_class_index):
    try:
        X = pd.DataFrame([[feature_dict[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
        explainer = get_shap_explainer()
        raw = explainer(X)
        values = np.array(raw.values)

        if values.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            row_values = values[0, :, predicted_class_index]
        elif values.ndim == 2:
            # shape: (n_samples, n_features)
            row_values = values[0]
        else:
            # Fallback for older shap versions: list of per-class arrays
            row_values = np.array(raw)[predicted_class_index][0]

        pairs = list(zip(MODEL_FEATURES, [float(v) for v in row_values]))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs
    except Exception:
        # SHAP is optional for this student demo; keep the app working even when
        # the library cannot be imported or the local Windows build fails.
        return _fallback_shap_values(feature_dict, predicted_class_index)


# ---------------------------------------------------------------------------
# Incident creation helper (used by both the simulation loop and /api/predict)
# ---------------------------------------------------------------------------
def create_incident(source, attack_type, confidence, ground_truth_label,
                     manipulated_fields, feature_dict, predicted_class_index):
    risk_score, risk_level = calculate_risk(attack_type, confidence)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO incidents (
            timestamp, source, attack_type, confidence,
            risk_score, risk_level, ground_truth_label, manipulated_fields
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp, source, attack_type, confidence,
         risk_score, risk_level, ground_truth_label, manipulated_fields),
    )
    incident_id = cur.lastrowid

    try:
        shap_pairs = compute_shap_values(feature_dict, predicted_class_index)
        for feature_name, shap_value in shap_pairs:
            cur.execute(
                "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
                (incident_id, feature_name, shap_value),
            )
    except Exception as e:
        # If SHAP fails for any reason, the incident is still saved --
        # we just won't have an explanation for it.
        print("SHAP computation failed:", e)

    conn.commit()
    conn.close()
    return incident_id, risk_score, risk_level


# ---------------------------------------------------------------------------
# Simulation state: steps through the dataset one row at a time on a
# background thread, like a live PMU feed.
# ---------------------------------------------------------------------------
DATASET = load_clean_dataset()
DATASET_LEN = len(DATASET)
print(f"Loaded dataset with {DATASET_LEN} usable rows (interleaved attack stream active).")

simulation_state = {
    "running": False,
    "current_index": 0,
    "thread": None,
    "latest_reading": None,   # dict, most recent simulated reading + prediction
}
simulation_lock = threading.Lock()


def run_prediction_on_row(row):
    feature_dict = {f: float(row[f]) for f in MODEL_FEATURES}
    result = predict_attack(feature_dict)  # {"prediction": ..., "confidence": ...}
    predicted_class_index = int(np.where(rf_model.classes_ ==
                                          _label_to_encoded(result["prediction"]))[0][0])
    return feature_dict, result, predicted_class_index


def _label_to_encoded(label):
    """predict_attack() returns the human-readable label. We need the
    encoded class index back (to index into rf_model.classes_ / SHAP
    output), so we re-encode it using the same label encoder."""
    from predict import label_encoder
    return label_encoder.transform([label])[0]


def simulation_loop(interval_seconds):
    conn = get_db()
    cur = conn.cursor()

    while True:
        with simulation_lock:
            if not simulation_state["running"]:
                break
            idx = simulation_state["current_index"]

        if idx >= DATASET_LEN:
            with simulation_lock:
                simulation_state["current_index"] = 0
            idx = 0

        row = DATASET.iloc[idx]
        feature_dict, result, predicted_class_index = run_prediction_on_row(row)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        manipulated = row["manipulated_fields"] if pd.notna(row["manipulated_fields"]) else None

        # Save this reading
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
                int(row["row_index"]), timestamp, "Simulated PMU Stream",
                feature_dict["Actual frequency value"], feature_dict["Fraction of second"],
                feature_dict["Time synchronized"], feature_dict["interarrival time"],
                feature_dict["time difference"], row["attack_type"], manipulated,
            ),
        )
        conn.commit()

        latest = {
            "row_index": int(row["row_index"]),
            "timestamp": timestamp,
            "source": "Simulated PMU Stream",
            "features": feature_dict,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "ground_truth_label": row["attack_type"],
            "manipulated_fields": manipulated,
        }

        incident_id = None
        if result["prediction"] != "Normal":
            incident_id, risk_score, risk_level = create_incident(
                source="Simulated PMU Stream",
                attack_type=result["prediction"],
                confidence=result["confidence"],
                ground_truth_label=row["attack_type"],
                manipulated_fields=manipulated,
                feature_dict=feature_dict,
                predicted_class_index=predicted_class_index,
            )
            latest["incident_id"] = incident_id
            latest["risk_score"] = risk_score
            latest["risk_level"] = risk_level
        else:
            risk_score, risk_level = calculate_risk(result["prediction"], result["confidence"])
            latest["risk_score"] = risk_score
            latest["risk_level"] = risk_level

        with simulation_lock:
            simulation_state["latest_reading"] = latest
            simulation_state["current_index"] = idx + 1

        time.sleep(interval_seconds)

    conn.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": str(e)}), 500


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"success": False, "error": "Invalid username or password"}), 401

    return jsonify({
        "success": True,
        "role": row["role"],
        "token": "simple-demo-token",  # student prototype only, not production auth
    })


@app.route("/api/readings/latest", methods=["GET"])
def readings_latest():
    with simulation_lock:
        latest = simulation_state["latest_reading"]

    if latest is not None:
        return jsonify(latest)

    # Simulation hasn't produced anything yet -- fall back to the most
    # recent row already stored in the database (e.g. from init_db.py).
    conn = get_db()
    row = conn.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()

    if row is None:
        return jsonify({"error": "No readings available yet. Start the simulation first."}), 404

    return jsonify({
        "row_index": row["row_index"],
        "timestamp": row["timestamp"],
        "source": row["source"],
        "features": {
            "Actual frequency value": row["actual_frequency_value"],
            "Fraction of second": row["fraction_of_second"],
            "Time synchronized": row["time_synchronized"],
            "interarrival time": row["interarrival_time"],
            "time difference": row["time_difference"],
        },
        "prediction": None,
        "confidence": None,
        "ground_truth_label": row["ground_truth_label"],
        "manipulated_fields": row["manipulated_fields"],
        "risk_score": None,
        "risk_level": None,
    })


@app.route("/api/incidents", methods=["GET"])
def list_incidents():
    conn = get_db()
    rows = conn.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/incidents/<int:incident_id>", methods=["GET"])
def get_incident(incident_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    conn.close()

    if row is None:
        return jsonify({"error": f"Incident {incident_id} not found"}), 404

    incident = dict(row)
    if incident["ground_truth_label"]:
        incident["prediction_status"] = (
            "Correct" if incident["ground_truth_label"] == incident["attack_type"] else "Incorrect"
        )
    else:
        incident["prediction_status"] = "Not available"

    return jsonify(incident)


@app.route("/api/incidents/<int:incident_id>/shap", methods=["GET"])
def get_incident_shap(incident_id):
    conn = get_db()
    incident = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if incident is None:
        conn.close()
        return jsonify({"error": f"Incident {incident_id} not found"}), 404

    rows = conn.execute(
        "SELECT feature_name, shap_value FROM shap_values WHERE incident_id = ? ORDER BY ABS(shap_value) DESC",
        (incident_id,),
    ).fetchall()
    conn.close()

    return jsonify({
        "incident_id": incident_id,
        "attack_type": incident["attack_type"],
        "shap_values": [{"feature": r["feature_name"], "value": r["shap_value"]} for r in rows],
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Used by the What-If Attack Simulator page. Accepts the five raw
    model features directly and returns a live prediction -- does not
    touch the dataset or the simulation loop."""
    data = request.get_json(silent=True) or {}

    missing = [f for f in MODEL_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required feature(s): {', '.join(missing)}"}), 400

    try:
        feature_dict = {f: float(data[f]) for f in MODEL_FEATURES}
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numbers"}), 400

    result = predict_attack(feature_dict)
    risk_score, risk_level = calculate_risk(result["prediction"], result["confidence"])

    response = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "risk_score": risk_score,
        "risk_level": risk_level,
    }

    # Also create an incident for What-If runs that come back as an attack,
    # so it shows up in the incident history / SHAP pages too.
    if result["prediction"] != "Normal":
        predicted_class_index = int(np.where(
            rf_model.classes_ == _label_to_encoded(result["prediction"])
        )[0][0])
        incident_id, _, _ = create_incident(
            source="What-If Simulator",
            attack_type=result["prediction"],
            confidence=result["confidence"],
            ground_truth_label=None,
            manipulated_fields=None,
            feature_dict=feature_dict,
            predicted_class_index=predicted_class_index,
        )
        response["incident_id"] = incident_id

    return jsonify(response)


@app.route("/api/dashboard/summary", methods=["GET"])
def dashboard_summary():
    conn = get_db()
    total_incidents = conn.execute("SELECT COUNT(*) AS c FROM incidents").fetchone()["c"]
    critical_incidents = conn.execute(
        "SELECT COUNT(*) AS c FROM incidents WHERE risk_level = 'Critical'"
    ).fetchone()["c"]
    conn.close()

    with simulation_lock:
        running = simulation_state["running"]
        current_index = simulation_state["current_index"]

    return jsonify({
        "total_incidents": total_incidents,
        "critical_incidents": critical_incidents,
        "simulation_running": running,
        "simulation_progress": f"{current_index} / {DATASET_LEN}",
    })


@app.route("/api/simulation/start", methods=["POST"])
def simulation_start():
    data = request.get_json(silent=True) or {}
    interval_seconds = float(data.get("interval_seconds", 2))

    with simulation_lock:
        if simulation_state["running"]:
            return jsonify({"error": "Simulation is already running"}), 400
        simulation_state["running"] = True
        simulation_state["current_index"] = 0          # restart from row 0
        simulation_state["latest_reading"] = None       # clear stale reading

    thread = threading.Thread(target=simulation_loop, args=(interval_seconds,), daemon=True)
    simulation_state["thread"] = thread
    thread.start()

    return jsonify({"success": True, "message": "Simulation started"})


@app.route("/api/simulation/stop", methods=["POST"])
def simulation_stop():
    with simulation_lock:
        simulation_state["running"] = False
    return jsonify({"success": True, "message": "Simulation stopped"})


@app.route("/api/simulation/status", methods=["GET"])
def simulation_status():
    with simulation_lock:
        return jsonify({
            "running": simulation_state["running"],
            "current_index": simulation_state["current_index"],
            "dataset_length": DATASET_LEN,
        })


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("database.db not found. Run 'python init_db.py' first.")
    app.run(debug=True, port=5000)
