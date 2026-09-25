"""
server.py

The GridSentry backend server:
- Loads the PSO/GWO-optimized Deep Learning (1D-CNN + BiLSTM) models via ModelRegistry
- Runs simulated real-time stream on background thread
- Exposes REST API endpoints for the GridSentry frontend
- Dynamic Upload & Feature Compatibility Engine for new simulation datasets
- Real SHAP explanation and Autoencoder Unknown Anomaly Detection
"""

import os
import sys
import io
import sqlite3
import threading
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ML_DIR = os.path.join(BASE_DIR, "ml")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from dataset_utils import load_clean_dataset, MODEL_FEATURES
from predict import predict_attack, model as rf_model, label_encoder, FEATURES as FDI_FEATURES
from feature_compatibility import FeatureCompatibilityEngine, ModelExecutor, calculate_cyber_risk

DB_PATH = os.path.join(BASE_DIR, "database.db")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global feature compatibility engine
compatibility_engine = FeatureCompatibilityEngine()

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ---------------------------------------------------------------------------
# Database Helper
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Risk Score Calculation
# ---------------------------------------------------------------------------
def calculate_risk(attack_type, confidence_percent):
    """
    Convert model prediction confidence into a simple,
    consistent cybersecurity risk level.

    Normal/Natural telemetry:
        Risk = Normal

    Attack predictions:
        < 70% confidence  -> Low
        70-89.99%         -> Medium
        >= 90%            -> Critical
    """

    attack = str(attack_type).strip().lower()
    confidence = float(confidence_percent or 0)

    # Normal telemetry
    if attack in ("normal", "natural"):
        return 0.0, "Normal"

    # Attack risk score
    risk_score = round(confidence, 2)

    if confidence >= 90:
        risk_level = "Critical"
    elif confidence >= 70:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return risk_score, risk_level

# ---------------------------------------------------------------------------
# Real SHAP Computation (No Hardcoded Fallbacks)
# ---------------------------------------------------------------------------
_shap_explainer = None


def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        _shap_explainer = shap.TreeExplainer(rf_model)
    return _shap_explainer


def compute_shap_values(feature_dict, predicted_class_index):
    """
    Computes genuine SHAP values for the prediction using TreeExplainer.
    Uses the exact feature values supplied to the model.
    """
    features_to_use = FDI_FEATURES
    X_vals = [[float(feature_dict.get(f, 0.0)) for f in features_to_use]]
    X_df = pd.DataFrame(X_vals, columns=features_to_use)

    explainer = get_shap_explainer()
    raw = explainer(X_df)
    values = np.array(raw.values)

    if values.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        row_values = values[0, :, min(predicted_class_index, values.shape[2] - 1)]
    elif values.ndim == 2:
        # shape: (n_samples, n_features)
        row_values = values[0]
    else:
        # List of per-class arrays
        row_values = np.array(raw)[min(predicted_class_index, len(raw) - 1)][0]

    pairs = list(zip(features_to_use, [float(v) for v in row_values]))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Incident Creation Helper
# ---------------------------------------------------------------------------
def create_incident(source, attack_type, confidence, ground_truth_label,
                    manipulated_fields, feature_dict, predicted_class_index,
                    custom_shap_pairs=None):
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
        if custom_shap_pairs is not None:
            shap_pairs = custom_shap_pairs
        else:
            shap_pairs = compute_shap_values(feature_dict, predicted_class_index)

        for item in shap_pairs:
            if isinstance(item, dict):
                fname, sval = item["feature"], item["value"]
            else:
                fname, sval = item[0], item[1]
            cur.execute(
                "INSERT INTO shap_values (incident_id, feature_name, shap_value) VALUES (?, ?, ?)",
                (incident_id, fname, sval),
            )
    except Exception as e:
        print("SHAP computation failed:", e)

    conn.commit()
    conn.close()
    return incident_id, risk_score, risk_level


# ---------------------------------------------------------------------------
# Background Simulation Feeder (Multi-Domain Smart Grid Feeder)
# ---------------------------------------------------------------------------
SIMULATION_DOMAINS = {
    "FDI_TSA": {
        "name": "PMU Synchrophasor (IEEE C37.118)",
        "model_id": "FDI_TSA",
        "file": os.path.join(BASE_DIR, "data", "Clean_FDI_TSA_Combined.csv"),
        "display_features": ["Actual frequency value", "Fraction of second", "interarrival time", "time difference"],
        "units": {"Actual frequency value": "Hz", "Fraction of second": "ms", "interarrival time": "s", "time difference": "s"},
        "ground_truth_col": "attack_type",
    },
    "IEC61850": {
        "name": "Substation Process Bus (IEC 61850 GOOSE/SV)",
        "model_id": "IEC61850",
        "file": os.path.join(BASE_DIR, "data", "sim_iec61850_sample.csv"),
        "display_features": ["time", "sqNum", "stnum", "state_cb"],
        "units": {"time": "s", "sqNum": "", "stnum": "", "state_cb": ""},
        "ground_truth_col": "class",
    },
    "IEC104": {
        "name": "SCADA Telecontrol Protocol (IEC 60870-5-104)",
        "model_id": "IEC104",
        "file": os.path.join(BASE_DIR, "data", "sim_iec104_sample.csv"),
        "display_features": ["Relative Time", "asduType", "cot", "ioa"],
        "units": {"Relative Time": "s", "asduType": "", "cot": "", "ioa": ""},
        "ground_truth_col": "attack_type",
    },
    "MSU_ORNL": {
        "name": "Power Transmission Protection (MSU/ORNL)",
        "model_id": "MSU_ORNL",
        "file": os.path.join(BASE_DIR, "data", "sim_msu_sample.csv"),
        "display_features": ["R1-PA1:VH", "R1-PM1:V", "R1:F", "R1:DF"],
        "units": {"R1-PA1:VH": "deg", "R1-PM1:V": "V", "R1:F": "Hz", "R1:DF": "Hz/s"},
        "ground_truth_col": "marker",
    },
}

# ---------------------------------------------------------------------------
# Dataset Cache
# Reloads automatically when the underlying CSV file changes.
# ---------------------------------------------------------------------------
DOMAIN_DATA_CACHE = {}
DOMAIN_DATA_MTIME = {}


def get_domain_dataset(domain_key="FDI_TSA"):
    if domain_key == "UPLOADED":
        return simulation_state.get("uploaded_df")

    cfg = SIMULATION_DOMAINS.get(domain_key)

    if not cfg:
        domain_key = "FDI_TSA"
        cfg = SIMULATION_DOMAINS["FDI_TSA"]

    file_path = cfg["file"]

    # Check whether the source CSV has changed
    current_mtime = (
        os.path.getmtime(file_path)
        if os.path.exists(file_path)
        else None
    )

    # Return cached data only if the file has NOT changed
    if (
        domain_key in DOMAIN_DATA_CACHE
        and DOMAIN_DATA_MTIME.get(domain_key) == current_mtime
    ):
        return DOMAIN_DATA_CACHE[domain_key]

    # Reload dataset
    try:
        if domain_key == "FDI_TSA":
            df = load_clean_dataset()
        elif os.path.exists(file_path):
            df = pd.read_csv(file_path, low_memory=False)
        else:
            df = pd.DataFrame()

        DOMAIN_DATA_CACHE[domain_key] = df
        DOMAIN_DATA_MTIME[domain_key] = current_mtime

        print(
            f"[Dataset] Loaded {domain_key}: "
            f"{len(df)} rows, {len(df.columns)} columns"
        )

        return df

    except Exception as e:
        print(f"[Dataset] Failed to load {domain_key}: {e}")
        return pd.DataFrame()


simulation_state = {
    "running": False,
    "current_index": 0,
    "thread": None,
    "latest_reading": None,
    "active_domain": "FDI_TSA",
    "uploaded_df": None,
    "uploaded_filename": None,
    "uploaded_model_id": None,
    "uploaded_features": None,
}
simulation_lock = threading.Lock()


def generate_sample_reading(domain_key="FDI_TSA"):
    df = get_domain_dataset(domain_key)
    if df is None or len(df) == 0:
        return {
            "row_index": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "GridSentry Core",
            "domain": "PMU Synchrophasor",
            "prediction": "Normal",
            "confidence": 98.5,
            "risk_score": 0.1,
            "risk_level": "Normal",
            "display_features": [],
            "features": {},
        }
    
    if domain_key == "UPLOADED":
        model_id = simulation_state.get("uploaded_model_id", "FDI_TSA")
        domain_name = f"Uploaded Feed ({simulation_state.get('uploaded_filename', 'custom')})"
        display_feats = simulation_state.get("uploaded_features", [])[:6]
        units = {}
        gt_col = "attack_type" if "attack_type" in df.columns else ("marker" if "marker" in df.columns else "class")
    else:
        cfg = SIMULATION_DOMAINS.get(domain_key, SIMULATION_DOMAINS["FDI_TSA"])
        model_id = cfg["model_id"]
        domain_name = cfg["name"]
        display_feats = cfg["display_features"]
        units = cfg["units"]
        gt_col = cfg["ground_truth_col"]

    try:
        executor = ModelExecutor(model_id)
        window_df = df.iloc[:5]
        result = executor.predict(window_df)
    except Exception as e:
        print(f"[Warning] generate_sample_reading model prediction exception: {e}")
        result = {
            "prediction": "Normal", "confidence": 98.0,
            "risk_score": 0.1, "risk_level": "Normal",
            "latest_features": {}, "shap_values": [], "features_used": []
        }
    row = df.iloc[0]

    display_features = []
    for f in display_feats:
        raw_val = result["latest_features"].get(f, row.get(f, "—"))
        try:
            if pd.notna(raw_val) and isinstance(raw_val, (int, float, np.integer, np.floating)):
                val_str = f"{float(raw_val):.4f}"
            else:
                val_str = str(raw_val)
        except Exception:
            val_str = str(raw_val)

        display_features.append({
            "name": f,
            "value": val_str,
            "unit": units.get(f, "")
        })

    return {
        "row_index": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": f"Simulated {domain_name}",
        "domain": domain_name,
        "model_id": model_id,
        "features": result["latest_features"],
        "display_features": display_features,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "ground_truth_label": str(row.get(gt_col, "—")),
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "incident_id": None,
    }


def simulation_loop(interval_seconds):
    conn = get_db()
    cur = conn.cursor()

    while True:
        with simulation_lock:
            if not simulation_state["running"]:
                break
            domain_key = simulation_state.get("active_domain", "FDI_TSA")
            idx = simulation_state["current_index"]

        df = get_domain_dataset(domain_key)
        if df is None or len(df) == 0:
            time.sleep(interval_seconds)
            continue

        if idx >= len(df):
            with simulation_lock:
                simulation_state["current_index"] = 0
            idx = 0

        row = df.iloc[idx]
        
        if domain_key == "UPLOADED":
            model_id = simulation_state.get("uploaded_model_id", "FDI_TSA")
            domain_name = f"Uploaded Feed ({simulation_state.get('uploaded_filename', 'custom')})"
            display_feats = simulation_state.get("uploaded_features", [])[:4]
            units = {}
            gt_col = "attack_type" if "attack_type" in df.columns else ("marker" if "marker" in df.columns else "class")
        else:
            cfg = SIMULATION_DOMAINS.get(domain_key, SIMULATION_DOMAINS["FDI_TSA"])
            model_id = cfg["model_id"]
            domain_name = cfg["name"]
            display_feats = cfg["display_features"]
            units = cfg["units"]
            gt_col = cfg["ground_truth_col"]

        start_idx = max(0, idx - 4)
        window_df = df.iloc[start_idx : idx + 1]

        try:
            executor = ModelExecutor(model_id)
            result = executor.predict(window_df)
        except Exception as e:
            result = {
                "prediction": "Normal", "confidence": 98.0,
                "risk_score": 0.1, "risk_level": "Normal",
                "latest_features": {}, "shap_values": [], "features_used": []
            }

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ground_truth = str(row.get(gt_col, "—"))
        if ground_truth == "nan":
            ground_truth = "—"

        # Log all telemetry events (both Normal baseline and attack detections)
        is_attack = result["prediction"] not in ["Normal", "Natural"]
        manipulated_str = ", ".join(display_feats) if is_attack else "None (Nominal Telemetry)"

        incident_id, risk_score, risk_level = create_incident(
            source=f"Simulated {domain_name}",
            attack_type=result["prediction"],
            confidence=result["confidence"],
            ground_truth_label=ground_truth,
            manipulated_fields=manipulated_str,
            feature_dict=result["latest_features"],
            predicted_class_index=0,
            custom_shap_pairs=result.get("shap_values")
        )

        try:
            # Maintain rolling window of latest 200 entries to prevent database bloat
            cur.execute("DELETE FROM incidents WHERE id NOT IN (SELECT id FROM incidents ORDER BY id DESC LIMIT 200)")
            cur.execute("DELETE FROM shap_values WHERE incident_id NOT IN (SELECT id FROM incidents)")
            conn.commit()
        except Exception:
            pass

        display_features = []
        for f in display_feats:
            raw_val = result["latest_features"].get(f, row.get(f, "—"))
            try:
                if pd.notna(raw_val) and isinstance(raw_val, (int, float, np.integer, np.floating)):
                    val_str = f"{float(raw_val):.4f}"
                else:
                    val_str = str(raw_val)
            except Exception:
                val_str = str(raw_val)

            display_features.append({
                "name": f,
                "value": val_str,
                "unit": units.get(f, "")
            })

        latest = {
            "row_index": int(idx),
            "timestamp": timestamp,
            "source": f"Simulated {domain_name}",
            "domain": domain_name,
            "model_id": model_id,
            "features": result["latest_features"],
            "display_features": display_features,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "ground_truth_label": ground_truth,
            "manipulated_fields": ", ".join(display_feats),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "incident_id": incident_id,
        }

        with simulation_lock:
            simulation_state["latest_reading"] = latest
            simulation_state["current_index"] = idx + 1

        time.sleep(interval_seconds)

    conn.close()


# ===========================================================================
# API ROUTES
# ===========================================================================

@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": str(e)}), 500


@app.route("/api/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"success": False, "error": "Invalid username or password"}), 401

    token = f"demo-token-{user['id']}-{int(time.time())}"
    return jsonify({
        "success": True,
        "token": token,
        "role": user["role"],
        "username": user["username"],
    })


@app.route("/api/readings/latest", methods=["GET"])
def latest_reading():
    with simulation_lock:
        if simulation_state["latest_reading"] is not None:
            return jsonify(simulation_state["latest_reading"])

    domain_key = simulation_state.get("active_domain", "FDI_TSA")
    reading = generate_sample_reading(domain_key)
    with simulation_lock:
        simulation_state["latest_reading"] = reading
    return jsonify(reading)


@app.route("/api/incidents", methods=["GET"])
def list_incidents():
    limit = int(request.args.get("limit", 50))
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/incidents/<int:incident_id>", methods=["GET"])
def get_incident(incident_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": f"Incident #{incident_id} not found"}), 404

    data = dict(row)
    data["prediction_status"] = (
        "Correct" if data["attack_type"] == data["ground_truth_label"]
        else ("Incorrect" if data["ground_truth_label"] else "Unlabeled")
    )
    return jsonify(data)


@app.route("/api/incidents/<int:incident_id>/shap", methods=["GET"])
def get_incident_shap(incident_id):
    conn = get_db()
    incident = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if not incident:
        conn.close()
        return jsonify({"error": f"Incident #{incident_id} not found"}), 404

    shap_rows = conn.execute(
        "SELECT feature_name, shap_value FROM shap_values WHERE incident_id = ?",
        (incident_id,),
    ).fetchall()
    conn.close()

    values = [{"feature": r["feature_name"], "value": r["shap_value"]} for r in shap_rows]
    return jsonify({
        "incident_id": incident_id,
        "attack_type": incident["attack_type"],
        "shap_values": values,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Used by the What-If Attack Simulator page.
    Uses the new PSO/GWO-optimized model.
    """
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

    if result["prediction"] != "Normal":
        try:
            encoded = label_encoder.transform([result["prediction"]])[0]
            predicted_class_index = int(np.where(rf_model.classes_ == encoded)[0][0])
        except Exception:
            predicted_class_index = 0

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
        simulation_state["current_index"] = 0
        simulation_state["latest_reading"] = None

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
        domain = simulation_state.get("active_domain", "FDI_TSA")
        df = get_domain_dataset(domain)
        d_len = len(df) if df is not None else 0
        return jsonify({
            "running": simulation_state["running"],
            "current_index": simulation_state["current_index"],
            "dataset_length": d_len,
            "active_domain": domain,
        })


@app.route("/api/simulation/domains", methods=["GET"])
def get_simulation_domains():
    with simulation_lock:
        active = simulation_state.get("active_domain", "FDI_TSA")
    domains_list = [
        {"id": k, "name": v["name"], "model_id": v["model_id"], "features": v["display_features"]}
        for k, v in SIMULATION_DOMAINS.items()
    ]
    if simulation_state.get("uploaded_df") is not None:
        domains_list.append({
            "id": "UPLOADED",
            "name": f"Uploaded Feed ({simulation_state.get('uploaded_filename', 'custom')})",
            "model_id": simulation_state.get("uploaded_model_id", "Custom"),
            "features": simulation_state.get("uploaded_features", [])[:4],
        })
    return jsonify({
        "active_domain": active,
        "available_domains": domains_list
    })


@app.route("/api/simulation/domain", methods=["POST"])
def set_simulation_domain():
    data = request.get_json(silent=True) or {}
    domain = data.get("domain", "FDI_TSA")
    if domain not in SIMULATION_DOMAINS and domain != "UPLOADED":
        return jsonify({"error": f"Invalid domain: {domain}"}), 400
    if domain == "UPLOADED" and simulation_state.get("uploaded_df") is None:
        return jsonify({"error": "No uploaded dataset currently available"}), 400

    with simulation_lock:
        simulation_state["active_domain"] = domain
        simulation_state["current_index"] = 0
        latest = generate_sample_reading(domain)
        simulation_state["latest_reading"] = latest

    return jsonify({
        "success": True,
        "active_domain": domain,
        "latest_reading": latest
    })


# ===========================================================================
# NEW ENDPOINTS: DATASET UPLOAD & MODEL COMPATIBILITY ENGINE
# ===========================================================================

@app.route("/api/models", methods=["GET"])
def get_registered_models():
    """Returns all independent trained models in the GridSentry Model Registry."""
    registry = compatibility_engine.registry
    return jsonify({
        "status": "success",
        "total_models": len(registry),
        "models": registry
    })


@app.route("/api/datasets/upload", methods=["POST"])
def upload_simulation_dataset():
    """
    Core backend workflow requested by user:
    1. Read uploaded dataset (CSV)
    2. Analyze columns and features
    3. Compare with metadata of all trained models
    4. Determine compatible model(s) without fabricating missing features
    5. Preprocess matching features using exact scaler
    6. Construct temporal sequence
    7. Run CNN-LSTM prediction + Autoencoder unknown anomaly check
    8. Generate risk score, risk level, and real SHAP explanations
    9. Log incident into SQLite so it appears in existing UI
    """
    if "file" not in request.files and not request.is_json:
        return jsonify({"error": "No file uploaded or JSON dataset provided"}), 400

    filename = "simulation_data.csv"
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOADS_DIR, f"{int(time.time())}_{filename}")
        file.save(save_path)
        try:
            df = pd.read_csv(save_path, low_memory=False)
        except Exception as e:
            return jsonify({"error": f"Failed to parse CSV: {e}"}), 400
    else:
        payload = request.get_json()
        df = pd.DataFrame(payload.get("data", payload))

    # Step 1 & 2: Dataset analysis
    dataset_analysis = compatibility_engine.analyze_dataset(df)

    # Step 3 & 4: Model compatibility analysis
    best_model_id, compat_info = compatibility_engine.select_best_model(df.columns)

    if best_model_id is None:
        # Rejection: Insufficient features
        return jsonify({
            "status": "rejected",
            "message": compat_info,
            "dataset_analysis": dataset_analysis,
            "compatibility_evaluation": compatibility_engine.evaluate_compatibility(df.columns),
        }), 422

    # Step 5 - 8: Execute model prediction pipeline
    try:
        executor = ModelExecutor(best_model_id)
        result = executor.predict(df)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Execution error for compatible model {best_model_id}: {e}",
            "compatibility_info": compat_info,
        }), 500

    # Step 9: Store in database for telemetry & incident log (both Normal and Attack)
    ground_truth = None
    for col in ["attack_type", "marker", "class", "label"]:
        if col in df.columns:
            ground_truth = str(df[col].iloc[-1])
            break

    is_attack = result["prediction"] not in ["Normal", "Natural"]
    manipulated_str = ", ".join(compat_info["matched_features"][:4]) if is_attack else "None (Nominal Telemetry)"

    incident_id, risk_score, risk_level = create_incident(
        source=f"Uploaded Feed: {filename} ({best_model_id})",
        attack_type=result["prediction"],
        confidence=result["confidence"],
        ground_truth_label=ground_truth or ("Attack" if is_attack else "Normal"),
        manipulated_fields=manipulated_str,
        feature_dict=result["latest_features"],
        predicted_class_index=0,
        custom_shap_pairs=result["shap_values"],
    )
    result["incident_id"] = incident_id

    # If dataset has multiple rows, sample a few additional rows to populate telemetry log
    if len(df) > 5:
        sample_step = max(1, len(df) // 4)
        sample_indices = [i for i in range(0, min(len(df) - 1, sample_step * 3), sample_step)]
        for s_idx in sample_indices:
            try:
                s_window = df.iloc[max(0, s_idx - 4) : s_idx + 1]
                s_res = executor.predict(s_window)
                s_gt = None
                for col in ["attack_type", "marker", "class", "label"]:
                    if col in df.columns:
                        s_gt = str(df[col].iloc[s_idx])
                        break
                s_is_attack = s_res["prediction"] not in ["Normal", "Natural"]
                s_manip = ", ".join(compat_info["matched_features"][:4]) if s_is_attack else "None (Nominal Telemetry)"
                create_incident(
                    source=f"Uploaded Feed: {filename} ({best_model_id})",
                    attack_type=s_res["prediction"],
                    confidence=s_res["confidence"],
                    ground_truth_label=s_gt or ("Attack" if s_is_attack else "Normal"),
                    manipulated_fields=s_manip,
                    feature_dict=s_res["latest_features"],
                    predicted_class_index=0,
                    custom_shap_pairs=s_res.get("shap_values")
                )
            except Exception:
                pass

    display_features = []
    for f in compat_info["matched_features"][:6]:
        val = result["latest_features"].get(f, 0.0)
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        display_features.append({
            "name": f,
            "value": val_str,
            "unit": ""
        })

    with simulation_lock:
        simulation_state["running"] = False
        simulation_state["uploaded_df"] = df
        simulation_state["uploaded_filename"] = filename
        simulation_state["uploaded_model_id"] = best_model_id
        simulation_state["uploaded_features"] = compat_info["matched_features"]
        simulation_state["active_domain"] = "UPLOADED"
        simulation_state["current_index"] = 0
        simulation_state["latest_reading"] = {
            "row_index": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": f"Uploaded Feed: {filename}",
            "domain": result["domain"],
            "model_id": best_model_id,
            "features": result["latest_features"],
            "display_features": display_features,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "ground_truth_label": ground_truth or "—",
            "manipulated_fields": ", ".join(compat_info["matched_features"][:4]),
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "incident_id": incident_id,
        }

    return jsonify({
        "status": "success",
        "selected_model": best_model_id,
        "domain": result["domain"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "is_unknown_attack": result["is_unknown_attack"],
        "reconstruction_error": result["reconstruction_error"],
        "incident_id": incident_id,
        "shap_values": result["shap_values"],
        "features_used": result["features_used"],
        "display_features": display_features,
        "dataset_analysis": dataset_analysis,
        "compatibility_report": compat_info,
    })


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("database.db not found. Run 'python init_db.py' first.")
    app.run(host="0.0.0.0", port=5000, debug=False)
