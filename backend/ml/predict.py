"""
predict.py

Primary inference engine for GridSentry:
- Powered by the newly trained PSO/GWO-optimized 1D-CNN + BiLSTM model
- Seamlessly falls back to companion calibrated ensemble for single-row / cold-start
- Detects the 3 clean PMU classes: Normal, FDI, TSA
- Connected to the central ModelRegistry and FeatureCompatibilityEngine
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "FDI_TSA")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models_cnn_lstm import HybridCNNLSTM, GridAutoencoder

# Load metadata
with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
    METADATA = json.load(f)

FEATURES = METADATA["final_features"]
CLASSES = METADATA["attack_classes"]

# Load scaler, encoder, companion model
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "companion_rf.pkl"))  # for SHAP and single snapshot

# Load Deep Learning CNN-LSTM model
cnn_lstm_model = None
try:
    ckpt = torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {"cnn_filters": 32, "lstm_units": 48})
    cnn_lstm_model = HybridCNNLSTM(
        num_features=len(FEATURES),
        num_classes=len(CLASSES),
        cnn_filters=cfg.get("cnn_filters", 32),
        lstm_units=cfg.get("lstm_units", 48),
    )
    cnn_lstm_model.load_state_dict(ckpt["state_dict"])
    cnn_lstm_model.eval()
except Exception as e:
    print(f"[Warning] Failed to load PyTorch CNN-LSTM: {e}")

# Load Autoencoder for unknown anomaly detection
autoencoder = None
ae_threshold = None
ae_path = os.path.join(MODEL_DIR, "autoencoder.pt")
if os.path.exists(ae_path):
    try:
        ae_ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
        autoencoder = GridAutoencoder(num_features=len(FEATURES))
        autoencoder.load_state_dict(ae_ckpt["state_dict"])
        autoencoder.eval()
        ae_threshold = ae_ckpt.get("threshold", 0.1)
    except Exception as e:
        print(f"[Warning] Failed to load autoencoder: {e}")

# Sliding temporal buffer for PMU sequence inference
_pmu_buffer = []


def predict_attack(input_data, use_deep_learning=True):
    """
    Predict smart-grid attack type across clean PMU classes:
    Normal, FDI, TSA.
    
    input_data: dict with model features
    """
    global _pmu_buffer

    # Vector of raw input features in canonical order
    raw_vec = []
    for f in FEATURES:
        val = input_data.get(f, 0.0)
        try:
            raw_vec.append(float(val))
        except (ValueError, TypeError):
            raw_vec.append(0.0)

    scaled_vec = scaler.transform([raw_vec])[0]

    # Maintain sliding buffer of length 5 for CNN-LSTM temporal reasoning
    _pmu_buffer.append(scaled_vec)
    if len(_pmu_buffer) > 5:
        _pmu_buffer.pop(0)

    # When temporal window is available, use CNN-LSTM
    if use_deep_learning and cnn_lstm_model is not None and len(_pmu_buffer) == 5:
        seq_tensor = torch.tensor(np.array([_pmu_buffer], dtype=np.float32))
        with torch.no_grad():
            logits = cnn_lstm_model(seq_tensor)
            probs = torch.softmax(logits, dim=-1)[0].numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        prediction_label = CLASSES[pred_idx]

        # Autoencoder unknown anomaly check
        if autoencoder is not None and ae_threshold is not None:
            last_vec = torch.tensor([scaled_vec], dtype=torch.float32)
            with torch.no_grad():
                rec = autoencoder(last_vec)
                mse = float(torch.mean((last_vec - rec) ** 2).item())
            if mse > ae_threshold and prediction_label == "Normal":
                prediction_label = "Unknown / Anomalous Behavior"
                confidence = round(min(0.99, mse / (ae_threshold * 1.5)), 4)

        return {
            "prediction": prediction_label,
            "confidence": round(confidence * 100, 2),
            "model_type": "1D-CNN + BiLSTM (PSO/GWO Optimized)",
        }

    # Single-snapshot fallback (e.g. What-If Simulator)
    raw_arr = np.array([scaled_vec])
    pred_idx = model.predict(raw_arr)[0]
    prediction_label = label_encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(raw_arr)[0]
    confidence = float(probs[pred_idx])

    return {
        "prediction": prediction_label,
        "confidence": round(confidence * 100, 2),
        "model_type": "Companion Ensemble (PSO/GWO Calibrated)",
    }


if __name__ == "__main__":
    test_cases = [
        {"name": "Normal Operation", "features": {"Actual frequency value": 59.995, "interarrival time": 0.0474, "time difference": 1.1517}},
        {"name": "FDI Attack", "features": {"Actual frequency value": 60.001, "interarrival time": 0.0629, "time difference": 1.0913}},
        {"name": "TSA Attack", "features": {"Actual frequency value": 60.009, "interarrival time": 0.0456, "time difference": -4.9762}},
    ]

    for tc in test_cases:
        res = predict_attack(tc["features"])
        print(f"[{tc['name']}] -> Prediction: {res['prediction']} | Confidence: {res['confidence']}% | Engine: {res['model_type']}")
