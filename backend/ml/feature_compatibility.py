"""
feature_compatibility.py

Feature Compatibility and Dynamic Model Selection Engine for GridSentry:
- Analyzes uploaded simulation/test datasets
- Resolves verified semantic aliases using feature_mapping.json
- Evaluates feature overlap and minimum required features against all models in model_registry.json
- Selects the compatible trained model without fabricating missing data
- Prepares exact sequences and runs CNN-LSTM inference + Autoencoder anomaly check + SHAP explanation + Risk assessment
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ML_DIR = os.path.join(BASE_DIR, "ml")
MODELS_ROOT = os.path.join(PROJECT_DIR, "models")
MAPPING_FILE = os.path.join(BASE_DIR, "feature_mapping.json")
REGISTRY_FILE = os.path.join(MODELS_ROOT, "model_registry.json")

if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from models_cnn_lstm import HybridCNNLSTM, GridAutoencoder


def load_feature_mapping():
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "r") as f:
                data = json.load(f)
                return data.get("mappings", {})
        except Exception:
            pass
    return {}


def load_model_registry():
    paths = [
        REGISTRY_FILE,
        os.path.join(ML_DIR, "model_registry.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def resolve_aliases(columns):
    """
    Given a list of column names, map them to canonical feature names
    using feature_mapping.json. Returns dict: canonical_name -> original_col_name
    """
    mappings = load_feature_mapping()
    resolved = {}
    
    # Inverted map: alias.lower() -> canonical
    inv_map = {}
    for canonical, aliases in mappings.items():
        inv_map[canonical.lower()] = canonical
        for a in aliases:
            inv_map[a.lower()] = canonical

    for col in columns:
        col_clean = str(col).strip()
        col_lower = col_clean.lower()
        # Always preserve the exact raw column name
        resolved[col_clean] = col_clean
        if col_lower in inv_map:
            canonical = inv_map[col_lower]
            if canonical not in resolved:
                resolved[canonical] = col_clean

    return resolved


class FeatureCompatibilityEngine:
    def __init__(self):
        self.registry = load_model_registry()
        self.feature_mappings = load_feature_mapping()

    def refresh_registry(self):
        self.registry = load_model_registry()

    def analyze_dataset(self, df):
        """
        Deep analysis of uploaded dataset columns, data types, missingness, and aliases.
        """
        num_rows, num_cols = df.shape
        columns = list(df.columns)
        resolved_map = resolve_aliases(columns)

        num_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_list = df.select_dtypes(exclude=[np.number]).columns.tolist()

        missing_dict = df.isnull().sum()
        missing_dict = missing_dict[missing_dict > 0].to_dict()

        return {
            "num_rows": num_rows,
            "num_columns": num_cols,
            "columns": columns,
            "numerical_columns": num_cols_list,
            "categorical_columns": cat_cols_list,
            "missing_values": missing_dict,
            "resolved_aliases": resolved_map,
        }

    def evaluate_compatibility(self, uploaded_columns):
        """
        Compare uploaded columns against every model in model_registry.json.
        Returns detailed compatibility ranking.
        """
        self.refresh_registry()
        resolved_map = resolve_aliases(uploaded_columns)
        available_canonical = set(resolved_map.keys())

        results = {}

        for model_id, meta in self.registry.items():
            req_features = meta.get("final_features", meta.get("selected_features", []))
            min_features = meta.get("minimum_compatible_features", req_features[:2])

            req_set = set(req_features)
            min_set = set(min_features)

            matched_canonical = available_canonical.intersection(req_set)
            missing_features = list(req_set - available_canonical)
            additional_features = [c for c in uploaded_columns if c not in req_set]

            has_min = min_set.issubset(available_canonical)
            compat_ratio = len(matched_canonical) / max(len(req_features), 1)

            if has_min and compat_ratio >= 0.70:
                status = "COMPATIBLE"
            elif has_min:
                status = "PARTIALLY_COMPATIBLE"
            else:
                status = "INSUFFICIENT_FEATURES"

            # Reconstruct original column names for matched features
            matched_original = [resolved_map[feat] for feat in matched_canonical if feat in resolved_map]

            results[model_id] = {
                "model_id": model_id,
                "domain": meta.get("domain", model_id),
                "status": status,
                "compatibility_score_percent": round(compat_ratio * 100, 1),
                "has_minimum_features": has_min,
                "matched_features_count": len(matched_canonical),
                "required_features_count": len(req_features),
                "matched_features": sorted(list(matched_canonical)),
                "matched_original_columns": matched_original,
                "missing_required_features": sorted(missing_features),
                "additional_unused_features": additional_features[:10],
                "minimum_required_features": min_features,
            }

        return results

    def select_best_model(self, uploaded_columns):
        """
        Selects the single most compatible model.
        Returns: (model_id, compat_info) or (None, rejection_reason)
        """
        eval_results = self.evaluate_compatibility(uploaded_columns)
        
        # Filter for models that meet minimum feature criteria
        eligible = [res for res in eval_results.values() if res["has_minimum_features"]]
        if not eligible:
            # Rejection message
            reasons = []
            for m_id, res in eval_results.items():
                reasons.append(
                    f"{m_id}: missing minimum features {res['minimum_required_features']}"
                )
            return None, f"Insufficient compatible features for all trained models. ({'; '.join(reasons)})"

        # Sort by compatibility score descending
        eligible.sort(key=lambda x: x["compatibility_score_percent"], reverse=True)
        best = eligible[0]
        return best["model_id"], best


# ---------------------------------------------------------------------------
# Dynamic Model Executor
# ---------------------------------------------------------------------------
class ModelExecutor:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = os.path.join(MODELS_ROOT, model_id)
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Load metadata
        with open(os.path.join(self.model_dir, "metadata.json"), "r") as f:
            self.meta = json.load(f)

        self.features = self.meta["final_features"]
        self.classes = self.meta["attack_classes"]
        self.seq_len = self.meta.get("sequence_length", 5)

        # Load scaler & encoder
        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
        self.encoder = joblib.load(os.path.join(self.model_dir, "encoder.pkl"))

        # Load CNN-LSTM model
        ckpt = torch.load(os.path.join(self.model_dir, "model.pt"), map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {"cnn_filters": 32, "lstm_units": 48})
        self.model = HybridCNNLSTM(
            num_features=len(self.features),
            num_classes=len(self.classes),
            cnn_filters=cfg.get("cnn_filters", 32),
            lstm_units=cfg.get("lstm_units", 48),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # Load Autoencoder if present
        self.autoencoder = None
        self.ae_threshold = None
        ae_path = os.path.join(self.model_dir, "autoencoder.pt")
        if os.path.exists(ae_path):
            try:
                ae_ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
                self.autoencoder = GridAutoencoder(num_features=len(self.features))
                self.autoencoder.load_state_dict(ae_ckpt["state_dict"])
                self.autoencoder.eval()
                self.ae_threshold = ae_ckpt.get("threshold", 0.1)
            except Exception as e:
                print(f"[Warning] Failed to load autoencoder: {e}")

        # Load companion RF for SHAP TreeExplainer
        self.companion_rf = None
        rf_path = os.path.join(self.model_dir, "companion_rf.pkl")
        if os.path.exists(rf_path):
            self.companion_rf = joblib.load(rf_path)
            try:
                self.explainer = shap.TreeExplainer(self.companion_rf)
            except Exception:
                self.explainer = None

    def prepare_data(self, df):
        """
        Extract only the required features using alias mapping.
        Does NOT invent missing features.
        """
        resolved_map = resolve_aliases(df.columns)
        
        # Check if any required feature is missing
        missing = [f for f in self.features if f not in resolved_map]
        if missing:
            raise ValueError(
                f"Insufficient compatible features for {self.model_id}. Missing features: {missing}"
            )

        def _parse_val(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            s = str(v).strip()
            if s.lower() == "true":
                return 1.0
            if s.lower() == "false":
                return 0.0
            try:
                return pd.to_timedelta(s).total_seconds()
            except Exception:
                pass
            try:
                return float(s)
            except Exception:
                return np.nan

        extracted = {}
        for f in self.features:
            orig_col = resolved_map[f]
            val = df[orig_col].copy()
            if not pd.api.types.is_numeric_dtype(val):
                val = val.apply(_parse_val)
            else:
                val = pd.to_numeric(val, errors="coerce")
            extracted[f] = val.values

        extracted_df = pd.DataFrame(extracted)
        # Drop rows with NAs
        extracted_clean = extracted_df.dropna().reset_index(drop=True)
        if len(extracted_clean) == 0:
            raise ValueError("All extracted rows contained invalid/NaN values.")

        scaled = self.scaler.transform(extracted_clean.values)
        return scaled, extracted_clean

    def predict(self, df):
        """
        Runs complete inference pipeline:
        1. Preprocess & scale
        2. Sequence construction
        3. CNN-LSTM prediction
        4. Autoencoder anomaly check
        5. Risk assessment
        6. Real SHAP calculation
        """
        scaled, clean_df = self.prepare_data(df)
        n_samples = len(scaled)

        # Make sequences
        if n_samples < self.seq_len:
            pad = np.tile(scaled[0], (self.seq_len - n_samples, 1))
            seq_arr = np.array([np.vstack([pad, scaled])], dtype=np.float32)
            eval_idx = -1
        else:
            # We predict the last available window (most recent reading)
            seq_arr = np.array([scaled[-self.seq_len :]], dtype=np.float32)
            eval_idx = -1

        with torch.no_grad():
            logits = self.model(torch.tensor(seq_arr))
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_class = self.classes[pred_idx]

        # Autoencoder Unknown Anomaly Check
        is_unknown = False
        reconstruction_error = 0.0
        if self.autoencoder is not None and self.ae_threshold is not None:
            last_vec = torch.tensor(scaled[-1:], dtype=torch.float32)
            with torch.no_grad():
                rec = self.autoencoder(last_vec)
                reconstruction_error = float(torch.mean((last_vec - rec) ** 2).item())

            # If error is abnormally high, flag as Unknown/Anomalous
            if reconstruction_error > self.ae_threshold and predicted_class == "Normal":
                predicted_class = "Unknown / Anomalous Behavior"
                confidence = round(min(0.99, reconstruction_error / (self.ae_threshold * 1.5)), 4)
                is_unknown = True

        # Risk assessment
        risk_score, risk_level = calculate_cyber_risk(predicted_class, confidence * 100)

        # SHAP calculation on the evaluated vector
        shap_values_list = []
        if self.companion_rf is not None and self.explainer is not None:
            try:
                target_vec = scaled[-1:]
                sv = self.explainer.shap_values(target_vec)
                # Handle varying SHAP output formats (binary/multiclass list or array)
                if isinstance(sv, list):
                    # Multi-class list: choose the predicted class slice
                    cls_shap = sv[min(pred_idx, len(sv) - 1)][0]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    cls_shap = sv[0, :, min(pred_idx, sv.shape[2] - 1)]
                else:
                    cls_shap = sv[0]

                for feat_name, val in zip(self.features, cls_shap):
                    shap_values_list.append({
                        "feature": feat_name,
                        "value": float(val)
                    })
            except Exception as e:
                print(f"[Warning] SHAP calculation error: {e}")

        # Return latest reading values
        latest_features = {f: float(clean_df[f].iloc[-1]) for f in self.features}

        return {
            "model_id": self.model_id,
            "domain": self.meta.get("domain", self.model_id),
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "is_unknown_attack": is_unknown,
            "reconstruction_error": round(reconstruction_error, 5),
            "features_used": self.features,
            "latest_features": latest_features,
            "shap_values": shap_values_list,
        }


def calculate_cyber_risk(attack_type: str, confidence_pct: float):
    """
    Standardized Smart Grid Cyber-Physical Risk Assessment:
    - Normal / Natural -> Score: 0.0 - 9.9, Level: "Normal" (Operating nominal)
    - Low Risk (10.0 - 39.9) -> Minor transient anomalies, low-confidence perturbations, reconnaissance
    - Medium Risk (40.0 - 69.9) -> Moderate attacks: Replay, Fault, Telemetry Spoofing, moderate FDI
    - Critical Risk (70.0 - 100.0) -> Catastrophic attacks: Command Injection, TSA, Masquerade, Packet Injection, Breaker Trip
    """
    conf = max(0.0, min(100.0, float(confidence_pct))) / 100.0

    if attack_type in ["Normal", "Natural"]:
        # Genuine normal state: explicitly Normal (NOT a risk state)
        score = round((1.0 - conf) * 4.0, 1)
        level = "Normal"
        return score, level

    # 1. Critical tier (Immediate grid hazard: breaker tripping, desynchronization, masquerade, packet injection)
    if any(k in attack_type for k in ["Command Injection", "TSA", "Time Synchronization", "Breaker Trip"]):
        score = round(75.0 + conf * 25.0, 1)  # 85.0 - 100.0 -> Critical
    elif any(k in attack_type for k in ["Masquerade", "Injection"]) or (attack_type == "FDI" and conf >= 0.90):
        score = round(70.0 + conf * 20.0, 1)  # 70.0 - 90.0 -> Critical

    # 2. Medium tier (Process disruption: Replay attacks, Faults, Telemetry Spoofing, moderate FDI, unknown anomalies)
    elif any(k in attack_type for k in ["Replay", "Fault", "Spoofing", "Set-Point", "Unknown", "Anomalous", "FDI"]):
        score = round(40.0 + conf * 28.0, 1)  # 45.0 - 68.0 -> Medium

    # 3. Low tier (Minor noise, parameter exploration, scanning)
    else:
        score = round(15.0 + conf * 22.0, 1)  # 15.0 - 37.0 -> Low

    score = min(100.0, max(0.0, score))

    if score < 10.0:
        level = "Normal"
    elif score < 40.0:
        level = "Low"
    elif score < 70.0:
        level = "Medium"
    else:
        level = "Critical"

    return score, level

