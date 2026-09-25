"""
train_all_models.py

Master multi-dataset training pipeline for GridSentry:
Trains 4 INDEPENDENT models without merging datasets:
1. FDI_TSA_Model (PMU Synchrophasor IEEE C37.118: Normal, FDI, TSA)
2. MSU_ORNL_Model (Transmission Line Protection: Natural, Attack)
3. IEC104_Model (SCADA Network Telecontrol: Normal + IEC-104 Attacks)
4. IEC61850_Model (Digital Substation Process Bus: Normal, Masquerade, Injection, Replay, Fault)

Each pipeline follows:
PREPROCESSING -> PSO FEATURE SELECTION -> GWO FEATURE SELECTION -> CNN-LSTM -> METRICS & ARTIFACT PERSISTENCE
"""

import os
import sys
import time
import json
import zipfile
import io
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure backend/ml is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ML_DIR = os.path.join(BASE_DIR, "ml")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")
MODELS_ROOT = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_ROOT, exist_ok=True)

if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from pso_feature_selection import PSOFeatureSelector
from gwo_feature_selection import GWOFeatureSelector
from models_cnn_lstm import HybridCNNLSTM, GridAutoencoder

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helper: Create Sliding Sequences
# ---------------------------------------------------------------------------
def make_sequences(X_arr, y_arr, seq_len=5):
    """
    Constructs sliding temporal windows of length seq_len.
    Returns: X_seq (N, seq_len, F), y_seq (N,)
    """
    if len(X_arr) < seq_len:
        # Pad if shorter than seq_len
        pad = np.tile(X_arr[0], (seq_len - len(X_arr), 1))
        X_padded = np.vstack([pad, X_arr])
        return np.array([X_padded]), np.array([y_arr[-1]])

    X_seq = []
    y_seq = []
    for i in range(len(X_arr) - seq_len + 1):
        X_seq.append(X_arr[i : i + seq_len])
        y_seq.append(y_arr[i + seq_len - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


# ---------------------------------------------------------------------------
# Helper: Train PyTorch CNN-LSTM Model
# ---------------------------------------------------------------------------
def train_cnn_lstm(
    X_train_seq,
    y_train_seq,
    X_test_seq,
    y_test_seq,
    num_features,
    num_classes,
    epochs=10,
    batch_size=64,
    lr=0.003,
):
    model = HybridCNNLSTM(
        num_features=num_features,
        num_classes=num_classes,
        cnn_filters=32,
        lstm_units=48,
        lstm_layers=2,
        dropout=0.25,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    start_train = time.time()
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    training_time = round(time.time() - start_train, 2)

    # Evaluate on test set
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        test_x = torch.tensor(X_test_seq)
        test_logits = model(test_x)
        test_probs = torch.softmax(test_logits, dim=-1).numpy()
        test_preds = np.argmax(test_probs, axis=1)
    inference_time_per_sample_ms = round(((time.time() - t0) / max(len(X_test_seq), 1)) * 1000, 4)

    return model, test_preds, test_probs, training_time, inference_time_per_sample_ms


# ---------------------------------------------------------------------------
# Helper: Train Domain Autoencoder for Unknown Anomaly Detection
# ---------------------------------------------------------------------------
def train_autoencoder(X_normal_scaled, num_features, epochs=8, batch_size=64):
    ae = GridAutoencoder(num_features=num_features, latent_dim=min(8, max(2, num_features // 2)))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.005)

    ds = TensorDataset(torch.tensor(X_normal_scaled, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ae.train()
    for epoch in range(epochs):
        for (bx,) in loader:
            optimizer.zero_grad()
            rec = ae(bx)
            loss = criterion(rec, bx)
            loss.backward()
            optimizer.step()

    ae.eval()
    with torch.no_grad():
        errors = ae.get_reconstruction_error(torch.tensor(X_normal_scaled, dtype=torch.float32))
    threshold = float(np.mean(errors) + 3.0 * np.std(errors))
    return ae, threshold


# ===========================================================================
# 1. PMU FDI / TSA MODEL TRAINING
# ===========================================================================
def train_fdi_tsa_pipeline():
    print("\n" + "=" * 70)
    print("TRAINING MODEL 1: PMU FDI / TSA MODEL (IEEE C37.118)")
    print("=" * 70)

    model_dir = os.path.join(MODELS_ROOT, "FDI_TSA")
    os.makedirs(model_dir, exist_ok=True)

    csv_path = os.path.join(DATASETS_DIR, "Clean_FDI_TSA_Combined.csv")
    df = pd.read_csv(csv_path, low_memory=False)

    def _parse_time(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        try:
            return pd.to_timedelta(s).total_seconds()
        except Exception:
            pass
        try:
            return float(s)
        except Exception:
            return None

    df["interarrival time"] = df["interarrival time"].apply(_parse_time)
    df["time difference"] = df["time difference"].apply(_parse_time)
    df["Time synchronized"] = pd.to_numeric(df["Time synchronized"], errors="coerce")
    df["Fraction of second"] = pd.to_numeric(df["Fraction of second"], errors="coerce")
    df["Actual frequency value"] = pd.to_numeric(df["Actual frequency value"], errors="coerce")

    candidate_features = [
        "Actual frequency value",
        "Fraction of second",
        "Time synchronized",
        "interarrival time",
        "time difference",
    ]

    df_clean = df.dropna(subset=candidate_features + ["attack_type"]).reset_index(drop=True)
    print(f"Loaded {len(df_clean):,} clean PMU records.")
    print("Class distribution:\n", df_clean["attack_type"].value_counts())

    # Stratified balance subset for responsive, high-quality training (up to 12,000 per class)
    subsets = []
    for cls in df_clean["attack_type"].unique():
        cls_sub = df_clean[df_clean["attack_type"] == cls]
        subsets.append(cls_sub.sample(n=min(len(cls_sub), 12000), random_state=42))
    df_train = pd.concat(subsets).sample(frac=1.0, random_state=42).reset_index(drop=True)

    encoder = LabelEncoder()
    y_raw = encoder.fit_transform(df_train["attack_type"].values)
    classes = encoder.classes_.tolist()

    X_raw = df_train[candidate_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. PSO Feature Selection
    print("\nRunning PSO Feature Selection...")
    pso = PSOFeatureSelector(num_particles=15, num_iterations=10, random_state=42)
    pso_res = pso.fit(X_scaled, y_raw, candidate_features, max_samples=6000)
    print(f"PSO Selected: {pso_res['selected_features']} (Fitness: {pso_res['best_fitness']})")

    # 2. GWO Feature Selection
    print("Running GWO Feature Selection...")
    gwo = GWOFeatureSelector(num_wolves=15, max_iterations=10, random_state=42)
    gwo_res = gwo.fit(X_scaled, y_raw, candidate_features, max_samples=6000)
    print(f"GWO Selected: {gwo_res['selected_features']} (Fitness: {gwo_res['best_fitness']})")

    # Final selected features (union of best features, preserving parsimony)
    final_features = sorted(list(set(pso_res["selected_features"]).union(gwo_res["selected_features"])))
    if len(final_features) < 2:
        final_features = candidate_features
    print(f"Final Model Features: {final_features}")

    # Scaler on final features
    X_model = df_train[final_features].values
    model_scaler = StandardScaler()
    X_model_scaled = model_scaler.fit_transform(X_model)

    # Construct temporal sequences (seq_len=5)
    X_seq, y_seq = make_sequences(X_model_scaled, y_raw, seq_len=5)
    X_tr_seq, X_te_seq, y_tr_seq, y_te_seq = train_test_split(
        X_seq, y_seq, test_size=0.25, stratify=y_seq, random_state=42
    )

    print(f"Training CNN-LSTM on {len(X_tr_seq)} sequences, testing on {len(X_te_seq)} sequences...")
    model, preds, probs, train_time, inf_time = train_cnn_lstm(
        X_tr_seq, y_tr_seq, X_te_seq, y_te_seq,
        num_features=len(final_features),
        num_classes=len(classes),
        epochs=10
    )

    # Companion Random Forest for instant SHAP TreeExplainer
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_model_scaled[:15000], y_raw[:15000])

    # Autoencoder on Normal class
    normal_idx = encoder.transform(["Normal"])[0] if "Normal" in classes else 0
    normal_X = X_model_scaled[y_raw == normal_idx]
    ae, ae_thresh = train_autoencoder(normal_X, num_features=len(final_features))

    # Metrics
    acc = round(accuracy_score(y_te_seq, preds) * 100, 2)
    prec_macro = round(precision_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    rec_macro = round(recall_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    f1_macro = round(f1_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    f1_weighted = round(f1_score(y_te_seq, preds, average="weighted", zero_division=0) * 100, 2)
    conf_mat = confusion_matrix(y_te_seq, preds).tolist()
    cls_report = classification_report(y_te_seq, preds, target_names=classes, output_dict=True, zero_division=0)

    print(f"Test Accuracy: {acc}% | Macro F1: {f1_macro}% | Training Time: {train_time}s")

    # Persist Artifacts
    torch.save({
        "state_dict": model.state_dict(),
        "num_features": len(final_features),
        "num_classes": len(classes),
        "classes": classes,
        "config": {"cnn_filters": 32, "lstm_units": 48}
    }, os.path.join(model_dir, "model.pt"))

    torch.save({
        "state_dict": ae.state_dict(),
        "threshold": ae_thresh,
        "num_features": len(final_features)
    }, os.path.join(model_dir, "autoencoder.pt"))

    joblib.dump(model_scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "companion_rf.pkl"))

    with open(os.path.join(model_dir, "pso_features.json"), "w") as f:
        json.dump(pso_res, f, indent=2)

    with open(os.path.join(model_dir, "gwo_features.json"), "w") as f:
        json.dump(gwo_res, f, indent=2)

    meta = {
        "model_id": "FDI_TSA",
        "domain": "PMU Synchrophasor (IEEE C37.118)",
        "dataset_name": "Clean_FDI_TSA_Combined.csv",
        "attack_classes": classes,
        "sequence_length": 5,
        "original_features": candidate_features,
        "pso_selected_features": pso_res["selected_features"],
        "gwo_selected_features": gwo_res["selected_features"],
        "final_features": final_features,
        "minimum_compatible_features": ["Actual frequency value", "interarrival time"],
        "evaluation": {
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "confusion_matrix": conf_mat,
            "per_class_report": cls_report,
            "training_time_seconds": train_time,
            "inference_time_ms_per_sample": inf_time,
            "autoencoder_reconstruction_threshold": round(ae_thresh, 5)
        }
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Successfully trained and saved FDI_TSA model artifacts.")
    return meta


# ===========================================================================
# 2. MSU / ORNL POWER SYSTEM PROTECTION MODEL TRAINING
# ===========================================================================
def train_msu_ornl_pipeline():
    print("\n" + "=" * 70)
    print("TRAINING MODEL 2: MSU / ORNL POWER SYSTEM PROTECTION MODEL")
    print("=" * 70)

    model_dir = os.path.join(MODELS_ROOT, "MSU_ORNL")
    os.makedirs(model_dir, exist_ok=True)

    zip_path = os.path.join(DATASETS_DIR, "archive.zip")
    print(f"Reading scenario files from {zip_path}...")

    dfs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = sorted([f for f in z.namelist() if f.endswith(".csv")])
        # Read from first 4 scenarios for representative high-fidelity dataset
        for f in csv_files[:5]:
            sub = pd.read_csv(io.BytesIO(z.read(f)))
            dfs.append(sub)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total extracted MSU/ORNL rows: {len(df_all):,}")
    print("Class distribution:\n", df_all["marker"].value_counts())

    raw_features = [c for c in df_all.columns if c != "marker"]
    
    # Handle infinite and NaN values (physical saturation/open circuit calculations)
    df_all[raw_features] = df_all[raw_features].apply(pd.to_numeric, errors="coerce")
    df_all[raw_features] = df_all[raw_features].replace([np.inf, -np.inf], np.nan)
    # Fill remaining NaNs with column medians
    df_all[raw_features] = df_all[raw_features].fillna(df_all[raw_features].median().fillna(0))

    # Remove single-value/zero variance columns
    usable_features = [c for c in raw_features if df_all[c].nunique() > 1]
    print(f"Usable features after zero-variance filter: {len(usable_features)} / {len(raw_features)}")

    # Sample balanced subset for responsive training
    attacks = df_all[df_all["marker"] == "Attack"].sample(n=min(len(df_all[df_all["marker"] == "Attack"]), 10000), random_state=42)
    naturals = df_all[df_all["marker"] == "Natural"].sample(n=min(len(df_all[df_all["marker"] == "Natural"]), 10000), random_state=42)
    df_train = pd.concat([attacks, naturals]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    encoder = LabelEncoder()
    y_raw = encoder.fit_transform(df_train["marker"].values)
    classes = encoder.classes_.tolist()

    X_raw = df_train[usable_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. PSO Feature Selection
    print("\nRunning PSO Feature Selection on 120+ relay features...")
    pso = PSOFeatureSelector(num_particles=15, num_iterations=10, random_state=42)
    pso_res = pso.fit(X_scaled, y_raw, usable_features, max_samples=4000)
    print(f"PSO selected {pso_res['selected_feature_count']} features (Fitness: {pso_res['best_fitness']})")

    # 2. GWO Feature Selection
    print("Running GWO Feature Selection on 120+ relay features...")
    gwo = GWOFeatureSelector(num_wolves=15, max_iterations=10, random_state=42)
    gwo_res = gwo.fit(X_scaled, y_raw, usable_features, max_samples=4000)
    print(f"GWO selected {gwo_res['selected_feature_count']} features (Fitness: {gwo_res['best_fitness']})")

    final_features = sorted(list(set(pso_res["selected_features"]).union(gwo_res["selected_features"])))
    if len(final_features) < 4:
        final_features = usable_features[:16]
    print(f"Final Model Features ({len(final_features)}): {final_features[:8]}...")

    X_model = df_train[final_features].values
    model_scaler = StandardScaler()
    X_model_scaled = model_scaler.fit_transform(X_model)

    X_seq, y_seq = make_sequences(X_model_scaled, y_raw, seq_len=5)
    X_tr_seq, X_te_seq, y_tr_seq, y_te_seq = train_test_split(
        X_seq, y_seq, test_size=0.25, stratify=y_seq, random_state=42
    )

    print(f"Training CNN-LSTM on {len(X_tr_seq)} sequences, testing on {len(X_te_seq)} sequences...")
    model, preds, probs, train_time, inf_time = train_cnn_lstm(
        X_tr_seq, y_tr_seq, X_te_seq, y_te_seq,
        num_features=len(final_features),
        num_classes=len(classes),
        epochs=10
    )

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_model_scaled[:15000], y_raw[:15000])

    natural_idx = encoder.transform(["Natural"])[0] if "Natural" in classes else 0
    natural_X = X_model_scaled[y_raw == natural_idx]
    ae, ae_thresh = train_autoencoder(natural_X, num_features=len(final_features))

    acc = round(accuracy_score(y_te_seq, preds) * 100, 2)
    prec_macro = round(precision_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    rec_macro = round(recall_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    f1_macro = round(f1_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    f1_weighted = round(f1_score(y_te_seq, preds, average="weighted", zero_division=0) * 100, 2)
    conf_mat = confusion_matrix(y_te_seq, preds).tolist()
    cls_report = classification_report(y_te_seq, preds, target_names=classes, output_dict=True, zero_division=0)

    print(f"MSU/ORNL Test Accuracy: {acc}% | Macro F1: {f1_macro}% | Training Time: {train_time}s")

    torch.save({
        "state_dict": model.state_dict(),
        "num_features": len(final_features),
        "num_classes": len(classes),
        "classes": classes,
        "config": {"cnn_filters": 32, "lstm_units": 48}
    }, os.path.join(model_dir, "model.pt"))

    torch.save({
        "state_dict": ae.state_dict(),
        "threshold": ae_thresh,
        "num_features": len(final_features)
    }, os.path.join(model_dir, "autoencoder.pt"))

    joblib.dump(model_scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "companion_rf.pkl"))

    with open(os.path.join(model_dir, "pso_features.json"), "w") as f:
        json.dump(pso_res, f, indent=2)

    with open(os.path.join(model_dir, "gwo_features.json"), "w") as f:
        json.dump(gwo_res, f, indent=2)

    meta = {
        "model_id": "MSU_ORNL",
        "domain": "Power Transmission Protection (MSU/ORNL)",
        "dataset_name": "binaryAllNaturalPlusNormalVsAttacks",
        "attack_classes": classes,
        "sequence_length": 5,
        "original_features": usable_features,
        "pso_selected_features": pso_res["selected_features"],
        "gwo_selected_features": gwo_res["selected_features"],
        "final_features": final_features,
        "minimum_compatible_features": final_features[:4],
        "evaluation": {
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "confusion_matrix": conf_mat,
            "per_class_report": cls_report,
            "training_time_seconds": train_time,
            "inference_time_ms_per_sample": inf_time,
            "autoencoder_reconstruction_threshold": round(ae_thresh, 5)
        }
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Successfully trained and saved MSU_ORNL model artifacts.")
    return meta


# ===========================================================================
# 3. IEC 60870-5-104 SCADA NETWORK MODEL TRAINING
# ===========================================================================
def train_iec104_pipeline():
    print("\n" + "=" * 70)
    print("TRAINING MODEL 3: IEC 60870-5-104 SCADA NETWORK ATTACK MODEL")
    print("=" * 70)

    model_dir = os.path.join(MODELS_ROOT, "IEC104")
    os.makedirs(model_dir, exist_ok=True)

    csv_path = os.path.join(DATASETS_DIR, "data_with_ioa.csv")
    print(f"Reading sample from {csv_path}...")
    
    # Read sample including all attack rows + normal sample
    # First inspect label distribution
    df_chunk = pd.read_csv(csv_path, nrows=150000, low_memory=False)
    
    candidate_features = [
        "Relative Time", "srcPort", "dstPort", "ipLen", "len",
        "fmt", "uType", "asduType", "numix", "cot", "addr", "ioa"
    ]
    
    # Fill NAs
    df_chunk[candidate_features] = df_chunk[candidate_features].fillna(0)
    
    # Map classes into descriptive names
    attack_names = {
        0: "Normal",
        1: "Command Injection",
        2: "Telemetry Spoofing",
        3: "Interrogation Attack",
        4: "Set-Point Modification",
        5: "Parameter Tampering",
        6: "IOA Scanning"
    }
    df_chunk["attack_type"] = df_chunk["label"].map(lambda x: attack_names.get(x, f"Attack_Type_{x}"))
    print("IEC-104 class distribution:\n", df_chunk["attack_type"].value_counts())

    # Balance sample: up to 3000 per class
    subsets = []
    for cls in df_chunk["attack_type"].unique():
        cls_sub = df_chunk[df_chunk["attack_type"] == cls]
        subsets.append(cls_sub.sample(n=min(len(cls_sub), 3000), random_state=42))
    df_train = pd.concat(subsets).sample(frac=1.0, random_state=42).reset_index(drop=True)

    encoder = LabelEncoder()
    y_raw = encoder.fit_transform(df_train["attack_type"].values)
    classes = encoder.classes_.tolist()

    X_raw = df_train[candidate_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. PSO Feature Selection
    print("\nRunning PSO Feature Selection on IEC-104 protocol fields...")
    pso = PSOFeatureSelector(num_particles=15, num_iterations=10, random_state=42)
    pso_res = pso.fit(X_scaled, y_raw, candidate_features, max_samples=4000)
    print(f"PSO Selected: {pso_res['selected_features']} (Fitness: {pso_res['best_fitness']})")

    # 2. GWO Feature Selection
    print("Running GWO Feature Selection on IEC-104 protocol fields...")
    gwo = GWOFeatureSelector(num_wolves=15, max_iterations=10, random_state=42)
    gwo_res = gwo.fit(X_scaled, y_raw, candidate_features, max_samples=4000)
    print(f"GWO Selected: {gwo_res['selected_features']} (Fitness: {gwo_res['best_fitness']})")

    final_features = sorted(list(set(pso_res["selected_features"]).union(gwo_res["selected_features"])))
    if len(final_features) < 3:
        final_features = candidate_features
    print(f"Final Model Features: {final_features}")

    X_model = df_train[final_features].values
    model_scaler = StandardScaler()
    X_model_scaled = model_scaler.fit_transform(X_model)

    X_seq, y_seq = make_sequences(X_model_scaled, y_raw, seq_len=5)
    X_tr_seq, X_te_seq, y_tr_seq, y_te_seq = train_test_split(
        X_seq, y_seq, test_size=0.25, stratify=y_seq, random_state=42
    )

    print(f"Training CNN-LSTM on {len(X_tr_seq)} sequences, testing on {len(X_te_seq)} sequences...")
    model, preds, probs, train_time, inf_time = train_cnn_lstm(
        X_tr_seq, y_tr_seq, X_te_seq, y_te_seq,
        num_features=len(final_features),
        num_classes=len(classes),
        epochs=10
    )

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_model_scaled[:15000], y_raw[:15000])

    normal_idx = encoder.transform(["Normal"])[0] if "Normal" in classes else 0
    normal_X = X_model_scaled[y_raw == normal_idx]
    ae, ae_thresh = train_autoencoder(normal_X, num_features=len(final_features))

    acc = round(accuracy_score(y_te_seq, preds) * 100, 2)
    prec_macro = round(precision_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    rec_macro = round(recall_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    f1_macro = round(f1_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    f1_weighted = round(f1_score(y_te_seq, preds, average="weighted", zero_division=0) * 100, 2)
    conf_mat = confusion_matrix(y_te_seq, preds).tolist()
    cls_report = classification_report(y_te_seq, preds, target_names=classes, output_dict=True, zero_division=0)

    print(f"IEC-104 Test Accuracy: {acc}% | Macro F1: {f1_macro}% | Training Time: {train_time}s")

    torch.save({
        "state_dict": model.state_dict(),
        "num_features": len(final_features),
        "num_classes": len(classes),
        "classes": classes,
        "config": {"cnn_filters": 32, "lstm_units": 48}
    }, os.path.join(model_dir, "model.pt"))

    torch.save({
        "state_dict": ae.state_dict(),
        "threshold": ae_thresh,
        "num_features": len(final_features)
    }, os.path.join(model_dir, "autoencoder.pt"))

    joblib.dump(model_scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "companion_rf.pkl"))

    with open(os.path.join(model_dir, "pso_features.json"), "w") as f:
        json.dump(pso_res, f, indent=2)

    with open(os.path.join(model_dir, "gwo_features.json"), "w") as f:
        json.dump(gwo_res, f, indent=2)

    meta = {
        "model_id": "IEC104",
        "domain": "SCADA Telecontrol Protocol (IEC 60870-5-104)",
        "dataset_name": "data_with_ioa.csv",
        "attack_classes": classes,
        "sequence_length": 5,
        "original_features": candidate_features,
        "pso_selected_features": pso_res["selected_features"],
        "gwo_selected_features": gwo_res["selected_features"],
        "final_features": final_features,
        "minimum_compatible_features": ["asduType", "cot", "ioa"],
        "evaluation": {
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "confusion_matrix": conf_mat,
            "per_class_report": cls_report,
            "training_time_seconds": train_time,
            "inference_time_ms_per_sample": inf_time,
            "autoencoder_reconstruction_threshold": round(ae_thresh, 5)
        }
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Successfully trained and saved IEC104 model artifacts.")
    return meta


# ===========================================================================
# 4. IEC 61850 SUBSTATION PROCESS BUS MODEL TRAINING
# ===========================================================================
def train_iec61850_pipeline():
    print("\n" + "=" * 70)
    print("TRAINING MODEL 4: IEC 61850 SUBSTATION PROCESS BUS MODEL")
    print("=" * 70)

    model_dir = os.path.join(MODELS_ROOT, "IEC61850")
    os.makedirs(model_dir, exist_ok=True)

    zip_path = os.path.join(DATASETS_DIR, "archive (1).zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        df_train_raw = pd.read_csv(io.BytesIO(z.read("Train.csv")))
        df_test_raw = pd.read_csv(io.BytesIO(z.read("Test.csv")))

    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
    print(f"Total IEC-61850 rows: {len(df_all):,}")
    print("Class distribution:\n", df_all["class"].value_counts())

    raw_features = [c for c in df_all.columns if c != "class"]
    usable_features = [c for c in raw_features if df_all[c].nunique() > 1]
    print(f"Usable features: {len(usable_features)} / {len(raw_features)}")

    encoder = LabelEncoder()
    y_raw = encoder.fit_transform(df_all["class"].values)
    classes = encoder.classes_.tolist()

    X_raw = df_all[usable_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. PSO Feature Selection
    print("\nRunning PSO Feature Selection on IEC-61850 GOOSE/SV features...")
    pso = PSOFeatureSelector(num_particles=15, num_iterations=10, random_state=42)
    pso_res = pso.fit(X_scaled, y_raw, usable_features, max_samples=2000)
    print(f"PSO Selected: {pso_res['selected_feature_count']} features (Fitness: {pso_res['best_fitness']})")

    # 2. GWO Feature Selection
    print("Running GWO Feature Selection on IEC-61850 GOOSE/SV features...")
    gwo = GWOFeatureSelector(num_wolves=15, max_iterations=10, random_state=42)
    gwo_res = gwo.fit(X_scaled, y_raw, usable_features, max_samples=2000)
    print(f"GWO Selected: {gwo_res['selected_feature_count']} features (Fitness: {gwo_res['best_fitness']})")

    final_features = sorted(list(set(pso_res["selected_features"]).union(gwo_res["selected_features"])))
    if len(final_features) < 4:
        final_features = usable_features[:10]
    print(f"Final Model Features ({len(final_features)}): {final_features}")

    X_model = df_all[final_features].values
    model_scaler = StandardScaler()
    X_model_scaled = model_scaler.fit_transform(X_model)

    X_seq, y_seq = make_sequences(X_model_scaled, y_raw, seq_len=5)
    X_tr_seq, X_te_seq, y_tr_seq, y_te_seq = train_test_split(
        X_seq, y_seq, test_size=0.25, stratify=y_seq, random_state=42
    )

    print(f"Training CNN-LSTM on {len(X_tr_seq)} sequences, testing on {len(X_te_seq)} sequences...")
    model, preds, probs, train_time, inf_time = train_cnn_lstm(
        X_tr_seq, y_tr_seq, X_te_seq, y_te_seq,
        num_features=len(final_features),
        num_classes=len(classes),
        epochs=12
    )

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_model_scaled, y_raw)

    normal_idx = encoder.transform(["Normal"])[0] if "Normal" in classes else 0
    normal_X = X_model_scaled[y_raw == normal_idx]
    ae, ae_thresh = train_autoencoder(normal_X, num_features=len(final_features))

    acc = round(accuracy_score(y_te_seq, preds) * 100, 2)
    prec_macro = round(precision_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    rec_macro = round(recall_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    f1_macro = round(f1_score(y_te_seq, preds, average="macro", zero_division=0) * 100, 2)
    f1_weighted = round(f1_score(y_te_seq, preds, average="weighted", zero_division=0) * 100, 2)
    conf_mat = confusion_matrix(y_te_seq, preds).tolist()
    cls_report = classification_report(y_te_seq, preds, target_names=classes, output_dict=True, zero_division=0)

    print(f"IEC-61850 Test Accuracy: {acc}% | Macro F1: {f1_macro}% | Training Time: {train_time}s")

    torch.save({
        "state_dict": model.state_dict(),
        "num_features": len(final_features),
        "num_classes": len(classes),
        "classes": classes,
        "config": {"cnn_filters": 32, "lstm_units": 48}
    }, os.path.join(model_dir, "model.pt"))

    torch.save({
        "state_dict": ae.state_dict(),
        "threshold": ae_thresh,
        "num_features": len(final_features)
    }, os.path.join(model_dir, "autoencoder.pt"))

    joblib.dump(model_scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "companion_rf.pkl"))

    with open(os.path.join(model_dir, "pso_features.json"), "w") as f:
        json.dump(pso_res, f, indent=2)

    with open(os.path.join(model_dir, "gwo_features.json"), "w") as f:
        json.dump(gwo_res, f, indent=2)

    meta = {
        "model_id": "IEC61850",
        "domain": "Substation Process Bus (IEC 61850 GOOSE/SV)",
        "dataset_name": "Train.csv & Test.csv",
        "attack_classes": classes,
        "sequence_length": 5,
        "original_features": usable_features,
        "pso_selected_features": pso_res["selected_features"],
        "gwo_selected_features": gwo_res["selected_features"],
        "final_features": final_features,
        "minimum_compatible_features": ["sqNum", "stnum", "time"],
        "evaluation": {
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "confusion_matrix": conf_mat,
            "per_class_report": cls_report,
            "training_time_seconds": train_time,
            "inference_time_ms_per_sample": inf_time,
            "autoencoder_reconstruction_threshold": round(ae_thresh, 5)
        }
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Successfully trained and saved IEC61850 model artifacts.")
    return meta


# ===========================================================================
# MASTER EXECUTION & REGISTRY CREATION
# ===========================================================================
def train_all():
    print("=" * 80)
    print("STARTING GRIDSENTRY MULTI-DATASET INDEPENDENT TRAINING")
    print("=" * 80)

    registry = {}

    # Train each model independently
    meta_pmu = train_fdi_tsa_pipeline()
    registry["FDI_TSA"] = meta_pmu

    meta_msu = train_msu_ornl_pipeline()
    registry["MSU_ORNL"] = meta_msu

    meta_iec104 = train_iec104_pipeline()
    registry["IEC104"] = meta_iec104

    meta_iec61850 = train_iec61850_pipeline()
    registry["IEC61850"] = meta_iec61850

    # Save central model registry
    registry_path = os.path.join(PROJECT_DIR, "models", "model_registry.json")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\n[DONE] Model Registry saved to: {registry_path}")

    # Also mirror into backend/ml for backend server convenience
    backend_reg_path = os.path.join(ML_DIR, "model_registry.json")
    with open(backend_reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[DONE] Model Registry mirrored to: {backend_reg_path}")


if __name__ == "__main__":
    train_all()
