"""
test_simulation_scenarios.py

End-to-end verification of GridSentry with newly uploaded simulation datasets:
1. PMU Simulation Dataset -> routes to FDI_TSA model
2. IEC-104 SCADA Simulation Dataset -> routes to IEC104 model
3. Substation GOOSE/SV Simulation Dataset -> routes to IEC61850 model
4. Transmission Relay Simulation Dataset -> routes to MSU_ORNL model
5. Incompatible Dataset -> rejected with informative feature mismatch details
"""

import os
import sys
import io
import json
import zipfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import app

client = app.test_client()

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
SCRATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "test_samples")
os.makedirs(SCRATCH_DIR, exist_ok=True)

print("=" * 80)
print("GRIDSENTRY END-TO-END SIMULATION DATASET UPLOAD TEST SUITE")
print("=" * 80)

# ---------------------------------------------------------------------------
# Test 1: PMU Simulation Dataset
# ---------------------------------------------------------------------------
print("\n[TEST 1] Testing PMU Simulation Dataset Upload...")
df_pmu = pd.read_csv(os.path.join(DATASETS_DIR, "Clean_FDI_TSA_Combined.csv"), low_memory=False)
pmu_cols = ["Actual frequency value", "Fraction of second", "Time synchronized", "interarrival time", "time difference"]
pmu_sample = df_pmu[pmu_cols].dropna().head(20).copy()
pmu_csv_path = os.path.join(SCRATCH_DIR, "sim_pmu_test.csv")
pmu_sample.to_csv(pmu_csv_path, index=False)

with open(pmu_csv_path, "rb") as f:
    res1 = client.post("/api/datasets/upload", data={"file": (f, "sim_pmu_test.csv")})

print(f"Status: {res1.status_code}")
d1 = res1.get_json()
if res1.status_code != 200:
    print(f"  Error message: {d1}")
print(f"  Selected Model: {d1.get('selected_model')}")
print(f"  Domain: {d1.get('domain')}")
print(f"  Prediction: {d1.get('prediction')} | Confidence: {d1.get('confidence')}% | Risk: {d1.get('risk_level')} ({d1.get('risk_score')})")
print(f"  Incident ID Created: {d1.get('incident_id')}")
print(f"  SHAP Features: {[s['feature'] for s in d1.get('shap_values', [])]}")
assert d1.get("selected_model") == "FDI_TSA", f"Failed to select FDI_TSA model! Response: {d1}"

# ---------------------------------------------------------------------------
# Test 2: IEC-104 SCADA Simulation Dataset
# ---------------------------------------------------------------------------
print("\n[TEST 2] Testing IEC-104 SCADA Simulation Dataset Upload...")
df_iec104 = pd.read_csv(os.path.join(DATASETS_DIR, "data_with_ioa.csv"), nrows=20, low_memory=False)
iec_cols = ["Relative Time", "srcPort", "dstPort", "ipLen", "len", "fmt", "uType", "asduType", "numix", "cot", "addr", "ioa"]
iec_sample = df_iec104[iec_cols].copy()
iec_csv_path = os.path.join(SCRATCH_DIR, "sim_iec104_test.csv")
iec_sample.to_csv(iec_csv_path, index=False)

with open(iec_csv_path, "rb") as f:
    res2 = client.post("/api/datasets/upload", data={"file": (f, "sim_iec104_test.csv")})

print(f"Status: {res2.status_code}")
d2 = res2.get_json()
print(f"  Selected Model: {d2.get('selected_model')}")
print(f"  Domain: {d2.get('domain')}")
print(f"  Prediction: {d2.get('prediction')} | Confidence: {d2.get('confidence')}% | Risk: {d2.get('risk_level')} ({d2.get('risk_score')})")
print(f"  Zero-Day/Unknown: {d2.get('is_unknown_attack')}")
assert d2.get("selected_model") == "IEC104", "Failed to select IEC104 model!"

# ---------------------------------------------------------------------------
# Test 3: IEC 61850 Substation Simulation Dataset
# ---------------------------------------------------------------------------
print("\n[TEST 3] Testing IEC 61850 Substation Simulation Dataset Upload...")
with zipfile.ZipFile(os.path.join(DATASETS_DIR, "archive (1).zip"), "r") as z:
    df_61850 = pd.read_csv(io.BytesIO(z.read("Test.csv")), nrows=20)
sub_cols = [c for c in df_61850.columns if c != "class"]
sub_sample = df_61850[sub_cols].copy()
sub_csv_path = os.path.join(SCRATCH_DIR, "sim_iec61850_test.csv")
sub_sample.to_csv(sub_csv_path, index=False)

with open(sub_csv_path, "rb") as f:
    res3 = client.post("/api/datasets/upload", data={"file": (f, "sim_iec61850_test.csv")})

print(f"Status: {res3.status_code}")
d3 = res3.get_json()
if res3.status_code != 200:
    print(f"  Error message: {d3}")
print(f"  Selected Model: {d3.get('selected_model')}")
print(f"  Domain: {d3.get('domain')}")
print(f"  Prediction: {d3.get('prediction')} | Confidence: {d3.get('confidence')}% | Risk: {d3.get('risk_level')} ({d3.get('risk_score')})")
assert d3.get("selected_model") == "IEC61850", f"Failed to select IEC61850 model! Response: {d3}"

# ---------------------------------------------------------------------------
# Test 4: Incompatible Dataset Rejection
# ---------------------------------------------------------------------------
print("\n[TEST 4] Testing Incompatible Dataset Rejection Policy...")
df_bad = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "account_balance": [5000.0, 12500.5, 300.0],
    "transaction_type": ["deposit", "withdrawal", "transfer"]
})
bad_csv_path = os.path.join(SCRATCH_DIR, "banking_data.csv")
df_bad.to_csv(bad_csv_path, index=False)

with open(bad_csv_path, "rb") as f:
    res4 = client.post("/api/datasets/upload", data={"file": (f, "banking_data.csv")})

print(f"Status: {res4.status_code}")
d4 = res4.get_json()
print(f"  Status: {d4.get('status')}")
print(f"  Rejection Message: {d4.get('message')}")
assert res4.status_code == 422, "Failed to reject incompatible dataset!"

# ---------------------------------------------------------------------------
# Test 5: MSU/ORNL Transmission Relay Simulation Dataset
# ---------------------------------------------------------------------------
print("\n[TEST 5] Testing MSU/ORNL Transmission Relay Simulation Dataset Upload...")
with zipfile.ZipFile(os.path.join(DATASETS_DIR, "archive.zip"), "r") as z:
    df_msu = pd.read_csv(io.BytesIO(z.read("binaryAllNaturalPlusNormalVsAttacks/data1.csv")), nrows=20)
msu_cols = [c for c in df_msu.columns if c != "marker"]
msu_sample = df_msu[msu_cols].copy()
msu_csv_path = os.path.join(SCRATCH_DIR, "sim_msu_test.csv")
msu_sample.to_csv(msu_csv_path, index=False)

with open(msu_csv_path, "rb") as f:
    res5 = client.post("/api/datasets/upload", data={"file": (f, "sim_msu_test.csv")})

print(f"Status: {res5.status_code}")
d5 = res5.get_json()
print(f"  Selected Model: {d5.get('selected_model')}")
print(f"  Domain: {d5.get('domain')}")
print(f"  Prediction: {d5.get('prediction')} | Confidence: {d5.get('confidence')}% | Risk: {d5.get('risk_level')} ({d5.get('risk_score')})")
assert d5.get("selected_model") == "MSU_ORNL", "Failed to select MSU_ORNL model!"

print("\n" + "=" * 80)
print("ALL 5 END-TO-END TESTS PASSED WITH 100% SUCCESS!")
print("=" * 80)
