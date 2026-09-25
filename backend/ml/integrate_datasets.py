"""
integrate_datasets.py

Integrates the MSU/ORNL Power System Attack Dataset (from Kaggle)
with the existing FDI and TSA dataset, creating a unified smart-grid
dataset with 4 comprehensive classes:
  1. Normal (Baseline grid operation + natural transients)
  2. FDI (False Data Injection)
  3. TSA (Time Synchronization Attack)
  4. Command Injection (MSU/ORNL Relay Trip & Command Attacks)
"""

import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ORIGINAL_CSV = os.path.join(DATA_DIR, "Clean_FDI_TSA_Combined.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Clean_FDI_TSA_Combined.csv")  # In-place upgrade

MSU_DIR = r"C:\Users\HP\.cache\kagglehub\datasets\bachirbarika\power-system\versions\1\binaryAllNaturalPlusNormalVsAttacks"


def load_and_integrate():
    print(f"Loading original FDI/TSA dataset from: {ORIGINAL_CSV}")
    df_orig = pd.read_csv(ORIGINAL_CSV, low_memory=False)
    print("Original attack distribution:")
    print(df_orig["attack_type"].value_counts())

    # Find MSU/ORNL CSV files
    msu_files = sorted(glob.glob(os.path.join(MSU_DIR, "*.csv")))
    if not msu_files:
        raise FileNotFoundError(f"No MSU/ORNL CSV files found in {MSU_DIR}")

    print(f"\nProcessing {len(msu_files)} MSU/ORNL datasets...")
    msu_attack_rows = []
    msu_natural_rows = []

    # Read from the first 5 CSVs for a balanced high-quality subset (~15k attack rows)
    for f in msu_files[:6]:
        df_msu = pd.read_csv(f)
        attacks = df_msu[df_msu["marker"] == "Attack"].copy()
        naturals = df_msu[df_msu["marker"] == "Natural"].copy()

        msu_attack_rows.append(attacks)
        msu_natural_rows.append(naturals)

    df_msu_attacks = pd.concat(msu_attack_rows, ignore_index=True)
    df_msu_naturals = pd.concat(msu_natural_rows, ignore_index=True)
    print(f"Extracted {len(df_msu_attacks)} MSU attack rows and {len(df_msu_naturals)} MSU natural rows.")

    # Subsample to keep the classes well-balanced with FDI (~12k) and TSA (~6k)
    sample_size = min(12000, len(df_msu_attacks))
    df_msu_attacks = df_msu_attacks.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Convert MSU/ORNL columns to the standard PMU features:
    # 1. Actual frequency value: R1:F
    # 2. Fraction of second: 30 fps periodic stream
    # 3. Time synchronized: 1.0
    # 4. interarrival time: 0.0333s (30 Hz PMU reporting rate)
    # 5. time difference: ~0.045s transmission delay
    n = len(df_msu_attacks)
    step_ms = 33.333
    fraction_seq = np.array([(i * step_ms) % 1000.0 for i in range(n)])

    msu_formatted = pd.DataFrame({
        "attack_type": "Command Injection",
        "Actual frequency value": df_msu_attacks["R1:F"].values,
        "Message Time Quality indicator code": "0x00",
        "Fraction of second": fraction_seq,
        "PMU Time Quality": "0x0000",
        "Time synchronized": 1.0,
        "PMU time": "12:00",
        "PDC time": "12:00.045",
        "interarrival time": "00:00:00.033333",
        "time difference": "00:00:00.045000",
        "attack_active": 1,
        "manipulated_fields": "Relay Trip / Breaker Command"
    })

    # Combine with original dataset
    combined_df = pd.concat([df_orig, msu_formatted], ignore_index=True)

    print("\nCombined Dataset Attack Distribution:")
    print(combined_df["attack_type"].value_counts())

    # Save to disk
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccessfully wrote unified dataset ({len(combined_df)} rows) to: {OUTPUT_CSV}")


if __name__ == "__main__":
    load_and_integrate()
