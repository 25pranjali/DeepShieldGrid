"""
dataset_utils.py

Helper module shared by init_db.py and server.py.
Loads and preprocesses the clean PMU dataset (Normal, FDI, TSA) for simulated real-time streaming.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "Clean_FDI_TSA_Combined.csv")

MODEL_FEATURES = [
    "Actual frequency value",
    "Fraction of second",
    "Time synchronized",
    "interarrival time",
    "time difference",
]


def _duration_to_seconds(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        return pd.to_timedelta(text).total_seconds()
    except (ValueError, TypeError):
        pass
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def load_clean_dataset(interleave=True, normal_chunk=2, attack_chunk=1):
    """
    Load the clean PMU dataset from disk and return a cleaned pandas DataFrame
    ready for simulated live feed. Interleaves Normal, FDI, and TSA rows into
    realistic continuous cycles.
    """
    if not os.path.exists(DATASET_PATH):
        # Fallback to datasets/ directory if needed
        alt_path = os.path.join(os.path.dirname(BASE_DIR), "datasets", "Clean_FDI_TSA_Combined.csv")
        if os.path.exists(alt_path):
            df = pd.read_csv(alt_path, low_memory=False)
        else:
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH} or {alt_path}.")
    else:
        df = pd.read_csv(DATASET_PATH, low_memory=False)

    df["interarrival time"] = df["interarrival time"].apply(_duration_to_seconds)
    df["time difference"] = df["time difference"].apply(_duration_to_seconds)
    df["Time synchronized"] = pd.to_numeric(df["Time synchronized"], errors="coerce")
    df["Fraction of second"] = pd.to_numeric(df["Fraction of second"], errors="coerce")
    df["Actual frequency value"] = pd.to_numeric(df["Actual frequency value"], errors="coerce")

    # Drop missing values
    df = df.dropna(subset=MODEL_FEATURES).reset_index(drop=True)

    if interleave:
        df_norm = df[df["attack_type"] == "Normal"]
        df_fdi = df[df["attack_type"] == "FDI"]
        df_tsa = df[df["attack_type"] == "TSA"]

        # Cycle: normal_chunk -> FDI -> normal_chunk -> TSA
        cycle_len = 2 * normal_chunk + 2 * attack_chunk
        num_cycles = min(
            len(df_norm) // (2 * normal_chunk),
            len(df_fdi) // attack_chunk,
            len(df_tsa) // attack_chunk,
        )

        used_norm = df_norm.iloc[: num_cycles * 2 * normal_chunk].copy()
        used_fdi = df_fdi.iloc[: num_cycles * attack_chunk].copy()
        used_tsa = df_tsa.iloc[: num_cycles * attack_chunk].copy()

        norm_offsets = np.concatenate([
            np.arange(0, normal_chunk),
            np.arange(normal_chunk + attack_chunk, 2 * normal_chunk + attack_chunk),
        ])
        norm_positions = np.repeat(np.arange(num_cycles) * cycle_len, 2 * normal_chunk) + np.tile(
            norm_offsets, num_cycles
        )

        fdi_offsets = np.arange(normal_chunk, normal_chunk + attack_chunk)
        fdi_positions = np.repeat(np.arange(num_cycles) * cycle_len, attack_chunk) + np.tile(
            fdi_offsets, num_cycles
        )

        tsa_offsets = np.arange(2 * normal_chunk + attack_chunk, cycle_len)
        tsa_positions = np.repeat(np.arange(num_cycles) * cycle_len, attack_chunk) + np.tile(
            tsa_offsets, num_cycles
        )

        used_norm = used_norm.copy()
        used_fdi = used_fdi.copy()
        used_tsa = used_tsa.copy()

        used_norm["sim_order"] = norm_positions
        used_fdi["sim_order"] = fdi_positions
        used_tsa["sim_order"] = tsa_positions

        interleaved = (
            pd.concat([used_norm, used_fdi, used_tsa], ignore_index=True)
            .sort_values("sim_order")
            .drop(columns=["sim_order"])
            .reset_index(drop=True)
        )
        interleaved["row_index"] = interleaved.index
        return interleaved

    df["row_index"] = df.index
    return df
