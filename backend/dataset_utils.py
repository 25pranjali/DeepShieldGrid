"""
dataset_utils.py

Small helper module shared by init_db.py and server.py.

The raw CSV has two columns ("interarrival time" and "time difference")
stored as duration strings like "00:00:00.018209" instead of plain
numbers. The trained model expects these as floats (seconds), so we
convert them here in ONE place so both scripts stay consistent.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "Clean_FDI_TSA_Combined.csv")

# These are the five columns the ML model was trained on, in this exact order.
MODEL_FEATURES = [
    "Actual frequency value",
    "Fraction of second",
    "Time synchronized",
    "interarrival time",
    "time difference",
]


def _duration_to_seconds(value):
    """Convert a duration value into a float number of seconds.

    This column comes in TWO different formats in the raw dataset:
      - Normal / FDI rows: a duration string like '00:00:00.018209'
      - TSA rows: already a plain (sometimes negative) number of
        seconds, e.g. '-4.992581583333333' -- this happens because a
        Time Synchronization Attack can push the time difference far
        outside a normal HH:MM:SS range, so it wasn't saved in that
        format for those rows.

    We try the duration-string format first, and fall back to parsing
    it as a plain float if that fails. Returns None if neither works
    or the value is missing.
    """
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


def load_clean_dataset():
    """
    Load the PMU dataset from disk and return a cleaned pandas DataFrame
    that is ready to feed into the ML model.

    Steps performed:
    1. Read the raw CSV.
    2. Convert 'interarrival time' and 'time difference' from duration
       strings into plain float seconds.
    3. Drop rows that are missing any of the five required model features
       (the model cannot make a prediction without all five).
    4. Reset the row index so we can use it as a simple sequential
       "row_index" when simulating a real-time stream.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Make sure Clean_FDI_TSA_Combined.csv is inside backend/data/."
        )

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    # Convert the two duration-string columns into numeric seconds.
    df["interarrival time"] = df["interarrival time"].apply(_duration_to_seconds)
    df["time difference"] = df["time difference"].apply(_duration_to_seconds)

    # The model cannot handle missing values, so we drop rows that don't
    # have all five required features. This keeps the simulation clean.
    df = df.dropna(subset=MODEL_FEATURES).reset_index(drop=True)

    # A simple sequential id we will step through during simulation.
    df["row_index"] = df.index

    return df
