import joblib
import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "smart-grid_attack_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.json")

# Load model, encoder and feature list
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)


def predict_attack(input_data):
    """
    Predict the smart-grid attack type.

    input_data can be a dictionary containing the five
    required features.
    """

    # Create DataFrame using the exact feature order
    input_df = pd.DataFrame(
        [[input_data[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    # Make prediction
    prediction_encoded = model.predict(input_df)[0]

    # Convert numerical prediction to attack name
    prediction_label = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    # Get class probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Highest probability is used as confidence
    confidence = float(max(probabilities))

    return {
        "prediction": prediction_label,
        "confidence": round(confidence * 100, 2)
    }


if __name__ == "__main__":

    # Example input
    sample_input = {
        "Actual frequency value": 50.0,
        "Fraction of second": 0.5,
        "Time synchronized": 1,
        "interarrival time": 0.1,
        "time difference": 0.01
    }

    result = predict_attack(sample_input)

    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"], "%")
