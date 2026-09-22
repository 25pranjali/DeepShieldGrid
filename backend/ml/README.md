# GridSentry ML Model

## Model
Random Forest Classifier

## Model Parameters
- n_estimators: 200
- random_state: 42
- n_jobs: -1

## Classes
- FDI
- Normal
- TSA

## Input Features
1. Actual frequency value
2. Fraction of second
3. Time synchronized
4. interarrival time
5. time difference

## Preprocessing
No input feature scaler was used.
The target labels were encoded using LabelEncoder.

## Prediction
The model can be used to predict:
- FDI
- Normal
- TSA

The Random Forest probability output can be used as the prediction confidence.
