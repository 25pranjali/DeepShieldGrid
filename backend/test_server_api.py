import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import app

client = app.test_client()

print("=" * 70)
print("1. TEST /api/models")
print("=" * 70)
res = client.get('/api/models')
print('Status:', res.status_code)
data = res.get_json()
print('Registered models:', list(data['models'].keys()))
for m, info in data['models'].items():
    print(f"  {m}: domain=\"{info['domain']}\", acc={info['evaluation']['accuracy']}%, macro_f1={info['evaluation']['macro_f1']}%, pso_feats={len(info['pso_selected_features'])}, gwo_feats={len(info['gwo_selected_features'])}")

print("\n" + "=" * 70)
print("2. TEST /api/predict (What-If Simulator with PSO/GWO Model)")
print("=" * 70)
payload = {
    'Actual frequency value': 60.009,
    'Fraction of second': 700.0,
    'Time synchronized': 0.0,
    'interarrival time': 0.0456,
    'time difference': -4.9762
}
res_pred = client.post('/api/predict', json=payload)
print('Status:', res_pred.status_code)
pred_json = res_pred.get_json()
print('Prediction result:', pred_json)

print("\n" + "=" * 70)
print("3. TEST /api/datasets/upload (Compatible IEC-104 Test Data)")
print("=" * 70)
sample_iec = {
    'data': [
        {'Relative Time': 0.025, 'srcPort': 2404, 'dstPort': 52140, 'ipLen': 52, 'len': 14,
         'fmt': 0, 'uType': 0, 'asduType': 36.0, 'numix': 1.0, 'cot': 3.0, 'addr': 55.0, 'ioa': 136.0},
        {'Relative Time': 0.050, 'srcPort': 2404, 'dstPort': 52140, 'ipLen': 52, 'len': 14,
         'fmt': 0, 'uType': 0, 'asduType': 36.0, 'numix': 1.0, 'cot': 3.0, 'addr': 55.0, 'ioa': 136.0},
        {'Relative Time': 0.075, 'srcPort': 2404, 'dstPort': 52140, 'ipLen': 52, 'len': 14,
         'fmt': 0, 'uType': 0, 'asduType': 36.0, 'numix': 1.0, 'cot': 3.0, 'addr': 55.0, 'ioa': 136.0},
        {'Relative Time': 0.100, 'srcPort': 2404, 'dstPort': 52140, 'ipLen': 52, 'len': 14,
         'fmt': 0, 'uType': 0, 'asduType': 36.0, 'numix': 1.0, 'cot': 3.0, 'addr': 55.0, 'ioa': 136.0},
        {'Relative Time': 0.125, 'srcPort': 2404, 'dstPort': 52140, 'ipLen': 52, 'len': 14,
         'fmt': 0, 'uType': 0, 'asduType': 36.0, 'numix': 1.0, 'cot': 3.0, 'addr': 55.0, 'ioa': 136.0}
    ]
}
res_upload = client.post('/api/datasets/upload', json=sample_iec)
print('Status:', res_upload.status_code)
upload_data = res_upload.get_json()
print('Selected model:', upload_data.get('selected_model'))
print('Prediction:', upload_data.get('prediction'))
print('Confidence:', upload_data.get('confidence'))
print('Risk:', upload_data.get('risk_score'), upload_data.get('risk_level'))
print('Incident ID in DB:', upload_data.get('incident_id'))
print('SHAP values count:', len(upload_data.get('shap_values', [])))

print("\n" + "=" * 70)
print("4. TEST /api/datasets/upload (INSUFFICIENT FEATURES REJECTION)")
print("=" * 70)
bad_sample = {'data': [{'random_col_x': 1, 'random_col_y': 2}]}
res_bad = client.post('/api/datasets/upload', json=bad_sample)
print('Status:', res_bad.status_code)
print('Rejection message:', res_bad.get_json().get('message'))
