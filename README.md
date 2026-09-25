# GridSentry — Multi-Domain AI-Based Smart Grid Cybersecurity Intrusion Detection System

**GridSentry** is an industrial-grade, AI-driven cybersecurity intrusion detection and threat mitigation platform designed for modern electrical smart grids. It monitors real-time and simulated telemetry across multiple critical power grid domains, optimizes sensor and protocol feature selection via meta-heuristic swarm algorithms (**PSO** and **GWO**), performs high-speed deep sequence classification (**1D-CNN + BiLSTM**), flags unknown/zero-day anomalies via unsupervised deep autoencoders, computes genuine local feature attributions via **SHAP (TreeExplainer)**, and assesses cyber risk scores for grid security operators.

---

## 1. Core Architectural Principles

### 1.1 Strict Dataset Independence (No Cross-Domain Merging)
Modern power grids are complex, multi-tiered cyber-physical systems consisting of distinct network hierarchies and communication protocols:
- **PMU Synchrophasor Layer (IEEE C37.118)**: High-speed analog micro-second dynamics (voltage phase angles, frequency deviations, ROCOF, interarrival intervals).
- **Substation Process Bus (IEC 61850 GOOSE/Sampled Values)**: Ultra-fast peer-to-peer multicast Ethernet frames for circuit breaker tripping, interlocking, and protection relays.
- **SCADA Telecontrol Protocol (IEC 60870-5-104)**: Client/server telecontrol TCP/IP communications carrying ASDU frames, information object addresses (IOA), and supervisory setpoints.
- **Transmission Protection & Relay Analytics (MSU/ORNL)**: Multi-bus impedance calculations, relay trip logs, distance protection zones, and Snort signature alerts.

> **CRITICAL ARCHITECTURAL POLICY:**
> Under no circumstances are the training datasets concatenated or merged into a single training file. Merging PMU phasor data with IEC-104 SCADA network frames or IEC-61850 GOOSE messages would corrupt physical telemetry semantics and force artificial feature fabrication. Each domain maintains its own independent feature pipeline, optimization runs, trained deep learning models, scalers, and metadata.

### 1.2 Dynamic Feature Compatibility & Upload Engine
When a user uploads a new simulation or test dataset to the backend:
1. **Semantic Feature Alias Mapping**: Resolves vendor-specific and protocol-specific column variants to canonical smart-grid features (`backend/feature_mapping.json`).
2. **Multi-Model Compatibility Evaluation**: Assesses the uploaded dataset against all registered domain models in `models/model_registry.json`.
3. **Strict Missing Feature Rejection**: If an uploaded dataset lacks the minimum operational features of any model, it is **strictly rejected** (HTTP 422). GridSentry **NEVER fabricates or hallucinates synthetic features**.
4. **Automated Pipeline Execution**: Preprocesses the matched features, generates temporal sliding sequences, executes the hybrid 1D-CNN + BiLSTM model, performs Autoencoder zero-day anomaly checks, computes cyber risk scores, calculates real SHAP feature attributions, logs incidents to the SQLite database, and streams real-time readings to the dashboard.

---

## 2. Machine Learning & Optimization Pipeline

For each independent domain, GridSentry applies a rigorous 4-stage pipeline:

```
[Raw Domain Dataset]
         │
         ▼
[1. Preprocessing & Temporal Normalization]
         │
         ▼
[2. Swarm Optimization: PSO & GWO Feature Selection]
         ├── Particle Swarm Optimization (Macro-F1 + Parsimony penalty)
         └── Grey Wolf Optimizer (Alpha, Beta, Delta pack hierarchy)
         │
         ▼
[3. Hybrid Deep Sequence Modeling & Autoencoder Baseline]
         ├── 1D-CNN: Extracts local temporal n-gram patterns & spatial features
         ├── Bidirectional LSTM (BiLSTM): Models long-range sequential dependencies
         └── Unsupervised GridAutoencoder: Detects zero-day anomalies via reconstruction error
         │
         ▼
[4. Explainability & Risk Quantification]
         ├── Real SHAP Explanations (TreeExplainer on companion ensemble)
         └── Cyber Risk Scoring (0–100 scaled, Low / Medium / High / Critical)
```

### 2.1 Meta-Heuristic Feature Optimization
- **Particle Swarm Optimization (PSO)**: Models feature subsets as binary position vectors in $N$-dimensional space, balancing classification fitness (Macro-F1 score) against feature parsimony:
  $$\text{Fitness} = 0.95 \times \text{Macro-F1} + 0.05 \times \left(1 - \frac{|S|}{N}\right)$$
- **Grey Wolf Optimizer (GWO)**: Employs pack leadership ($\alpha, \beta, \delta$) and encirclement mechanisms to escape local minima and select the most discriminative telemetry signals.

### 2.2 Deep Learning Architecture (1D-CNN + BiLSTM)
- **1D-CNN Block**: Conv1D (kernel size 3) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ MaxPool1d / Dropout. Captures instantaneous rate-of-change and sharp transient spikes.
- **BiLSTM Block**: 2-layer Bidirectional LSTM processing temporal sequences of length $T=5$. Captures forward and reverse time-series trends (frequency drift, time sync offsets, replay attacks).
- **Dense Classifier**: Fully connected projection $\rightarrow$ BatchNorm $\rightarrow$ Dropout $\rightarrow$ Softmax output for multiclass attack classification.

### 2.3 Zero-Day / Unknown Anomaly Detection
- Each model is accompanied by an unsupervised **GridAutoencoder** trained strictly on normal operational traffic.
- Reconstruction error (Mean Squared Error across feature dimensions) is evaluated against a dynamic 99.5th percentile threshold ($\mu + 3\sigma$).
- Any test sample exhibiting an abnormally high reconstruction error is flagged as a potential **Zero-Day / Unknown Grid Anomaly**.

---

## 3. Trained Model Registry & Performance Metrics

| Model ID | Domain & Standard | Attack Classes | Selected Features (PSO / GWO) | Accuracy | Macro-F1 | Precision | Recall | AE Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`FDI_TSA`** | PMU Synchrophasor (IEEE C37.118) | Normal, FDI, TSA | 3 features (Frequency, Interarrival, Time Diff) | **97.73%** | **97.73%** | 97.84% | 97.73% | 3.1221 |
| **`MSU_ORNL`** | Transmission Protection (MSU/ORNL) | Attack, Natural | 95 features (Relay logs, impedance, phasor angles) | **61.73%** | **56.78%** | 59.64% | 57.50% | 22.4578 |
| **`IEC104`** | SCADA Telecontrol (IEC 60870-5-104) | Normal, Command Injection, Telemetry Spoofing, etc. | 12 features (ASDU type, COT, IOA, length, ports) | **75.81%** | **76.23%** | 76.54% | 75.81% | 1.8344 |
| **`IEC61850`** | Substation Process Bus (GOOSE/SV) | Normal, Masquerade, Injection, Replay, Fault | 34 features (stNum, sqNum, state_cb, 3-phase angles) | **85.16%** | **68.85%** | 73.12% | 68.85% | 1.5422 |

Artifacts for each model are organized independently in `models/<Model_ID>/`:
- `model.pt`: Trained PyTorch Hybrid 1D-CNN + BiLSTM weights and architecture configuration.
- `autoencoder.pt`: Trained PyTorch GridAutoencoder for unsupervised anomaly detection.
- `scaler.pkl`: StandardScaler fitted to training feature distributions.
- `encoder.pkl`: LabelEncoder mapping domain attack classes.
- `companion_rf.pkl`: High-fidelity companion model for real-time SHAP TreeExplainer explanations.
- `pso_features.json` & `gwo_features.json`: Convergence logs and selected feature sets.
- `metadata.json`: Full evaluation benchmarks, confusion matrices, and minimum compatible feature definitions.

---

## 4. Project Directory Structure

```
DeepShieldGrid/
├── backend/
│   ├── data/
│   │   └── Clean_FDI_TSA_Combined.csv     <- Dedicated PMU synchrophasor dataset
│   ├── ml/
│   │   ├── models_cnn_lstm.py              <- PyTorch HybridCNNLSTM & GridAutoencoder
│   │   ├── pso_feature_selection.py        <- Binary PSO feature selection engine
│   │   ├── gwo_feature_selection.py        <- Grey Wolf Optimizer feature selection engine
│   │   ├── train_all_models.py             <- Independent training pipeline script
│   │   ├── feature_compatibility.py        <- Multi-model upload, alias resolution & inference engine
│   │   ├── predict.py                      <- Real-time inference bridge & SHAP integration
│   │   └── model_registry.json             <- Central metadata registry for trained models
│   ├── uploads/                            <- Directory for user-uploaded simulation datasets
│   ├── dataset_utils.py                    <- PMU data loader and formatting utilities
│   ├── feature_mapping.json                <- Smart-grid feature aliases and taxonomy
│   ├── init_db.py                          <- Database initialization and seeding script
│   ├── server.py                           <- Flask REST API server with simulation loop
│   ├── test_simulation_scenarios.py       <- End-to-end integration test suite
│   ├── database.db                         <- SQLite database storing incidents & readings
│   └── requirements.txt                    <- Python dependencies
├── frontend/
│   ├── index.html                          <- GridSentry Web Dashboard UI (Untouched visual layout)
│   ├── css/
│   │   └── style.css                       <- Dark cyber-defense styling and responsive design
│   └── js/
│       └── app.js                          <- Dynamic UI updater, live simulation handler, upload modal
├── models/                                 <- Persisted trained model artifacts (1 folder per domain)
│   ├── FDI_TSA/
│   ├── MSU_ORNL/
│   ├── IEC104/
│   ├── IEC61850/
│   └── model_registry.json
└── README.md
```

---

## 5. Getting Started

### 5.1 Prerequisites
- Python 3.10+ (Tested on Python 3.12 and Python 3.13)
- Node.js / VS Code Live Server (or any static HTTP server for the frontend)

### 5.2 Backend Setup
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Initialize the SQLite database:
   ```bash
   python backend/init_db.py
   ```
4. Start the backend server:
   ```bash
   python backend/server.py
   ```
   The backend will start at `http://127.0.0.1:5000`.

### 5.3 Frontend Setup
Launch `frontend/index.html` via **Live Server** in VS Code (or serve using `npx serve frontend` on port 5500).

Default login credentials:
| Username | Password | Role | Access Level |
| :--- | :--- | :--- | :--- |
| `admin` | `admin123` | Administrator | Full Access + What-If Simulator + Upload Engine |
| `operator` | `operator123` | Operator | Live Dashboard + Incidents + SHAP Explanations |

---

## 6. API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/login` | Authenticate user and return session token & role |
| `GET` | `/api/models` | List all registered domain models and their capabilities |
| `GET` | `/api/models/<model_id>` | Detailed metadata, metrics, and feature lists for a specific model |
| `POST` | `/api/datasets/upload` | **Upload simulation dataset**: Analyzes features, matches model, runs inference, logs incidents |
| `GET` | `/api/readings/latest` | Retrieve latest simulated or live telemetry reading |
| `GET` | `/api/incidents` | Query incident logs with risk scores, timestamps, and attack types |
| `GET` | `/api/incidents/<id>` | Full incident details with ground-truth comparisons and mitigation steps |
| `GET` | `/api/incidents/<id>/shap` | Retrieve real SHAP feature attributions for a given incident |
| `POST` | `/api/predict` | Manual What-If scenario prediction endpoint |
| `GET` | `/api/dashboard/summary` | Real-time statistics, incident tallies, and stream progress |
| `POST` | `/api/simulation/start` | Launch background playback stream through active dataset |
| `POST` | `/api/simulation/stop` | Stop background simulation stream |
| `GET` | `/api/simulation/status` | Current streaming status and row index |

---

## 7. Cyber Risk Scoring Matrix

GridSentry quantifies physical and cyber impacts into a standardized risk index (0–100):
```
For "Normal" classification:
    Risk Score = (1.0 - Confidence) * 30.0

For Detected Attack classifications:
    Risk Score = Base Severity + (Confidence * 40.0)
    where Base Severity is:
        - Critical Attacks (Command Injection, TSA): 60.0
        - High-Impact Attacks (FDI, Masquerade, Replay): 50.0
        - Fault / Telemetry Inconsistencies: 40.0
```

| Risk Level | Score Range | Operator Action |
| :--- | :--- | :--- |
| **Low** | 0.0 – 29.9 | Routine monitoring; baseline normal operations |
| **Medium** | 30.0 – 59.9 | Elevated watch; verify synchrophasor time sync |
| **High** | 60.0 – 84.9 | Isolate suspect PMU stream; switch to state estimation backup |
| **Critical** | 85.0 – 100.0 | Immediate breaker interlock; trip protection relays; dispatch incident response |

---

## 8. Verification & Integration Testing

To run the complete automated test suite verifying all 4 domain models, alias resolution, sequence generation, autoencoder zero-day detection, cyber risk calculations, and upload rejection handling:

```bash
python backend/test_simulation_scenarios.py
```

All 5 integration scenarios execute and validate:
- Scenario 1: PMU Synchrophasor (`FDI_TSA`)
- Scenario 2: SCADA Telecontrol Protocol (`IEC104`)
- Scenario 3: Substation Process Bus GOOSE/SV (`IEC61850`)
- Scenario 4: Transmission Line Protection (`MSU_ORNL`)
- Scenario 5: Missing Feature Rejection Policy (Incompatible upload rejected with HTTP 422)
