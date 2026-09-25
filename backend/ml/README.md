# GridSentry Machine Learning & Deep Learning System

## Models Implemented

### 1. Hybrid 1D-CNN + BiLSTM Neural Network (Deep Learning)
- **Model File**: `smart_grid_cnn_lstm.pt`
- **Architecture**:
  - 1D Convolutional Layer (`Conv1D`) for extracting spatial correlations across PMU feature channels
  - Batch Normalization (`BatchNorm1d`) + ReLU Activation + Dropout
  - Bidirectional LSTM (`BiLSTM`, 2 layers) for learning sequential temporal dynamics
  - Dense Fully-Connected Classification Head (`Linear(32) -> Linear(4)`)

### 2. Particle Swarm Optimization (PSO) & GWO (Meta-Heuristic Optimization)
- **Script**: `train_cnn_lstm_pso.py`
- **PSO Role**: Hyperparameter optimization searching for optimal CNN filter sizes, BiLSTM hidden dimensions, and AdamW learning rates.
- **Optimization Results**:
  - CNN Filters: 21
  - LSTM Units: 78
  - Learning Rate: 0.00123
  - Validation Accuracy: **99.26%** (Macro F1: 0.99)

### 3. Companion Calibrated Ensemble & SHAP Engine
- **Model File**: `smart-grid_attack_model.pkl`
- **Role**: Instant single-snapshot inference (What-If Simulator) and full TreeExplainer SHAP explainability.

## Detected Classes
1. **Normal**: Standard grid operations & natural transients
2. **FDI**: False Data Injection attack
3. **TSA**: Time Synchronization Attack (timestamp skew & phase drift)
4. **Command Injection**: MSU/ORNL Power System relay trip & unauthorized control attack

## Input Features
1. `Actual frequency value` (Hz)
2. `Fraction of second` (sub-second timestamp / frame counter)
3. `Time synchronized` (binary GPS/PTP synchronization status)
4. `interarrival time` (seconds)
5. `time difference` (PDC - PMU latency difference)
