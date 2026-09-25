"""
models_cnn_lstm.py

Deep Learning Neural Architectures for GridSentry:
1. HybridCNNLSTM: 1D-CNN + Bidirectional LSTM classifier for temporal grid attacks.
2. GridAutoencoder: Deep Autoencoder for unsupervised unknown anomaly detection.
"""

import torch
import torch.nn as nn
import numpy as np


class HybridCNNLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        cnn_filters: int = 32,
        kernel_size: int = 3,
        lstm_units: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # 1D-CNN block to capture local physical/protocol correlations
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)

        # Optional second conv layer for richer spatial feature representation
        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters)

        # Bidirectional LSTM to model temporal sequence transitions
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Classification Head
        self.dropout_dense = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units * 2, 48)
        self.fc_out = nn.Linear(48, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        # Conv1d expects (batch, in_channels, seq_len)
        x_trans = x.transpose(1, 2)
        c1 = self.dropout_cnn(self.relu(self.bn1(self.conv1(x_trans))))
        c2 = self.dropout_cnn(self.relu(self.bn2(self.conv2(c1))))

        # Back to (batch, seq_len, cnn_filters)
        c_seq = c2.transpose(1, 2)
        lstm_out, _ = self.lstm(c_seq)

        # Take last time-step representation
        last_step = lstm_out[:, -1, :]
        h = self.relu(self.fc1(self.dropout_dense(last_step)))
        logits = self.fc_out(h)
        return logits


class GridAutoencoder(nn.Module):
    """
    Autoencoder trained exclusively on Normal grid operation data.
    Detects zero-day / unknown cyber attacks via high reconstruction error.
    """
    def __init__(self, num_features: int, latent_dim: int = None):
        super().__init__()
        self.num_features = num_features
        if latent_dim is None:
            latent_dim = min(8, max(2, num_features // 2))
        self.latent_dim = latent_dim
        hidden_dim = max(latent_dim * 2, num_features // 2, 16)

        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, num_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

    def get_reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            x_rec = self.forward(x)
            mse = torch.mean((x - x_rec) ** 2, dim=-1)
        return mse.cpu().numpy()
