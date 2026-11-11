"""
EduSense Temporal Sequence Modeling Suite
==========================================
Implements temporal sequence candidate architectures:
- Temporal LSTM
- Temporal GRU
- Temporal Convolutional Network (TCN)
- Temporal Transformer Encoder

Eliminates single-frame bias by processing sliding time-series windows (T = 30s).
Input shape: (B, T, feature_dim) where T = 30 sequence frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        # Use final time-step hidden state
        pred = self.fc(out[:, -1, :])
        return torch.sigmoid(pred) * 100.0

class TemporalGRU(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, hn = self.gru(x)
        pred = self.fc(out[:, -1, :])
        return torch.sigmoid(pred) * 100.0

class TemporalTCN(nn.Module):
    def __init__(self, input_dim=30, num_channels=[64, 64, 32], kernel_size=3):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) * dilation // 2, dilation=dilation),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x shape: (B, T, D) -> Permute to (B, D, T)
        x = x.transpose(1, 2)
        out = self.network(x)
        pooled = torch.mean(out, dim=2)
        pred = self.fc(pooled)
        return torch.sigmoid(pred) * 100.0

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim=30, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        out = self.transformer(x)
        pooled = torch.mean(out, dim=1)
        pred = self.fc(pooled)
        return torch.sigmoid(pred) * 100.0

def build_temporal_model(model_name="lstm", input_dim=30):
    name = model_name.lower()
    if "gru" in name:
        return TemporalGRU(input_dim)
    elif "tcn" in name:
        return TemporalTCN(input_dim)
    elif "transformer" in name:
        return TemporalTransformerEncoder(input_dim)
    else:
        return TemporalLSTM(input_dim)
