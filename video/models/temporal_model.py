"""Temporal model for video deepfake detection."""
import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """
    Temporal consistency model for detecting frame-level anomalies.
    Analyzes optical flow and temporal patterns across video frames.
    """

    def __init__(self, input_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) temporal feature sequence
               B = batch size, T = time steps, D = feature dimension

        Returns:
            (B, T) or (B,) temporal anomaly scores
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]
        scores = self.classifier(last_hidden)
        return scores.squeeze(-1)


class OpticalFlowEncoder(nn.Module):
    """
    Encodes optical flow for temporal analysis.
    """

    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(128, out_dim)

    def forward(self, optical_flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            optical_flow: (B, 2, H, W) optical flow in x, y directions

        Returns:
            (B, D) encoded features
        """
        features = self.encoder(optical_flow)
        features = features.view(features.size(0), -1)
        return self.fc(features)
