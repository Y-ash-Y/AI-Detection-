"""Lip-sync model for detecting audio-visual mismatches."""
import torch
import torch.nn as nn


class LipSyncModel(nn.Module):
    """
    Detects lip-sync inconsistencies between audio and video.
    Analyzes mouth region motion against audio spectrogram.
    """

    def __init__(self, audio_dim=256, video_dim=256, hidden_dim=512):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim

        # Audio encoder (processes MFCC or spectrogram)
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Video encoder (processes mouth region features)
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Synchronization classifier
        self.sync_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (B, T, audio_dim) audio spectrogram features
            video_features: (B, T, video_dim) mouth region features

        Returns:
            (B,) sync mismatch scores (higher = more mismatch)
        """
        # Encode each modality
        audio_encoded = self.audio_encoder(audio_features)  # (B, T, hidden_dim//2)
        video_encoded = self.video_encoder(video_features)  # (B, T, hidden_dim//2)

        # Concatenate and compute temporal mean
        fused = torch.cat([audio_encoded, video_encoded], dim=-1)  # (B, T, hidden_dim)
        fused_mean = fused.mean(dim=1)  # (B, hidden_dim)

        # Classify sync consistency
        sync_score = self.sync_classifier(fused_mean)
        return sync_score.squeeze(-1)


class MouthRegionDetector(nn.Module):
    """
    Detects and extracts mouth region from face.
    """

    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim

        # Feature extractor for mouth region
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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

    def forward(self, mouth_region: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mouth_region: (B, 3, H, W) cropped mouth region images

        Returns:
            (B, D) mouth region features
        """
        features = self.extractor(mouth_region)
        features = features.view(features.size(0), -1)
        return self.fc(features)
