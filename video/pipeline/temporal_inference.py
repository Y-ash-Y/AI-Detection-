"""Temporal inference for video deepfake detection."""
import torch
from typing import Tuple, Optional


class TemporalInference:
    """
    Runs temporal analysis on video frame sequences.
    """

    def __init__(self, model, device="cpu"):
        """
        Args:
            model: Temporal model instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def infer_sequence(
        self,
        frame_features: torch.Tensor,
        window_size: int = 16,
        stride: int = 8,
    ) -> Tuple[torch.Tensor, float]:
        """
        Run inference on a sequence of frame features with sliding window.

        Args:
            frame_features: (F, D) temporal feature sequence
            window_size: Number of frames per window
            stride: Stride for sliding window

        Returns:
            scores: (N,) anomaly scores for each window
            final_score: Average anomaly score across sequence
        """
        num_frames = frame_features.shape[0]
        scores = []

        with torch.no_grad():
            # Sliding window inference
            for start in range(0, num_frames - window_size + 1, stride):
                end = start + window_size
                window = frame_features[start:end].unsqueeze(0).to(self.device)
                score = self.model(window)
                scores.append(score.cpu())

        scores_tensor = torch.cat(scores, dim=0)
        final_score = float(scores_tensor.mean().item())

        return scores_tensor, final_score

    def detect_temporal_inconsistency(
        self,
        frame_features: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Detect temporal inconsistencies (deepfake indicators).

        Args:
            frame_features: (F, D) temporal features
            threshold: Decision threshold

        Returns:
            is_fake: Boolean prediction
            confidence: Confidence score
        """
        _, avg_score = self.infer_sequence(frame_features)
        is_fake = avg_score > threshold
        return is_fake, avg_score
