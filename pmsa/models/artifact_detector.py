# pmsa/models/artifact_detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from pmsa.models.features import ResidualHighPass, block_dct2, DCTBandPool


class ArtifactDetector(nn.Module):
    """
    Interpretable artifact detector.
    Produces scalar artifact scores used for explanations (not final decision).
    """

    def __init__(self, dct_keep: int = 16):
        super().__init__()
        self.hp = ResidualHighPass()
        self.dct_pool = DCTBandPool(block=8, keep=dct_keep)

    @staticmethod
    def _checkerboard_score(frame: torch.Tensor) -> float:
        """
        Measures high-frequency grid artifacts via Laplacian energy.
        """
        lap = torch.tensor(
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]],
            device=frame.device
        ).view(1, 1, 3, 3)

        gray = 0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3]
        pad = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        filt = F.conv2d(pad, lap)
        return float(filt.abs().mean().item())

    def forward(self, frames: torch.Tensor) -> Dict[str, float]:
        """
        frames: (1,3,H,W)
        returns interpretable artifact scores
        """
        with torch.no_grad():
            residual = self.hp(frames)
            H, W = residual.shape[-2:]
            residual = residual[:, :, :H - (H % 8), :W - (W % 8)]

            dct_map = block_dct2(residual, block=8)
            dct_feats = self.dct_pool(dct_map)

            blockiness = float(dct_feats.abs().mean().item())
            residual_energy = float(residual.abs().mean().item())
            checkerboard = self._checkerboard_score(frames)

        return {
            "block_dct_energy": blockiness,
            "residual_energy": residual_energy,
            "checkerboard_score": checkerboard
        }
