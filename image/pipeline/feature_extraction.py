import numpy as np
import torch
from typing import Tuple

from image.models.clip_encoder import CLIPEncoder
from image.models.dino_encoder import DINOEncoder
from image.models.forensic_features import extract_forensic_features
from image.models.device import get_device


class FeaturePipeline:
    """
    Unified feature extraction pipeline.

    Combines three complementary feature streams:
      - CLIP  : semantic invariant features (architecture-agnostic)
      - DINO  : structural/geometric features
      - Forensic: DCT spectral + noise residual (signal-processing cues)

    Design rationale (from PMSA theory):
      We want x → z such that z approximates sufficient statistics
      for the LRT. Each encoder captures a different aspect of the
      real/fake manifold split, making fusion more powerful than
      any single stream.
    """

    def __init__(self, device=None):
        self.device = device or get_device()
        print(f"[FeaturePipeline] Using device: {self.device}")

        print("Loading CLIP encoder...")
        self.clip = CLIPEncoder(device=self.device)

        print("Loading DINOv2 encoder...")
        self.dino = DINOEncoder(device=self.device)

        print("Feature pipeline ready.")
        print(f"  CLIP dim     : {self.clip.dim}")
        print(f"  DINO dim     : {self.dino.dim}")
        print(f"  Forensic dim : 1024")

    def extract(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all features for a single image.

        Returns:
            clip_feat     : (512,)
            dino_feat     : (384,)
            forensic_feat : (1024,)
        """
        clip_feat     = self.clip.extract(image_path).numpy()
        dino_feat     = self.dino.extract(image_path).numpy()
        forensic_feat = extract_forensic_features(image_path)

        return clip_feat, dino_feat, forensic_feat
