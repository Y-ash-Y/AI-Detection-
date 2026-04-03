"""Wrappers that expose a unified feature extractor interface."""
import torch
from pmsa.image.models.clip_model import CLIPFeatureExtractor
from pmsa.image.models.dino_model import DINOFeatureExtractor
from pmsa.image.models.forensic_features import ResidualHighPass, block_dct2, DCTBandPool


class FeatureExtractor:
    def __init__(self, device="cpu"):
        self.clip = CLIPFeatureExtractor(device=device)
        self.dino = DINOFeatureExtractor(device=device)
        self.hp = ResidualHighPass()
        self.pool = DCTBandPool(block=8, keep=16)

    @torch.no_grad()
    def extract_all(self, img):
        clip_feat = self.clip(img)
        dino_feat = self.dino(img)

        r = self.hp(img)
        H, W = r.shape[-2:]
        r = r[:, :, : H - (H % 8), : W - (W % 8)]
        dct_map = block_dct2(r, block=8)
        forensic_feat = self.pool(dct_map)

        return clip_feat, dino_feat, forensic_feat
