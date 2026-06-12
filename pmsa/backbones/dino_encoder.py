"""DINOv2 ViT-B/14, frozen. Structural / geometric stream.

Upgraded from v1's ViT-S/14 to ViT-B/14 for a stronger structural representation
at modest extra extraction cost.
"""
from __future__ import annotations

import numpy as np

from .base import Backbone, register


@register("dino_b14")
class DINOEncoder(Backbone):
    dim = 768  # ViT-B/14 CLS embedding

    def __init__(self, device: str = "cpu",
                 weights: str = "facebook/dinov2-base",
                 image_size: int = 224, **kw):
        super().__init__(device=device)
        import torch
        from transformers import AutoModel, AutoImageProcessor

        self.torch = torch
        self.model = AutoModel.from_pretrained(weights).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(weights)
        self._image_size = image_size

    @property
    def preprocess(self):
        def _t(pil_image):
            out = self.processor(images=pil_image, return_tensors="pt")
            return out["pixel_values"][0]
        return _t

    def encode(self, images) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = self._to_batch(images).to(self.device)
            out = self.model(pixel_values=x)
            cls = out.last_hidden_state[:, 0]  # CLS token
            cls = torch.nn.functional.normalize(cls, dim=-1)
        return cls.float().cpu().numpy()

    def _to_batch(self, images):
        torch = self.torch
        if isinstance(images, torch.Tensor):
            return images
        return torch.stack([self.preprocess(im) for im in images])
