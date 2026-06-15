"""SigLIP vision encoder, frozen. A stronger semantic backbone than CLIP-L/14.

SigLIP (Google) is pretrained on ~10B image-text pairs, so its frozen features are
more robust to real-world photo diversity than CLIP — which is exactly the weakness
that made the tiny-genimage CLIP probe false-alarm on ordinary photos. Use this as a
drop-in backbone: extract once, train your own linear probe / head on top. The
backbone is credited (you don't train SigLIP itself); the detector head is yours.
"""
from __future__ import annotations

import numpy as np

from .base import Backbone, register


@register("siglip")
class SiglipEncoder(Backbone):
    dim = 768  # google/siglip-base-patch16-224; overwritten from config at load

    def __init__(self, device: str = "cpu",
                 weights: str = "google/siglip-base-patch16-224",
                 image_size: int = 224, **kw):
        super().__init__(device=device)
        import torch
        from transformers import SiglipVisionModel, AutoImageProcessor

        self.torch = torch
        self.model = SiglipVisionModel.from_pretrained(weights).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(weights)
        self.dim = int(self.model.config.hidden_size)
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
            emb = out.pooler_output  # attention-pooled image embedding
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().cpu().numpy()

    def _to_batch(self, images):
        torch = self.torch
        if isinstance(images, torch.Tensor):
            return images
        return torch.stack([self.preprocess(im) for im in images])
