"""Google's frozen SigLIP vision backbone, credited rather than trained here.

The custom detector head learns from these embeddings. Whether this backbone is
better than CLIP for a particular domain must be measured, not assumed from scale.
"""
from __future__ import annotations

import numpy as np

from .base import Backbone, register
from .hub_revision import resolve_hub_revision


@register("siglip")
class SiglipEncoder(Backbone):
    dim = 768  # google/siglip-base-patch16-224; overwritten from config at load

    def __init__(self, device: str = "cpu",
                 weights: str = "google/siglip-base-patch16-224",
                 image_size: int = 224, revision=None, use_fast=False, **kw):
        super().__init__(device=device)
        import torch
        from transformers import SiglipVisionModel, AutoImageProcessor

        self.torch = torch
        revision = resolve_hub_revision(weights, revision)
        self.model = SiglipVisionModel.from_pretrained(weights, revision=revision).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(weights, revision=revision, use_fast=use_fast)
        self.load_config = {"weights": weights, "revision": revision, "use_fast": use_fast}
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
