"""Google's frozen SigLIP vision backbone, credited rather than trained here.

The custom detector head learns from these embeddings. Generalization to a
particular domain must be measured, not assumed from pretraining scale.
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
        self.processor = AutoImageProcessor.from_pretrained(weights, revision=revision, use_fast=use_fast)
        size = self.processor.size
        if (type(image_size) is not int or image_size <= 0
                or size.get("height") != image_size or size.get("width") != image_size):
            raise ValueError(f"SigLIP image_size={image_size} does not match the pinned processor size {size}")
        self.model = SiglipVisionModel.from_pretrained(weights, revision=revision).to(device).eval()
        if self.model.config.image_size != image_size:
            raise ValueError("SigLIP model and processor image sizes differ")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.load_config = {"weights": weights, "revision": revision, "use_fast": use_fast}
        self.dim = int(self.model.config.hidden_size)
        self._image_size = image_size

    @property
    def preprocess(self):
        def _t(pil_image):
            out = self.processor(images=pil_image, return_tensors="pt")
            pixels = out["pixel_values"][0]
            if tuple(pixels.shape[-2:]) != (self._image_size, self._image_size):
                raise ValueError("SigLIP processor did not produce the declared image size")
            return pixels
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
