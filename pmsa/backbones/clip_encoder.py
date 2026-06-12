"""CLIP ViT-L/14, frozen. The main semantic stream and the UnivFD baseline backbone.

Deliberately L/14, not B/32: the generalization gap to in-the-wild generators is
much larger for the bigger model. This is the single most important backbone
choice in v2.
"""
from __future__ import annotations

import numpy as np

from .base import Backbone, register


@register("clip_l14")
class CLIPEncoder(Backbone):
    dim = 768  # ViT-L/14 pooled embedding

    def __init__(self, device: str = "cpu",
                 weights: str = "openai/clip-vit-large-patch14",
                 image_size: int = 224, **kw):
        super().__init__(device=device)
        import torch
        from transformers import CLIPModel, CLIPImageProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(weights).vision_model.to(device).eval()
        self.proj = CLIPModel.from_pretrained(weights).visual_projection.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        self.processor = CLIPImageProcessor.from_pretrained(weights)
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
            pooled = self.model(pixel_values=x).pooler_output
            emb = self.proj(pooled)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().cpu().numpy()

    def _to_batch(self, images):
        torch = self.torch
        if isinstance(images, torch.Tensor):
            return images
        return torch.stack([self.preprocess(im) for im in images])
