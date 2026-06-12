"""NPR — Neighboring Pixel Relations. A cheap artifact stream that actually transfers.

Generators upsample, and upsampling imprints local pixel correlations that survive
across architectures far better than v1's thumbnail-DCT / fake-PRNU features did.
Following Tan et al. 2024, the NPR residual is

    r = x - up(down(x, 1/2), 2)         (bilinear)

We keep the frozen-features philosophy: instead of training a CNN on r, we extract
a deterministic, handcrafted summary of r (per-channel moments + radial FFT power)
at NATIVE resolution. No training, no weights — cache it once like the others.
"""
from __future__ import annotations

import numpy as np

from .base import Backbone, register

_N_RADIAL = 32
_MOMENTS = 4  # mean, std, skew, kurtosis


@register("npr")
class NPRBackbone(Backbone):
    dim = 3 * _MOMENTS + _N_RADIAL  # 12 + 32 = 44

    def __init__(self, device: str = "cpu", image_size: int = 256, **kw):
        super().__init__(device=device)
        import torch

        self.torch = torch
        self._image_size = image_size

    @property
    def preprocess(self):
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.CenterCrop(self._image_size),
            transforms.ToTensor(),  # [3,H,W] in [0,1]
        ])

    def encode(self, images) -> np.ndarray:
        torch = self.torch
        F = torch.nn.functional
        with torch.no_grad():
            x = self._to_batch(images).to(self.device).float()  # [B,3,H,W]
            down = F.interpolate(x, scale_factor=0.5, mode="bilinear",
                                 align_corners=False, antialias=True)
            up = F.interpolate(down, size=x.shape[-2:], mode="bilinear",
                               align_corners=False)
            r = x - up  # NPR residual
            feats = self._summarize(r)
        return feats.float().cpu().numpy()

    def _summarize(self, r):
        torch = self.torch
        b = r.shape[0]
        # per-channel moments
        mean = r.mean(dim=(2, 3))
        std = r.std(dim=(2, 3)) + 1e-8
        centred = r - mean[:, :, None, None]
        skew = (centred ** 3).mean(dim=(2, 3)) / (std ** 3)
        kurt = (centred ** 4).mean(dim=(2, 3)) / (std ** 4)
        moments = torch.cat([mean, std, skew, kurt], dim=1)  # [B, 12]

        # radial FFT power on the grayscale residual
        gray = r.mean(dim=1)  # [B,H,W]
        spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1)).abs()
        radial = self._radial_profile(spec)  # [B, _N_RADIAL]
        radial = torch.log1p(radial)
        return torch.cat([moments, radial], dim=1)

    def _radial_profile(self, spec):
        torch = self.torch
        b, h, w = spec.shape
        cy, cx = h // 2, w // 2
        yy, xx = torch.meshgrid(
            torch.arange(h, device=spec.device) - cy,
            torch.arange(w, device=spec.device) - cx,
            indexing="ij",
        )
        rad = torch.sqrt((yy.float() ** 2 + xx.float() ** 2))
        rad = (rad / rad.max() * (_N_RADIAL - 1)).long().clamp(0, _N_RADIAL - 1)
        out = torch.zeros(b, _N_RADIAL, device=spec.device)
        flat_rad = rad.reshape(-1)
        for bi in range(b):
            out[bi].scatter_add_(0, flat_rad, spec[bi].reshape(-1))
        counts = torch.bincount(flat_rad, minlength=_N_RADIAL).clamp(min=1).float()
        return out / counts[None, :]

    def _to_batch(self, images):
        torch = self.torch
        if isinstance(images, torch.Tensor):
            return images
        return torch.stack([self.preprocess(im) for im in images])
