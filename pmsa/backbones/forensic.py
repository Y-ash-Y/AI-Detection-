"""Forensic feature stream — handcrafted artifacts derived from generator papers.

A frozen, deterministic, cache-compatible expert (like NPR) whose features target
the *documented physical/statistical weaknesses* of frontier generators. It's
complementary to learned semantic features (SigLIP) and — unlike them — explainable.

Signals and which paper-documented weakness each exploits:
  spectral   FFT high-frequency energy + spectral slope + radial profile
             -> Nano Banana Pro over-smoothing / loss of HF detail (2512.15110),
                GPT-Image upsampling-interpolation artifacts & over-refinement
                (GPT-ImgEval 2504.02782), VAE blur in latent diffusion (2603.07455).
  color      saturation / vibrance / channel-correlation statistics
             -> NB Pro non-physical color shifts & over-saturation; GPT-4o color bias.
  noise      denoising residual statistics (PRNU proxy)
             -> real photos carry camera sensor noise that AI images lack (2603.07455).

No training, no weights — extract once and cache, exactly like the other backbones.
"""
from __future__ import annotations

import numpy as np

from .base import Backbone, register

_RADIAL = 24


@register("forensic")
class ForensicBackbone(Backbone):
    # spectral(2 + _RADIAL) + color(12) + noise(1 + _RADIAL + 3)
    dim = (2 + _RADIAL) + 12 + (1 + _RADIAL + 3)

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
        with torch.no_grad():
            x = self._to_batch(images).to(self.device).float()  # [B,3,H,W]
            spec = self._spectral(x.mean(dim=1))                 # grayscale spectrum
            color = self._color(x)
            noise = self._noise(x)
            feats = torch.cat([spec, color, noise], dim=1)
        return feats.float().cpu().numpy()

    # ---- spectral: HF energy, slope, radial profile --------------------
    def _spectral(self, gray):
        torch = self.torch
        spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1)).abs()
        radial = self._radial_profile(spec)              # [B, _RADIAL]
        radial_log = torch.log1p(radial)
        # high-frequency energy ratio (outer half of the spectrum vs total)
        half = _RADIAL // 2
        hf_ratio = (radial[:, half:].sum(1) / (radial.sum(1) + 1e-8)).unsqueeze(1)
        # spectral slope: linear fit of log-power vs log-radius
        slope = self._slope(radial_log).unsqueeze(1)
        return torch.cat([hf_ratio, slope, radial_log], dim=1)

    def _slope(self, radial_log):
        torch = self.torch
        n = radial_log.shape[1]
        r = torch.log1p(torch.arange(1, n + 1, device=radial_log.device).float())
        r = (r - r.mean())
        num = (radial_log - radial_log.mean(1, keepdim=True)) * r
        return num.sum(1) / (r.pow(2).sum() + 1e-8)

    # ---- color: saturation, vibrance, channel correlation --------------
    def _color(self, x):
        torch = self.torch
        mean = x.mean(dim=(2, 3))                          # [B,3]
        std = x.std(dim=(2, 3))                            # [B,3]
        mx, _ = x.max(dim=1); mn, _ = x.min(dim=1)
        sat = (mx - mn) / (mx + 1e-6)                      # [B,H,W]
        sat_mean = sat.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        sat_std = sat.std(dim=(1, 2)).unsqueeze(1)
        oversat = (sat > 0.6).float().mean(dim=(1, 2)).unsqueeze(1)
        corr = self._channel_corr(x)                      # [B,3]
        return torch.cat([mean, std, sat_mean, sat_std, oversat, corr], dim=1)

    def _channel_corr(self, x):
        torch = self.torch
        b = x.shape[0]
        flat = x.reshape(b, 3, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        rg = (flat[:, 0] * flat[:, 1]).mean(1)
        gb = (flat[:, 1] * flat[:, 2]).mean(1)
        rb = (flat[:, 0] * flat[:, 2]).mean(1)
        denom = flat.pow(2).mean(2).sqrt()
        eps = 1e-8
        return torch.stack([
            rg / (denom[:, 0] * denom[:, 1] + eps),
            gb / (denom[:, 1] * denom[:, 2] + eps),
            rb / (denom[:, 0] * denom[:, 2] + eps),
        ], dim=1)

    # ---- noise residual (PRNU proxy) -----------------------------------
    def _noise(self, x):
        torch = self.torch
        F = torch.nn.functional
        gray = x.mean(dim=1, keepdim=True)
        blur = F.avg_pool2d(F.pad(gray, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        residual = (gray - blur).squeeze(1)               # [B,H,W]
        energy = residual.pow(2).mean(dim=(1, 2)).unsqueeze(1)
        spec = torch.fft.fftshift(torch.fft.fft2(residual), dim=(-2, -1)).abs()
        radial = torch.log1p(self._radial_profile(spec))  # [B,_RADIAL]
        corr = self._channel_corr(x - x.mean(dim=(2, 3), keepdim=True))
        return torch.cat([energy, radial, corr], dim=1)

    # ---- shared --------------------------------------------------------
    def _radial_profile(self, spec):
        torch = self.torch
        b, h, w = spec.shape
        cy, cx = h // 2, w // 2
        yy, xx = torch.meshgrid(
            torch.arange(h, device=spec.device) - cy,
            torch.arange(w, device=spec.device) - cx,
            indexing="ij",
        )
        rad = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
        rad = (rad / rad.max() * (_RADIAL - 1)).long().clamp(0, _RADIAL - 1)
        flat_rad = rad.reshape(-1)
        out = torch.zeros(b, _RADIAL, device=spec.device)
        for bi in range(b):
            out[bi].scatter_add_(0, flat_rad, spec[bi].reshape(-1))
        counts = torch.bincount(flat_rad, minlength=_RADIAL).clamp(min=1).float()
        return out / counts[None, :]

    def _to_batch(self, images):
        torch = self.torch
        if isinstance(images, torch.Tensor):
            return images
        return torch.stack([self.preprocess(im) for im in images])
