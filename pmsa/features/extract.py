"""The one heavy pass: manifest -> per-backbone npz feature caches.

Run this on the GPU that holds the data (T4 on Kaggle). Every downstream
experiment then loads npz and trains on the laptop in minutes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backbones import build_backbone
from ..data.manifest import Manifest
from .cache import FeatureSet


def _load_image(path: str, normalize_jpeg: bool = False):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if normalize_jpeg:
        # re-encode everything to the same JPEG quality so the model can't cheat on
        # source compression (reals JPEG vs fakes PNG) — it must learn generation.
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


class _ManifestDataset:
    """Maps manifest rows -> (preprocessed tensor, index). torch Dataset-shaped."""

    def __init__(self, manifest: Manifest, preprocess, normalize_jpeg: bool = False):
        self.records = manifest.records
        self.preprocess = preprocess
        self.normalize_jpeg = normalize_jpeg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        img = _load_image(self.records[i].path, self.normalize_jpeg)
        return self.preprocess(img), i


def extract_backbone(
    manifest: Manifest,
    backbone_name: str,
    out_path: str | Path,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    weights: str = "",
    image_size: int = 224,
    log_every: int = 20,
    normalize_jpeg: bool = False,
) -> FeatureSet:
    """Extract one backbone's features over the whole manifest and cache to npz."""
    import torch
    from torch.utils.data import DataLoader

    kw = {"image_size": image_size}
    if weights:
        kw["weights"] = weights
    bb = build_backbone(backbone_name, device=device, **kw)

    ds = _ManifestDataset(manifest, bb.preprocess, normalize_jpeg=normalize_jpeg)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=(device == "cuda"))

    feats = np.empty((len(ds), bb.dim), dtype=np.float32)
    for step, (x, idx) in enumerate(loader):
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        feats[idx.numpy()] = bb.encode(x)
        if step % log_every == 0:
            print(f"[{backbone_name}] {step * batch_size}/{len(ds)}", flush=True)

    recs = manifest.records
    fs = FeatureSet(
        features=feats,
        labels=np.array([r.label for r in recs], dtype=np.int8),
        paths=np.array([r.path for r in recs], dtype=str),
        domain=np.array([r.domain for r in recs], dtype=str),
        source=np.array([r.source for r in recs], dtype=str),
        backbone=backbone_name,
    )
    fs.save(out_path)
    print(f"[{backbone_name}] saved {len(ds)} x {bb.dim} -> {out_path}", flush=True)
    return fs
