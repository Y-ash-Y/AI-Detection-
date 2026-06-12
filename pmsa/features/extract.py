"""The one heavy pass: manifest -> per-backbone npz feature caches.

Run this on the GPU that holds the data (P100 on Kaggle). Every downstream
experiment then loads npz and trains on the laptop in minutes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backbones import build_backbone
from ..data.manifest import Manifest
from .cache import FeatureSet


def _load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


class _ManifestDataset:
    """Maps manifest rows -> (preprocessed tensor, index). torch Dataset-shaped."""

    def __init__(self, manifest: Manifest, preprocess):
        self.records = manifest.records
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        img = _load_image(self.records[i].path)
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
) -> FeatureSet:
    """Extract one backbone's features over the whole manifest and cache to npz."""
    import torch
    from torch.utils.data import DataLoader

    kw = {"image_size": image_size}
    if weights:
        kw["weights"] = weights
    bb = build_backbone(backbone_name, device=device, **kw)

    ds = _ManifestDataset(manifest, bb.preprocess)
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
