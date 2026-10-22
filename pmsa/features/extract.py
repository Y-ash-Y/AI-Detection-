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
from ..preprocessing import ImagePolicy, load_image
from ..artifacts import sha256_file


def _load_image(path: str, normalize_jpeg: bool = False):
    return load_image(path, ImagePolicy(normalize_jpeg=normalize_jpeg))


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
    revision: str | None = None,
    use_fast: bool = False,
) -> FeatureSet:
    """Extract one backbone's features over the whole manifest and cache to npz."""
    import torch
    from torch.utils.data import DataLoader

    if Path(out_path).exists():
        raise FileExistsError(f"Cache already exists: {out_path}; choose a NEW tag")
    if not len(manifest):
        raise ValueError("Empty extraction manifest")

    kw = {"image_size": image_size}
    if weights:
        kw["weights"] = weights
    if backbone_name in ("siglip", "clip_l14", "dino_b14"):
        kw.update(revision=revision, use_fast=use_fast)
    bb = build_backbone(backbone_name, device=device, **kw)
    from importlib.metadata import version
    metadata = {"schema_version": 1,
                "image_policy": ImagePolicy(normalize_jpeg=normalize_jpeg).to_dict(),
                "backbone_kwargs": {"image_size": image_size, **getattr(bb, "load_config", {})},
                "packages": {p: version(p) for p in ("numpy", "Pillow", "torch", "torchvision")}}
    if getattr(bb, "load_config", {}).get("weights"):
        metadata["packages"]["transformers"] = version("transformers")
    # Fail before image hashing or a GPU pass if provenance cannot be recorded.
    from ..artifacts import validate_stream_metadata
    validate_stream_metadata({backbone_name: metadata})
    print("Hashing files for exact-duplicate grouping...", flush=True)
    hashes = np.array([sha256_file(r.path) for r in manifest.records])

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
        metadata=metadata, content_hashes=hashes,
        related_groups=np.array([r.group_id for r in recs], dtype=str),
    )
    fs.save(out_path)
    print(f"[{backbone_name}] saved {len(ds)} x {bb.dim} -> {out_path}", flush=True)
    return fs
