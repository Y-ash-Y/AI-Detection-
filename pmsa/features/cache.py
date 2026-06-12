"""npz-backed feature cache. One extraction pass on the GPU, train forever on CPU.

Layout of a cache file (one per dataset split per backbone, or fused):
    features : float32  [N, D]
    labels   : int8     [N]        1 = fake, 0 = real
    paths    : str      [N]        source image path (provenance)
    domain   : str      [N]        real-domain label (e.g. "ffhq", "imagenet")
    source   : str      [N]        generator label for fakes ("sdxl"...), "real"
    backbone : scalar str         which backbone produced `features`
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureSet:
    features: np.ndarray   # [N, D] float32
    labels: np.ndarray     # [N] int8
    paths: np.ndarray      # [N] str
    domain: np.ndarray     # [N] str
    source: np.ndarray     # [N] str
    backbone: str

    def __len__(self) -> int:
        return self.features.shape[0]

    @property
    def dim(self) -> int:
        return self.features.shape[1]

    def select(self, mask: np.ndarray) -> "FeatureSet":
        return FeatureSet(
            features=self.features[mask],
            labels=self.labels[mask],
            paths=self.paths[mask],
            domain=self.domain[mask],
            source=self.source[mask],
            backbone=self.backbone,
        )

    def where_real(self) -> "FeatureSet":
        return self.select(self.labels == 0)

    def where_domain(self, domain: str) -> "FeatureSet":
        return self.select(self.domain == domain)

    def where_source(self, source: str) -> "FeatureSet":
        return self.select(self.source == source)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features.astype(np.float32),
            labels=self.labels.astype(np.int8),
            paths=self.paths.astype(str),
            domain=self.domain.astype(str),
            source=self.source.astype(str),
            backbone=np.array(self.backbone),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        d = np.load(path, allow_pickle=False)
        return cls(
            features=d["features"],
            labels=d["labels"],
            paths=d["paths"],
            domain=d["domain"],
            source=d["source"],
            backbone=str(d["backbone"]),
        )


def concat_streams(sets: list[FeatureSet]) -> FeatureSet:
    """Late-fusion concat of multiple backbones over the SAME, aligned samples.

    All sets must share paths/labels in the same order (same extraction run).
    """
    assert sets, "no feature sets to concat"
    base = sets[0]
    for s in sets[1:]:
        if not np.array_equal(s.paths, base.paths):
            raise ValueError("feature sets are not aligned (paths differ)")
    feats = np.concatenate([s.features for s in sets], axis=1)
    return FeatureSet(
        features=feats,
        labels=base.labels,
        paths=base.paths,
        domain=base.domain,
        source=base.source,
        backbone="+".join(s.backbone for s in sets),
    )
