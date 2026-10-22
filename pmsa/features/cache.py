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

from dataclasses import dataclass, field
import hashlib
import json
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
    metadata: dict = field(default_factory=dict)
    content_hashes: np.ndarray | None = None
    related_groups: np.ndarray | None = None

    def validate(self):
        if self.features.ndim != 2 or not np.isfinite(self.features).all():
            raise ValueError("Features must be a finite [N,D] matrix")
        for key in ("labels", "paths", "domain", "source", "content_hashes", "related_groups"):
            x = getattr(self, key)
            if x is not None and (x.ndim != 1 or len(x) != len(self)):
                raise ValueError(f"Invalid cache field shape: {key}")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Labels must be 0=real or 1=fake")

    def groups(self):
        from ..data.splits import group_ids
        # Legacy features can expose identical encodings, but cannot establish
        # image-level uniqueness. Such runs must remain exploratory.
        hashes = self.content_hashes
        if hashes is None:
            hashes = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in self.features])
        return group_ids(self.paths, hashes, self.related_groups)

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
            metadata=self.metadata.copy(),
            content_hashes=None if self.content_hashes is None else self.content_hashes[mask],
            related_groups=None if self.related_groups is None else self.related_groups[mask],
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
        self.validate()
        optional = {k: getattr(self, k).astype(str) for k in ("content_hashes", "related_groups")
                    if getattr(self, k) is not None}
        np.savez_compressed(
            path,
            features=self.features.astype(np.float32),
            labels=self.labels.astype(np.int8),
            paths=self.paths.astype(str),
            domain=self.domain.astype(str),
            source=self.source.astype(str),
            backbone=np.array(self.backbone),
            metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
            **optional,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        with np.load(path, allow_pickle=False) as d:
            obj = cls(d["features"], d["labels"], d["paths"], d["domain"], d["source"], str(d["backbone"]),
                      metadata=json.loads(str(d["metadata"])) if "metadata" in d else {},
                      content_hashes=d["content_hashes"] if "content_hashes" in d else None,
                      related_groups=d["related_groups"] if "related_groups" in d else None)
        obj.validate()
        return obj


def validate_alignment(sets):
    if not sets:
        raise ValueError("No feature streams supplied")
    for s in sets:
        s.validate()
        for key in ("paths", "labels", "domain", "source", "content_hashes", "related_groups"):
            if not np.array_equal(getattr(s, key), getattr(sets[0], key)):
                raise ValueError(f"Feature streams not aligned: {key}")


def concat_streams(sets: list[FeatureSet]) -> FeatureSet:
    """Late-fusion concat of multiple backbones over the SAME, aligned samples.

    All sets must share paths/labels in the same order (same extraction run).
    """
    validate_alignment(sets)
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
        metadata={"streams": {s.backbone: s.metadata for s in sets}},
        content_hashes=base.content_hashes, related_groups=base.related_groups,
    )
