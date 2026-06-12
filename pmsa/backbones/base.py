"""Backbone interface + registry. All backbones are FROZEN feature extractors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_REGISTRY: dict[str, type["Backbone"]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def build_backbone(name: str, device: str = "cpu", **kw) -> "Backbone":
    if name not in _REGISTRY:
        raise KeyError(f"unknown backbone '{name}'. registered: {list(_REGISTRY)}")
    return _REGISTRY[name](device=device, **kw)


class Backbone(ABC):
    """A frozen image -> feature-vector extractor.

    Implementations must not require gradients and should run in eval/no_grad.
    """

    name: str = "base"
    dim: int = 0

    def __init__(self, device: str = "cpu", **kw):
        self.device = device

    @abstractmethod
    def encode(self, images) -> np.ndarray:
        """images: a batch (tensor [B,3,H,W] or list of PIL). Returns [B, dim]."""

    @property
    @abstractmethod
    def preprocess(self):
        """Return a torchvision-style transform mapping PIL -> model tensor."""
