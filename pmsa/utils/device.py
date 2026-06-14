"""Device selection. T4 (CUDA) for extraction, M-series (MPS) for training.

The whole architecture is shaped by this split: a frozen backbone runs ONE
extraction pass on the GPU that has the data (T4 on Kaggle; P100 is sm_60 and
incompatible with Kaggle's current PyTorch), caches features to
npz, and every downstream experiment trains in minutes on the laptop (MPS).
"""
from __future__ import annotations


def get_device(prefer: str | None = None) -> str:
    """Return the best available torch device string.

    prefer: optional override ("cuda", "mps", "cpu"). Falls back gracefully.
    """
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
