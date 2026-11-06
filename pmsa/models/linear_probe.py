"""Standardized logistic-regression baseline on frozen features (SigLIP by default).

This reusable head also supports a forensic-only ablation. Using SigLIP features
does not constitute a reproduction of the CLIP-based UnivFD baseline.

The decision function is a discriminative score; higher = more generated-like.
It is not a validated likelihood ratio or a calibrated deployment probability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class LinearProbe:
    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Log-odds of FAKE. Higher = more fake."""
        Xs = self.scaler.transform(X)
        return self.clf.decision_function(Xs).ravel()

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "clf": self.clf}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        import joblib

        obj = cls.__new__(cls)
        state = joblib.load(path)
        obj.scaler, obj.clf = state["scaler"], state["clf"]
        return obj
