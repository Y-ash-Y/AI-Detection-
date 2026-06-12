"""UnivFD baseline (Ojha et al. 2023): a linear probe on frozen CLIP features.

This is the floor and the reference point. Every PMSA v2 result must beat this or
the added complexity isn't earning its keep. v1's fatal flaw was having no
baseline at all; here it is step one.

The probe's decision function (log-odds) is the score; higher = more fake — the
same NP log-likelihood-ratio surrogate the rest of the pipeline expects.
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
