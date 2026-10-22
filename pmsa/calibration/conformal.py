"""Calibrator: a tau fit on real-only scores, with bookkeeping for shift studies.

A Calibrator stores the threshold and the metadata needed to reason about *which*
real domain it was calibrated on — central to the shift-experiment matrix, where
the same detector is recalibrated per real domain.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np

from .np_threshold import (
    conformal_threshold,
    empirical_threshold,
    min_calibration_size,
    realized_fpr,
)


@dataclass
class Calibrator:
    tau: float
    alpha: float
    method: str            # "conformal" | "empirical"
    n_cal: int             # number of real calibration scores used
    real_domain: str       # label of the real domain tau was fit on
    certified: bool        # sufficient n for finite tau, NOT proof of exchangeability

    def __post_init__(self):
        if (not np.isfinite(self.alpha) or not 0 < self.alpha < 1
                or self.method not in ("conformal", "empirical")
                or not isinstance(self.n_cal, int) or isinstance(self.n_cal, bool)
                or self.n_cal < 1 or np.isnan(self.tau) or self.tau == -np.inf
                or (self.method == "empirical" and not np.isfinite(self.tau))):
            raise ValueError("Invalid calibrator metadata")
        if self.certified and (self.method != "conformal" or self.n_cal < min_calibration_size(self.alpha)):
            raise ValueError("Inconsistent finite-threshold calibration metadata")

    @classmethod
    def fit(
        cls,
        real_scores: np.ndarray,
        alpha: float,
        method: str = "conformal",
        real_domain: str = "unknown",
    ) -> "Calibrator":
        real_scores = np.asarray(real_scores, dtype=np.float64).ravel()
        n = real_scores.size
        if method == "conformal":
            tau = conformal_threshold(real_scores, alpha)
            certified = n >= min_calibration_size(alpha)
        elif method == "empirical":
            tau = empirical_threshold(real_scores, alpha)
            certified = False
        else:
            raise ValueError(f"unknown calibration method: {method}")
        return cls(
            tau=tau, alpha=alpha, method=method, n_cal=n,
            real_domain=real_domain, certified=certified,
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """1 = predicted FAKE, 0 = predicted REAL."""
        scores = np.asarray(scores).ravel()
        if not np.isfinite(scores).all():
            raise ValueError("Prediction scores must be finite")
        return (scores > self.tau).astype(int)

    def fpr_on(self, real_scores: np.ndarray) -> float:
        """Realized FPR when this tau is applied to a (possibly shifted) real set."""
        return realized_fpr(real_scores, self.tau)

    def tpr_on(self, fake_scores: np.ndarray) -> float:
        """Detection power on a fake set: fraction of fakes scored above tau."""
        fake_scores = np.asarray(fake_scores).ravel()
        if fake_scores.size == 0:
            return float("nan")
        return float(np.mean(fake_scores > self.tau))

    # ---- io -------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        return cls(**json.loads(Path(path).read_text()))
