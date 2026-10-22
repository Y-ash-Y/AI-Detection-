"""Neyman-Pearson thresholds calibrated on REAL images only.

Decision rule:  score(x) > tau  =>  predict FAKE.
A false positive is a real image with score > tau.

The central idea of PMSA v2: tau is a function of the real-score distribution
*alone*. No fakes are needed to set it, and no fakes are needed to recalibrate it
under a new real domain. This is what makes the false-alarm guarantee cheap to
restore under shift.

Two estimators:
  - empirical_threshold:  plug-in (1-alpha) quantile. Simple, no finite-sample
    guarantee — realized FPR fluctuates around alpha.
  - conformal_threshold:  split-conformal order statistic with a finite-sample,
    distribution-free marginal guarantee  P(score(X_real) > tau) <= alpha.
"""
from __future__ import annotations

import math

import numpy as np


def _validate(scores, alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration requires nonempty finite scores")
    return scores


def empirical_threshold(real_scores: np.ndarray, alpha: float) -> float:
    """Plug-in NP threshold: smallest tau with empirical FPR <= alpha.

    tau = (1 - alpha) quantile of the real calibration scores.
    """
    real_scores = _validate(real_scores, alpha)
    if real_scores.size == 0:
        raise ValueError("empirical_threshold: no calibration scores")
    # 'higher' interpolation => realized FPR <= alpha on the calibration set.
    return float(np.quantile(real_scores, 1.0 - alpha, method="higher"))


def conformal_threshold(real_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal NP threshold with a finite-sample guarantee.

    With n exchangeable real calibration scores, set tau to the m-th smallest
    score where m = ceil((n + 1) * (1 - alpha)). Then, for a fresh real point,

        P(score(X_real) > tau) <= alpha            (marginal, distribution-free)

    Returns +inf when m > n: valid but uninformative (zero FPR and zero power).
    Minimum n for a finite tau is ceil(1/alpha) - 1.
    """
    real_scores = _validate(real_scores, alpha)
    n = real_scores.size
    if n == 0:
        raise ValueError("conformal_threshold: no calibration scores")
    m = math.ceil((n + 1) * (1.0 - alpha))
    if m > n:
        return float("inf")  # insufficient data to certify the guarantee
    order = np.sort(real_scores)
    return float(order[m - 1])  # m-th smallest (1-indexed)


def min_calibration_size(alpha: float) -> int:
    """Smallest n for which conformal_threshold can return a finite tau."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    return math.ceil(1.0 / alpha) - 1


def realized_fpr(real_scores: np.ndarray, tau: float) -> float:
    """Fraction of real scores that exceed tau (i.e. false positives)."""
    real_scores = np.asarray(real_scores, dtype=np.float64).ravel()
    if real_scores.size == 0:
        return float("nan")
    return float(np.mean(real_scores > tau))
