"""Detection metrics with bootstrap confidence intervals.

Score convention everywhere: higher = more FAKE. Labels: 1 = fake, 0 = real.
Every reported number carries a CI — v1's headline failure was point estimates
with no uncertainty and no baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _binary_inputs(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if (scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape
            or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Metrics require aligned finite 1D scores and binary labels")
    return scores, labels


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney U) statistic. No sklearn dependency."""
    scores, labels = _binary_inputs(scores, labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty_like(scores, dtype=np.float64)
    # average ranks for ties
    _assign_tie_ranks(scores, ranks)
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks[:] = (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_target: float) -> float:
    """TPR (power) at a target FPR, threshold set on the real scores in this set."""
    scores, labels = _binary_inputs(scores, labels)
    if not np.isfinite(fpr_target) or not 0 < fpr_target < 1:
        raise ValueError("fpr_target must be in (0,1)")
    real = scores[labels == 0]
    fake = scores[labels == 1]
    if real.size == 0 or fake.size == 0:
        return float("nan")
    tau = np.quantile(real, 1.0 - fpr_target, method="higher")
    return float(np.mean(fake > tau))


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal error rate."""
    scores, labels = _binary_inputs(scores, labels)
    thr = np.unique(scores)
    real = scores[labels == 0]
    fake = scores[labels == 1]
    if real.size == 0 or fake.size == 0:
        return float("nan")
    # O(N log N), preserving strict score > tau and first-minimum tie behavior.
    fpr = (real.size - np.searchsorted(np.sort(real), thr, side="right")) / real.size
    fnr = np.searchsorted(np.sort(fake), thr, side="right") / fake.size
    best = np.argmin(np.abs(fpr - fnr))
    return float((fpr[best] + fnr[best]) / 2)


@dataclass
class CI:
    point: float
    lo: float
    hi: float

    def __repr__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def bootstrap_ci(
    fn,
    scores: np.ndarray,
    labels: np.ndarray,
    n: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> CI:
    """Stratified bootstrap CI for any metric fn(scores, labels)."""
    scores, labels = _binary_inputs(scores, labels)
    if n < 1 or not 0 < ci < 1 or scores.size != labels.size or not np.isfinite(scores).all():
        raise ValueError("Invalid bootstrap inputs")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Bootstrap evaluation requires both binary classes")
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    stats = np.empty(n, dtype=np.float64)
    for b in range(n):
        p = rng.choice(pos_idx, pos_idx.size, replace=True)
        q = rng.choice(neg_idx, neg_idx.size, replace=True)
        idx = np.concatenate([p, q])
        stats[b] = fn(scores[idx], labels[idx])
    lo = float(np.nanpercentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.nanpercentile(stats, (1 + ci) / 2 * 100))
    return CI(point=float(fn(scores, labels)), lo=lo, hi=hi)


def full_report(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_targets=(0.01, 0.05),
    bootstrap_n: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """AUC, EER, and TPR@FPR — each with a bootstrap CI."""
    out = {
        "n_real": int(np.sum(labels == 0)),
        "n_fake": int(np.sum(labels == 1)),
        "auc": _ci_dict(bootstrap_ci(roc_auc, scores, labels, bootstrap_n, ci, seed)),
        "eer": _ci_dict(bootstrap_ci(eer, scores, labels, bootstrap_n, ci, seed)),
    }
    for t in fpr_targets:
        out[f"tpr_at_{t:g}"] = _ci_dict(
            bootstrap_ci(lambda s, l: tpr_at_fpr(s, l, t), scores, labels,
                         bootstrap_n, ci, seed)
        )
    return out


def _ci_dict(c: CI) -> dict:
    return {"point": c.point, "lo": c.lo, "hi": c.hi}


def binomial_rate(successes, total, ci=.95):
    """Wilson interval conditional on fixed threshold; assumes independent images."""
    from statistics import NormalDist
    if not 0 < ci < 1 or not 0 <= successes <= total:
        raise ValueError("Invalid rate/interval inputs")
    if total == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    z, p = NormalDist().inv_cdf((1 + ci) / 2), successes / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return {"point": float(p), "lo": max(0., float(center-half)),
            "hi": min(1., float(center+half)), "n": int(total)}


def operating_report(scores, labels, tau, ci=.95):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if (scores.ndim != 1 or scores.shape != labels.shape or not len(scores)
            or not np.isfinite(scores).all() or not np.isfinite(tau)
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Invalid operating-point inputs")
    pred, real, fake = scores > tau, labels == 0, labels == 1
    return {"tau": float(tau),
            "fpr": binomial_rate(int(pred[real].sum()), int(real.sum()), ci),
            "tpr": binomial_rate(int(pred[fake].sum()), int(fake.sum()), ci),
            "accuracy": binomial_rate(int((pred == labels).sum()), len(labels), ci),
            "note": "Wilson intervals assume independent images and a frozen threshold."}
