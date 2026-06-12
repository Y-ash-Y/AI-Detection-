"""Detection metrics with bootstrap confidence intervals.

Score convention everywhere: higher = more FAKE. Labels: 1 = fake, 0 = real.
Every reported number carries a CI — v1's headline failure was point estimates
with no uncertainty and no baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney U) statistic. No sklearn dependency."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks for ties
    _assign_tie_ranks(scores, ranks)
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_target: float) -> float:
    """TPR (power) at a target FPR, threshold set on the real scores in this set."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    real = scores[labels == 0]
    fake = scores[labels == 1]
    if real.size == 0 or fake.size == 0:
        return float("nan")
    tau = np.quantile(real, 1.0 - fpr_target, method="higher")
    return float(np.mean(fake > tau))


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal error rate."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    thr = np.unique(scores)
    real = scores[labels == 0]
    fake = scores[labels == 1]
    if real.size == 0 or fake.size == 0:
        return float("nan")
    best, best_gap = 0.5, np.inf
    for t in thr:
        fpr = np.mean(real > t)
        fnr = np.mean(fake <= t)
        gap = abs(fpr - fnr)
        if gap < best_gap:
            best_gap, best = gap, (fpr + fnr) / 2.0
    return float(best)


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
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
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
