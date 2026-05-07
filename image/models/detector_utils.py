import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def calibrate_threshold(scores: torch.Tensor,
                         labels: torch.Tensor,
                         alpha: float = 0.01) -> float:
    """
    Neyman-Pearson threshold calibration.

    Fix FPR ≤ alpha on the REAL class, then pick the threshold
    that achieves exactly that bound.

    H0 = real  (label = 0)
    H1 = fake  (label = 1)

    tau = (1 - alpha) quantile of scores on real samples.
    Then:  P(s(x) > tau | x is real) ≤ alpha  by construction.

    Args:
        scores : (N,) float tensor — model output (logits)
        labels : (N,) float tensor — 0=real, 1=fake
        alpha  : target FPR (default 1%)

    Returns:
        tau : float threshold
    """
    real_scores = scores[labels == 0]

    if real_scores.numel() == 0:
        raise ValueError("No real samples found for calibration.")

    tau = torch.quantile(real_scores.float(), 1.0 - alpha)
    return tau.item()


def compute_metrics(scores: np.ndarray,
                    labels: np.ndarray,
                    tau: float) -> dict:
    """
    Compute full evaluation metrics.

    Returns dict with:
        auc       : AUROC
        tpr_at_1  : TPR @ FPR = 1%
        tpr_at_5  : TPR @ FPR = 5%
        eer       : Equal Error Rate
        accuracy  : accuracy at threshold tau
        fpr_at_tau: actual FPR achieved at tau
    """
    # AUC
    auc = roc_auc_score(labels, scores)

    # ROC curve
    fprs, tprs, thresholds = roc_curve(labels, scores)

    # TPR @ FPR = 1%
    idx_1 = np.searchsorted(fprs, 0.01)
    tpr_at_1 = tprs[min(idx_1, len(tprs) - 1)]

    # TPR @ FPR = 5%
    idx_5 = np.searchsorted(fprs, 0.05)
    tpr_at_5 = tprs[min(idx_5, len(tprs) - 1)]

    # EER: point where FPR = FNR = 1 - TPR
    fnrs = 1 - tprs
    eer_idx = np.argmin(np.abs(fprs - fnrs))
    eer = (fprs[eer_idx] + fnrs[eer_idx]) / 2.0

    # Accuracy at tau
    preds = (scores > tau).astype(int)
    accuracy = (preds == labels).mean()

    # Actual FPR at tau (on real samples)
    real_mask = labels == 0
    fpr_at_tau = (scores[real_mask] > tau).mean()

    return {
        "auc":       round(float(auc), 4),
        "tpr_at_1%": round(float(tpr_at_1), 4),
        "tpr_at_5%": round(float(tpr_at_5), 4),
        "eer":       round(float(eer), 4),
        "accuracy":  round(float(accuracy), 4),
        "fpr_at_tau": round(float(fpr_at_tau), 4),
        "tau":       round(float(tau), 4),
    }
