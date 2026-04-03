import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calibrate_threshold(scores, labels, alpha=0.01):
    """
    Neyman–Pearson threshold calibration.
    """
    scores = scores.detach()
    labels = labels.detach()

    neg_scores = scores[labels == 0]
    tau = torch.quantile(neg_scores, 1 - alpha)

    return tau.item()
