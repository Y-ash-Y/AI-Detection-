import torch
import torch.nn as nn

class LRTSurrogate(nn.Module):
    """
    Approximates log-likelihood ratio Λ(x) with a tiny MLP on frame pixels.
    Higher output => 'fake'. We'll calibrate the threshold later to meet α-FPR.
    """
    def __init__(self, in_dim=3*64*64, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x).squeeze(-1)

@torch.no_grad()
def calibrate_threshold(scores: torch.Tensor, labels: torch.Tensor, alpha: float=0.01) -> float:
    """
    Calibrate τ to satisfy approx P_FA = alpha on validation reals (label=0).
    Picks the (1-alpha) quantile of real scores.
    """
    real = scores[labels == 0]
    if real.numel() == 0:
        return float("inf")
    # sort ascending; choose index for (1-alpha) quantile
    sorted_real, _ = torch.sort(real)
    idx = min(int((1 - alpha) * (len(sorted_real) - 1)), len(sorted_real) - 1)
    return float(sorted_real[idx].item())

@torch.no_grad()
def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    AUC ≈ P(score_fake > score_real). O(n*m) but fine for toy data.
    """
    real = scores[labels == 0]
    fake = scores[labels == 1]
    if real.numel() == 0 or fake.numel() == 0:
        return float("nan")
    comp = (fake[:, None] > real[None, :]).float().mean().item()
    return comp
