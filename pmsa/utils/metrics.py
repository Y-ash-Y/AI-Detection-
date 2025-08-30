import torch

@torch.no_grad()
def roc_curve(scores: torch.Tensor, labels: torch.Tensor):
    """
    Returns FPR, TPR, thresholds sorted by descending score.
    labels: 0=real, 1=fake ; higher scores => more 'fake'
    """
    s, idx = torch.sort(scores, descending=True)
    y = labels[idx].float()
    P = (y == 1).sum().item()
    N = (y == 0).sum().item()
    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    tpr = (tp / max(P, 1)).cpu()
    fpr = (fp / max(N, 1)).cpu()
    return fpr, tpr, s.cpu()

@torch.no_grad()
def auc_trapezoid(fpr: torch.Tensor, tpr: torch.Tensor) -> float:
    # fpr must be sorted ascending; ensure monotonic by sorting
    f, idx = torch.sort(fpr)
    t = tpr[idx]
    df = f[1:] - f[:-1]
    area = (t[1:] + t[:-1]) * 0.5 * df
    return float(area.sum().item())

@torch.no_grad()
def eer(fpr: torch.Tensor, tpr: torch.Tensor) -> float:
    # EER at FPR ~ 1 - TPR
    fnr = 1 - tpr
    diff = torch.abs(fpr - fnr)
    i = int(torch.argmin(diff).item())
    return float((fpr[i] + fnr[i]).item() / 2.0)
