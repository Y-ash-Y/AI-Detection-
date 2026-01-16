import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pmsa.utils.attacks import jpeg_compress, add_gaussian_noise, random_crop
import random
from pmsa.data.datasets import ToyAVDataset
from pmsa.models.detector import LRTSurrogate
from pmsa.models.features import ResidualHighPass, block_dct2, DCTBandPool
from pmsa.utils.logging import set_seed
from pmsa.utils.metrics import roc_curve, auc_trapezoid, eer
from pmsa.data.datasets import ToyAVDataset
from pmsa.data.structured_dataset import StructuredAVDataset

def build_features(frames: torch.Tensor) -> torch.Tensor:
    """
    frames: (B,3,64,64) -> feature vector (B, D)
    Pipeline: residual high-pass -> block DCT -> zig-zag mean-pool -> feature
    """
    hp = ResidualHighPass().to(frames.device)
    dct_pool = DCTBandPool(block=8, keep=16).to(frames.device)
    res = hp(frames)                   # (B,1,H,W)
    # ensure divisible by 8
    H, W = res.shape[-2:]
    res = res[:, :, :H - (H % 8), :W - (W % 8)]
    dct_map = block_dct2(res, block=8) # (B,1,H,W)
    feat = dct_pool(dct_map)           # (B,16)
    return feat

def train(epochs=3, batch_size=128, lr=2e-3, device="cpu", seed=42, save_path="detector.pt", dataset="structured"):
    set_seed(seed)
    if dataset == "structured":
        full = StructuredAVDataset(n=6000, seed=seed)
    else:
        full = ToyAVDataset(n=6000, seed=seed)
    n_full = len(full)
    n_train = int(0.7 * n_full)
    n_val   = int(0.15 * n_full)
    n_test  = n_full - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # feature dim is 16; keep detector simple
    model = LRTSurrogate(in_dim=16, hid=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # ---- train
    model.train()
    for ep in range(epochs):
        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")
        for _, frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.float().to(device)
            with torch.no_grad():
                feats = build_features(frames)   # (B,16)
            logits = model(feats)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=loss.detach().item())
            if model.training:
                aug_type = random.choice(["none","jpeg","noise","crop"])
                if aug_type == "jpeg":
                    frames = jpeg_compress(frames, quality=30)
                elif aug_type == "noise":
                    frames = add_gaussian_noise(frames, sigma=0.1)
                elif aug_type == "crop":
                    frames = random_crop(frames, crop_frac=0.85)

    # ---- validate (for threshold calibration + AUC/EER)
    model.eval()
    def collect(dl):
        S, Y = [], []
        with torch.no_grad():
            for _, frames, labels in dl:
                frames = frames.to(device)
                feats = build_features(frames)
                s = model(feats).cpu()
                y = labels.cpu()
                S.append(s); Y.append(y)
        return torch.cat(S,0), torch.cat(Y,0)

    val_scores, val_labels   = collect(val_dl)
    test_scores, test_labels = collect(test_dl)

    # ROC/AUC/EER on val
    vfpr, vtpr, vthr = roc_curve(val_scores, val_labels)
    vauc = auc_trapezoid(vfpr, vtpr)
    veer = eer(vfpr, vtpr)

    # Choose τ for α=1% on validation reals
    real_scores = val_scores[val_labels == 0]
    real_sorted, _ = torch.sort(real_scores)
    idx = min(int((1 - 0.01) * (len(real_sorted) - 1)), max(len(real_sorted) - 1, 0))
    tau = float(real_sorted[idx].item())

    # Apply τ on test to get TPR@FPR≈1%
    t_fpr = float((test_scores[test_labels == 0] > tau).float().mean().item())
    t_tpr = float((test_scores[test_labels == 1] > tau).float().mean().item())

    # ROC/AUC/EER on test for completeness
    tfpr, ttpr, _ = roc_curve(test_scores, test_labels)
    tauc = auc_trapezoid(tfpr, ttpr)
    teer = eer(tfpr, ttpr)

    print(f"[VAL] AUC={vauc:.3f}  EER={veer:.3f}  (α=1%) τ={tau:.4f}")
    print(f"[TEST] AUC={tauc:.3f} EER={teer:.3f}  FPR@τ≈{t_fpr:.3f} TPR@τ={t_tpr:.3f}")

    torch.save({"state_dict": model.state_dict(),
                "tau": tau,
                "feature": {"type": "residual+dct16"}}, save_path)
    print(f"Saved: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="detector.pt")
    parser.add_argument("--dataset", type=str, default="structured", choices=["structured","toy"])
    args = parser.parse_args()
    train(**vars(args))
