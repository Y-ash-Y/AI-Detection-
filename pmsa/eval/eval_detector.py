import torch
from torch.utils.data import DataLoader
from pmsa.data.datasets import ToyAVDataset
from pmsa.models.detector import LRTSurrogate
from pmsa.models.features import ResidualHighPass, block_dct2, DCTBandPool
from pmsa.utils.metrics import roc_curve, auc_trapezoid, eer
from pmsa.data.structured_dataset import StructuredAVDataset
def build_features(frames: torch.Tensor) -> torch.Tensor:
    hp = ResidualHighPass().to(frames.device)
    pool = DCTBandPool(block=8, keep=16).to(frames.device)
    res = hp(frames)
    H, W = res.shape[-2:]
    res = res[:, :, :H - (H % 8), :W - (W % 8)]
    dct_map = block_dct2(res, block=8)
    return pool(dct_map)

def main(ckpt_path="detector.pt", device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = LRTSurrogate(in_dim=16, hid=128).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    ds = StructuredAVDataset(n=3000, seed=123)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    S, Y = [], []
    with torch.no_grad():
        for _, frames, labels in dl:
            frames = frames.to(device)
            feats = build_features(frames)
            s = model(feats).cpu()
            S.append(s); Y.append(labels.cpu())
    scores = torch.cat(S, 0)
    labels = torch.cat(Y, 0)

    fpr, tpr, _ = roc_curve(scores, labels)
    print(f"AUC={auc_trapezoid(fpr,tpr):.3f}  EER={eer(fpr,tpr):.3f}")

    # If ckpt has τ, show operating point
    tau = ckpt.get("tau", None)
    if tau is not None:
        fpr_tau = float((scores[labels==0] > tau).float().mean().item())
        tpr_tau = float((scores[labels==1] > tau).float().mean().item())
        print(f"At τ={tau:.4f}: FPR={fpr_tau:.3f} TPR={tpr_tau:.3f}")

if __name__ == "__main__":
    main()
