import torch
from torch.utils.data import DataLoader
from pmsa.data.structured_dataset import StructuredAVDataset
from pmsa.models.detector import LRTSurrogate
from pmsa.models.features import ResidualHighPass, block_dct2, DCTBandPool
from pmsa.utils.attacks import jpeg_compress, add_gaussian_noise, random_crop

def build_features(frames: torch.Tensor) -> torch.Tensor:
    hp = ResidualHighPass().to(frames.device)
    pool = DCTBandPool(block=8, keep=16).to(frames.device)
    res = hp(frames)
    H, W = res.shape[-2:]
    res = res[:, :, :H - (H % 8), :W - (W % 8)]
    dct_map = block_dct2(res, block=8)
    return pool(dct_map)

def eval_attack(model, tau, device, attack_fn, name):
    ds = StructuredAVDataset(n=2000, seed=99)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    S, Y = [], []
    with torch.no_grad():
        for _, frames, labels in dl:
            frames = frames.to(device)
            frames = attack_fn(frames)   # apply distortion
            feats = build_features(frames)
            s = model(feats).cpu()
            S.append(s); Y.append(labels.cpu())
    scores = torch.cat(S,0)
    labels = torch.cat(Y,0)
    fpr = float((scores[labels==0] > tau).float().mean().item())
    tpr = float((scores[labels==1] > tau).float().mean().item())
    print(f"{name:12s}  FPR={fpr:.3f}  TPR={tpr:.3f}")

def main(ckpt_path="detector.pt", device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = LRTSurrogate(in_dim=16, hid=128).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    tau = ckpt["tau"]

    print(f"Evaluating attacks at τ={tau:.4f}")
    eval_attack(model, tau, device, lambda x: x, "Clean")
    eval_attack(model, tau, device, lambda x: jpeg_compress(x, quality=30), "JPEG-30")
    eval_attack(model, tau, device, lambda x: add_gaussian_noise(x, sigma=0.1), "Noise-0.1")
    eval_attack(model, tau, device, lambda x: random_crop(x, crop_frac=0.85), "Crop-85%")

if __name__ == "__main__":
    main()
