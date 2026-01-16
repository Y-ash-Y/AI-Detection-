import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pmsa.data.structured_dataset import StructuredAVDataset
from pmsa.models.clip_features import CLIPFeatureExtractor
from pmsa.models.dino_features import DINOFeatureExtractor
from pmsa.models.features import ResidualHighPass, block_dct2, DCTBandPool

def extract_forensic(frames, hp, pool):
    with torch.no_grad():
        res = hp(frames)
        H, W = res.shape[-2:]
        res = res[:, :, :H - (H % 8), :W - (W % 8)]
        dct_map = block_dct2(res, block=8)
        feats = pool(dct_map)
    return feats.cpu().numpy()

def main(
    out_dir="feature_cache",
    batch_size=8,
    n_samples=6000,
    device="cpu"
):
    os.makedirs(out_dir, exist_ok=True)

    ds = StructuredAVDataset(n=n_samples, seed=42)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    clip_ext = CLIPFeatureExtractor(device=device)
    dino_ext = DINOFeatureExtractor(device=device)

    hp = ResidualHighPass().to(device)
    pool = DCTBandPool(block=8, keep=16).to(device)

    all_clip, all_dino, all_forensic, all_y = [], [], [], []

    for _, frames, labels in tqdm(dl, desc="Caching features"):
        frames = frames.to(device).float()

        with torch.no_grad():
            clip_f = clip_ext(frames)
            dino_f = dino_ext(frames)
            forensic_f = extract_forensic(frames, hp, pool)

        all_clip.append(clip_f.cpu().numpy())
        all_dino.append(dino_f.cpu().numpy())
        all_forensic.append(forensic_f)
        all_y.append(labels.numpy())

    np.savez_compressed(
        os.path.join(out_dir, "features.npz"),
        clip=np.concatenate(all_clip, axis=0),
        dino=np.concatenate(all_dino, axis=0),
        forensic=np.concatenate(all_forensic, axis=0),
        label=np.concatenate(all_y, axis=0),
    )

    print("Saved cached features to:", out_dir)

if __name__ == "__main__":
    main()

