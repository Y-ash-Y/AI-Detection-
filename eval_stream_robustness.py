"""
Per-stream robustness analysis.
Evaluates each feature stream independently to identify which
stream contributes most to adversarial vulnerability.
"""
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json

from image.models.clip_encoder      import CLIPEncoder
from image.models.dino_encoder      import DINOEncoder
from image.models.forensic_features import extract_forensic_features
from image.models.fusion_model      import FusionDetector

DEVICE = torch.device("cpu")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_model(path):
    ckpt    = torch.load(path, map_location="cpu")
    fusion  = FusionDetector(
        clip_dim=ckpt["clip_dim"],
        dino_dim=ckpt["dino_dim"],
        forensic_dim=ckpt["forensic_dim"])
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()
    return (fusion, ckpt["tau"],
            joblib.load(ckpt["scaler_clip"]),
            joblib.load(ckpt["scaler_dino"]),
            joblib.load(ckpt["scaler_forensic"]))


def get_zero_clip(batch_size, clip_dim=512):
    return torch.zeros(batch_size, clip_dim)

def get_zero_dino(batch_size, dino_dim=384):
    return torch.zeros(batch_size, dino_dim)


def main():
    fusion, tau, sc_clip, sc_dino, sc_forensic = \
        load_model("outputs/fusion_detector_mixed.pt")

    clip_enc = CLIPEncoder(device=DEVICE)
    dino_enc = DINOEncoder(device=DEVICE)

    fake_dir   = Path(
        "data/kaggle_deepfake/real_vs_fake/real-vs-fake/test/fake")
    fake_paths = sorted(fake_dir.glob("*.jpg"))[:100]

    print(f"Evaluating {len(fake_paths)} images per stream...\n")

    # Storage
    scores = {
        "clip_only"      : [],
        "dino_only"      : [],
        "forensic_only"  : [],
        "full"           : [],
    }

    for path in tqdm(fake_paths):
        try:
            img = TRANSFORM(
                Image.open(path).convert("RGB")).unsqueeze(0)

            # Extract all features
            with torch.no_grad():
                clip_f = clip_enc.encode_tensor(img)
                dino_f = dino_enc.encode_tensor(img)
            forensic_f = extract_forensic_features(str(path))

            clip_n = torch.from_numpy(
                sc_clip.transform(clip_f.numpy())).float()
            dino_n = torch.from_numpy(
                sc_dino.transform(dino_f.numpy())).float()
            forensic_n = torch.from_numpy(
                sc_forensic.transform(
                    forensic_f.reshape(1,-1))).float()

            with torch.no_grad():
                # Full model
                s_full = fusion(clip_n, dino_n, forensic_n).item()

                # CLIP only (zero out other streams)
                s_clip = fusion(
                    clip_n,
                    get_zero_dino(1),
                    torch.zeros(1, forensic_n.shape[1])
                ).item()

                # DINOv2 only
                s_dino = fusion(
                    get_zero_clip(1),
                    dino_n,
                    torch.zeros(1, forensic_n.shape[1])
                ).item()

                # Forensic only
                s_forensic = fusion(
                    get_zero_clip(1),
                    get_zero_dino(1),
                    forensic_n
                ).item()

            scores["full"].append(s_full)
            scores["clip_only"].append(s_clip)
            scores["dino_only"].append(s_dino)
            scores["forensic_only"].append(s_forensic)

        except Exception:
            continue

    print(f"\n{'Stream':<20} {'Mean Score':<14} "
          f"{'Detection Rate':<16} {'vs Threshold'}")
    print("-" * 65)
    for stream, vals in scores.items():
        arr = np.array(vals)
        mean_s = arr.mean()
        det_r  = (arr > tau).mean()
        print(f"{stream:<20} {mean_s:<14.4f} "
              f"{det_r:<16.4f} τ={tau:.4f}")

    results = {
        k: {
            "mean_score"      : float(np.mean(v)),
            "detection_rate"  : float(np.mean(
                np.array(v) > tau)),
            "std_score"       : float(np.std(v)),
        }
        for k, v in scores.items()
    }
    results["tau"] = float(tau)

    with open("outputs/stream_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → outputs/stream_robustness.json")


if __name__ == "__main__":
    main()