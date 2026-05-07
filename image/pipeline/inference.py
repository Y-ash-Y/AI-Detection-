"""
Single-image inference.

Usage:
    python -m image.pipeline.inference \\
        --image  path/to/image.jpg \\
        --model  outputs/fusion_detector.pt
"""

import argparse
import torch
import numpy as np

from image.models.fusion_model import FusionDetector
from image.models.forensic_features import extract_forensic_features
from image.models.clip_encoder import CLIPEncoder
from image.models.dino_encoder import DINOEncoder
from image.models.device import get_device


def load_model(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    model = FusionDetector(
        clip_dim     = ckpt["clip_dim"],
        dino_dim     = ckpt["dino_dim"],
        forensic_dim = ckpt["forensic_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["tau"], ckpt.get("alpha", 0.01)


def predict(image_path: str, model_path: str = "outputs/fusion_detector.pt"):
    """
    Run full inference on a single image.

    Returns:
        score : float  — log-likelihood ratio score
        pred  : int    — 0=REAL, 1=FAKE
        tau   : float  — calibrated threshold
    """
    device = get_device()

    # Load encoders
    clip_enc  = CLIPEncoder(device=device)
    dino_enc  = DINOEncoder(device=device)

    # Extract features
    clip_feat     = clip_enc.extract(image_path).unsqueeze(0)
    dino_feat     = dino_enc.extract(image_path).unsqueeze(0)
    forensic_feat = torch.tensor(
        extract_forensic_features(image_path)
    ).unsqueeze(0)

    # Load model
    model, tau, alpha = load_model(model_path)

    # Score
    with torch.no_grad():
        score = model(clip_feat, dino_feat, forensic_feat).item()

    pred = int(score > tau)

    label_str = "FAKE" if pred == 1 else "REAL"
    margin    = score - tau

    print(f"\n{'='*40}")
    print(f"Image  : {image_path}")
    print(f"Score  : {score:.4f}")
    print(f"τ      : {tau:.4f}  (α={alpha})")
    print(f"Margin : {margin:+.4f}")
    print(f"Result : {label_str}")
    print(f"{'='*40}\n")

    return score, pred, tau


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="outputs/fusion_detector.pt")
    args = parser.parse_args()

    predict(args.image, args.model)
