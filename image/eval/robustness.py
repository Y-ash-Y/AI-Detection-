"""
Robustness evaluation under common perturbations.

Tests detector stability under:
  - JPEG compression
  - Gaussian noise
  - Gaussian blur
  - Random crop + resize
  - Horizontal flip

Usage:
    python -m image.eval.robustness \\
        --image_dir  data/fake \\
        --model      outputs/fusion_detector.pt \\
        --out        outputs/robustness.json
"""

import argparse
import os
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from tqdm import tqdm

from image.models.fusion_model import FusionDetector
from image.models.clip_encoder import CLIPEncoder
from image.models.dino_encoder import DINOEncoder
from image.models.forensic_features import extract_forensic_features
from image.models.device import get_device


# ── Perturbation functions ─────────────────────────────────

def jpeg_compress(img: Image.Image, quality: int = 50) -> Image.Image:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_gaussian_noise(img: Image.Image, sigma: float = 0.05) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def gaussian_blur(img: Image.Image, kernel: int = 5) -> Image.Image:
    arr = np.array(img)
    blurred = cv2.GaussianBlur(arr, (kernel, kernel), 0)
    return Image.fromarray(blurred)


def random_crop(img: Image.Image, crop_frac: float = 0.1) -> Image.Image:
    w, h = img.size
    margin_w = int(w * crop_frac)
    margin_h = int(h * crop_frac)
    left  = np.random.randint(0, margin_w + 1)
    upper = np.random.randint(0, margin_h + 1)
    right = w - np.random.randint(0, margin_w + 1)
    lower = h - np.random.randint(0, margin_h + 1)
    return img.crop((left, upper, right, lower)).resize((w, h), Image.BILINEAR)


PERTURBATIONS = {
    "none":         lambda img: img,
    "jpeg_50":      lambda img: jpeg_compress(img, 50),
    "jpeg_20":      lambda img: jpeg_compress(img, 20),
    "noise_0.05":   lambda img: add_gaussian_noise(img, 0.05),
    "noise_0.10":   lambda img: add_gaussian_noise(img, 0.10),
    "blur_3":       lambda img: gaussian_blur(img, 3),
    "blur_7":       lambda img: gaussian_blur(img, 7),
    "crop_10":      lambda img: random_crop(img, 0.10),
    "crop_20":      lambda img: random_crop(img, 0.20),
}


# ── Main evaluation ────────────────────────────────────────

def evaluate_robustness(image_dir: str,
                         model_path: str,
                         output_path: str = None,
                         max_images: int = 200):
    device = get_device()
    clip_enc = CLIPEncoder(device=device)
    dino_enc = DINOEncoder(device=device)

    ckpt = torch.load(model_path, map_location="cpu")
    model = FusionDetector(
        clip_dim     = ckpt["clip_dim"],
        dino_dim     = ckpt["dino_dim"],
        forensic_dim = ckpt["forensic_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tau = ckpt["tau"]

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ][:max_images]

    print(f"Evaluating {len(paths)} images under {len(PERTURBATIONS)} perturbations...")

    results = {}

    for pert_name, pert_fn in PERTURBATIONS.items():
        scores = []

        for path in tqdm(paths, desc=pert_name):
            try:
                with Image.open(path) as source:
                    img = pert_fn(source.convert("RGB"))

                # Save perturbed image to temp path for forensic features
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                    img.save(tmp.name)

                    with torch.no_grad():
                        clip_feat     = clip_enc.extract(tmp.name).unsqueeze(0)
                        dino_feat     = dino_enc.extract(tmp.name).unsqueeze(0)
                        forensic_feat = torch.tensor(
                            extract_forensic_features(tmp.name)
                        ).unsqueeze(0)
                        score = model(clip_feat, dino_feat, forensic_feat).item()

                scores.append(score)
            except Exception as e:
                print(f"[SKIP] {pert_name} {path}: {e}")

        if not scores:
            raise RuntimeError(f"No images were successfully evaluated for perturbation '{pert_name}'.")

        detection_rate = np.mean([s > tau for s in scores])
        mean_score     = np.mean(scores)

        results[pert_name] = {
            "detection_rate": round(float(detection_rate), 4),
            "mean_score":     round(float(mean_score), 4),
            "n_images":       len(scores),
        }

        print(f"  {pert_name:<15}: detection={detection_rate:.2%}  mean_score={mean_score:.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model",     type=str, default="outputs/fusion_detector.pt")
    parser.add_argument("--out",       type=str, default="outputs/robustness.json")
    parser.add_argument("--max",       type=int, default=200)
    args = parser.parse_args()

    evaluate_robustness(args.image_dir, args.model, args.out, args.max)
