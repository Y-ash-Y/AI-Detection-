"""
Zero-shot evaluation on Kaggle 140k Real vs Fake Faces dataset.
Tests generalisation: model trained on CIFAKE, evaluated on unseen dataset.

Usage:
    python3 eval_kaggle.py \
        --model   outputs/fusion_detector_v3.pt \
        --real    data/kaggle_deepfake/real_vs_fake/real-vs-fake/test/real \
        --fake    data/kaggle_deepfake/real_vs_fake/real-vs-fake/test/fake \
        --out     outputs/kaggle_zeroshot.json
"""

import argparse
import json
import numpy as np
import torch
import joblib
from pathlib import Path
from tqdm import tqdm

from image.pipeline.feature_extraction import FeaturePipeline
from image.models.fusion_model import FusionDetector
from image.models.detector_utils import compute_metrics


def load_model(model_path: str):
    """Load model + scalers from saved checkpoint."""
    ckpt = torch.load(model_path, map_location="cpu")

    model = FusionDetector(
        clip_dim     = ckpt["clip_dim"],
        dino_dim     = ckpt["dino_dim"],
        forensic_dim = ckpt["forensic_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tau   = ckpt["tau"]
    alpha = ckpt["alpha"]

    # Load scalers
    scaler_clip     = joblib.load(ckpt["scaler_clip"])
    scaler_dino     = joblib.load(ckpt["scaler_dino"])
    scaler_forensic = joblib.load(ckpt["scaler_forensic"])

    return model, tau, alpha, scaler_clip, scaler_dino, scaler_forensic


def extract_and_predict(
    image_paths,
    pipeline,
    model,
    scaler_clip,
    scaler_dino,
    scaler_forensic,
    batch_size=64,
):
    """Extract features + run inference in batches."""
    all_scores = []
    failed     = 0

    for i in tqdm(range(0, len(image_paths), batch_size),
                  desc="Extracting features"):
        batch = image_paths[i:i + batch_size]
        batch_clip, batch_dino, batch_forensic = [], [], []

        for path in batch:
            try:
                clip_f, dino_f, forensic_f = pipeline.extract(str(path))
                batch_clip.append(clip_f)
                batch_dino.append(dino_f)
                batch_forensic.append(forensic_f)
            except Exception as e:
                failed += 1
                continue

        if not batch_clip:
            continue

        # Stack → normalize → predict
        clip_np     = np.stack(batch_clip)
        dino_np     = np.stack(batch_dino)
        forensic_np = np.stack(batch_forensic)

        clip_np     = scaler_clip.transform(clip_np)
        dino_np     = scaler_dino.transform(dino_np)
        forensic_np = scaler_forensic.transform(forensic_np)

        clip_t     = torch.from_numpy(clip_np).float()
        dino_t     = torch.from_numpy(dino_np).float()
        forensic_t = torch.from_numpy(forensic_np).float()

        with torch.no_grad():
            scores = model(clip_t, dino_t, forensic_t).numpy()
        all_scores.extend(scores.tolist())

    print(f"  Failed extractions: {failed}")
    return np.array(all_scores)


def main(model_path, real_dir, fake_dir, out_path, max_images=2000):
    print(f"Loading model from {model_path}...")
    model, tau, alpha, sc_clip, sc_dino, sc_forensic = load_model(model_path)

    print("Initialising feature pipeline...")
    pipeline = FeaturePipeline()

    # Collect image paths
    real_paths = sorted(Path(real_dir).glob("*.jpg"))[:max_images]
    fake_paths = sorted(Path(fake_dir).glob("*.jpg"))[:max_images]

    # Also try .png if .jpg is empty
    if len(real_paths) == 0:
        real_paths = sorted(Path(real_dir).glob("*.png"))[:max_images]
    if len(fake_paths) == 0:
        fake_paths = sorted(Path(fake_dir).glob("*.png"))[:max_images]

    print(f"Real images: {len(real_paths)}")
    print(f"Fake images: {len(fake_paths)}")

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise ValueError("No images found. Check --real and --fake paths.")

    # Extract features + predict
    print("\nProcessing REAL images...")
    real_scores = extract_and_predict(
        real_paths, pipeline, model, sc_clip, sc_dino, sc_forensic
    )

    print("\nProcessing FAKE images...")
    fake_scores = extract_and_predict(
        fake_paths, pipeline, model, sc_clip, sc_dino, sc_forensic
    )

    # Build labels: real=0, fake=1
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([
        np.zeros(len(real_scores)),
        np.ones(len(fake_scores))
    ])

    # Compute metrics using trained tau
    metrics = compute_metrics(scores, labels, tau)

    print("\n" + "=" * 50)
    print("ZERO-SHOT RESULTS ON KAGGLE 140K")
    print(f"  Model trained on : CIFAKE")
    print(f"  Evaluated on     : Kaggle Real vs Fake Faces")
    print(f"  Real images      : {len(real_scores)}")
    print(f"  Fake images      : {len(fake_scores)}")
    print(f"  Threshold τ      : {tau:.4f} (from CIFAKE calibration)")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v}")
    print("=" * 50)

    # Save results
    results = {
        "dataset"       : "kaggle_140k_real_vs_fake",
        "model"         : model_path,
        "trained_on"    : "CIFAKE",
        "n_real"        : len(real_scores),
        "n_fake"        : len(fake_scores),
        "tau"           : float(tau),
        "alpha"         : float(alpha),
        "metrics"       : metrics,
        "real_scores"   : real_scores.tolist(),
        "fake_scores"   : fake_scores.tolist(),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="outputs/fusion_detector_v3.pt")
    parser.add_argument("--real",  type=str, required=True)
    parser.add_argument("--fake",  type=str, required=True)
    parser.add_argument("--out",   type=str,
                        default="outputs/kaggle_zeroshot.json")
    parser.add_argument("--max",   type=int, default=2000,
                        help="Max images per class (default 2000)")
    args = parser.parse_args()

    main(args.model, args.real, args.fake, args.out, args.max)