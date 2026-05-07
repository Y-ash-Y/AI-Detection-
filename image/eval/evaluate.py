"""
Full evaluation on a feature cache.

Usage:
    python -m image.eval.evaluate \\
        --features feature_cache/cifake_test.npz \\
        --model    outputs/fusion_detector.pt \\
        --out      outputs/eval_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

from image.models.fusion_model import FusionDetector
from image.models.detector_utils import compute_metrics


def load_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    model = FusionDetector(
        clip_dim     = ckpt["clip_dim"],
        dino_dim     = ckpt["dino_dim"],
        forensic_dim = ckpt["forensic_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["tau"], ckpt.get("alpha", 0.01)


def evaluate(features_path: str,
             model_path: str,
             output_path: str = None,
             plot: bool = True):

    # Load features
    d = np.load(features_path)
    clip     = torch.from_numpy(d["clip"]).float()
    dino     = torch.from_numpy(d["dino"]).float()
    forensic = torch.from_numpy(d["forensic"]).float()
    labels   = d["label"].astype(int)

    # Load model
    model, tau, alpha = load_model(model_path)

    # Score
    with torch.no_grad():
        scores = model(clip, dino, forensic).numpy()

    # Metrics
    metrics = compute_metrics(scores, labels, tau)

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS  (α={alpha})")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v}")
    print("=" * 50)

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved → {output_path}")

    # ROC plot
    if plot:
        import matplotlib.pyplot as plt

        fprs, tprs, _ = roc_curve(labels, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fprs, tprs, lw=2, label=f"AUC = {metrics['auc']:.4f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.axvline(x=metrics["fpr_at_tau"], color="r", linestyle=":",
                    label=f"Operating point (FPR≤{alpha})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("PMSA Image Detector — ROC Curve")
        plt.legend()
        plt.tight_layout()
        roc_path = (output_path or "outputs/eval").replace(".json", "_roc.png")
        plt.savefig(roc_path, dpi=150)
        print(f"ROC curve saved → {roc_path}")
        plt.close()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--model",    type=str, default="outputs/fusion_detector.pt")
    parser.add_argument("--out",      type=str, default="outputs/eval_results.json")
    parser.add_argument("--no-plot",  action="store_true")
    args = parser.parse_args()

    evaluate(args.features, args.model, args.out, plot=not args.no_plot)
