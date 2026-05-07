"""
Train the FusionDetector on cached features.

Uses BCEWithLogitsLoss (surrogate for LRT training).
Post-training: calibrate threshold τ on validation real samples
               to fix FPR ≤ alpha (Neyman-Pearson guarantee).

Usage:
    python -m image.training.train_fusion \\
        --features feature_cache/cifake_train.npz \\
        --val      feature_cache/cifake_test.npz  \\
        --out      outputs/fusion_detector.pt \\
        --alpha    0.01
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from image.models.fusion_model import FusionDetector
from image.models.detector_utils import calibrate_threshold, compute_metrics


def load_features(path: str):
    d = np.load(path)
    clip     = torch.from_numpy(d["clip"]).float()
    dino     = torch.from_numpy(d["dino"]).float()
    forensic = torch.from_numpy(d["forensic"]).float()
    label    = torch.from_numpy(d["label"]).float()
    return clip, dino, forensic, label


def augment_features(clip, dino, forensic, noise_std=0.05):
    """
    Feature-space augmentation simulating perturbations.

    Instead of re-running encoders on perturbed images (expensive),
    we approximate the effect by adding calibrated noise to each
    feature stream. This teaches the fusion model to be robust
    to perturbation-induced feature drift.

    - CLIP features: small perturbations (semantic embeddings are robust)
    - DINO features: medium perturbations (structural features more sensitive)
    - Forensic features: large perturbations (DCT/noise most sensitive)
    """
    clip_aug     = clip     + torch.randn_like(clip)     * noise_std * 0.5
    dino_aug     = dino     + torch.randn_like(dino)     * noise_std * 1.0
    forensic_aug = forensic + torch.randn_like(forensic) * noise_std * 3.0

    # L2 renormalize CLIP (was normalized at extraction)
    clip_aug = clip_aug / (clip_aug.norm(dim=-1, keepdim=True) + 1e-8)

    return clip_aug, dino_aug, forensic_aug


def train(
    features_path: str,
    val_path: str       = None,
    output_path: str    = "outputs/fusion_detector.pt",
    alpha: float        = 0.01,
    epochs: int         = 100,
    lr: float           = 1e-3,
    val_split: float    = 0.15,
    augment: bool       = True,
    aug_noise: float    = 0.05,
):
    # ── Load features ──────────────────────────────────────
    clip, dino, forensic, y = load_features(features_path)
    print(f"Loaded {len(y)} samples  |  real={int((y==0).sum())}  fake={int((y==1).sum())}")

    # ── Train / val split ──────────────────────────────────
    if val_path:
        clip_val, dino_val, forensic_val, y_val = load_features(val_path)
        clip_tr, dino_tr, forensic_tr, y_tr = clip, dino, forensic, y
    else:
        idx = np.arange(len(y))
        idx_tr, idx_val = train_test_split(
            idx, test_size=val_split, stratify=y.numpy(), random_state=42
        )
        clip_tr,  dino_tr,  forensic_tr,  y_tr  = clip[idx_tr],  dino[idx_tr],  forensic[idx_tr],  y[idx_tr]
        clip_val, dino_val, forensic_val, y_val = clip[idx_val], dino[idx_val], forensic[idx_val], y[idx_val]

    print(f"Train: {len(y_tr)}  |  Val: {len(y_val)}")
    if augment:
        print(f"Feature augmentation: ON  (noise_std={aug_noise})")

    # ── Model ──────────────────────────────────────────────
    model = FusionDetector(
        clip_dim     = clip_tr.shape[1],
        dino_dim     = dino_tr.shape[1],
        forensic_dim = forensic_tr.shape[1],
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.BCEWithLogitsLoss()

    print(f"\nTraining FusionDetector for {epochs} epochs...")
    print(f"Input dim: {model.clip_proj.in_features} + "
          f"{model.dino_proj.in_features} + "
          f"{model.forensic_proj.in_features}")
    print("-" * 50)

    best_auc   = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── Train step ──────────────────────────────
        model.train()

        if augment:
            # Mix clean + augmented in each batch
            c_aug, d_aug, f_aug = augment_features(
                clip_tr, dino_tr, forensic_tr, noise_std=aug_noise
            )
            # Concatenate clean + augmented samples
            c_in = torch.cat([clip_tr,     c_aug], dim=0)
            d_in = torch.cat([dino_tr,     d_aug], dim=0)
            f_in = torch.cat([forensic_tr, f_aug], dim=0)
            y_in = torch.cat([y_tr,        y_tr],  dim=0)
        else:
            c_in, d_in, f_in, y_in = clip_tr, dino_tr, forensic_tr, y_tr

        logits = model(c_in, d_in, f_in)
        loss   = loss_fn(logits, y_in)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── Val step ────────────────────────────────
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(clip_val, dino_val, forensic_val)
                val_loss   = loss_fn(val_logits, y_val).item()

            val_scores = val_logits.numpy()
            val_labels = y_val.numpy()

            tau = calibrate_threshold(
                torch.tensor(val_scores),
                torch.tensor(val_labels),
                alpha=alpha
            )
            metrics = compute_metrics(val_scores, val_labels, tau)

            print(f"Epoch {epoch:3d} | "
                  f"Train Loss={loss.item():.4f} | "
                  f"Val Loss={val_loss:.4f} | "
                  f"AUC={metrics['auc']:.4f} | "
                  f"TPR@1%FPR={metrics['tpr_at_1%']:.4f} | "
                  f"τ={tau:.4f}")

            if metrics["auc"] > best_auc:
                best_auc   = metrics["auc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_tau   = tau
                best_metrics = metrics

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    # ── Final calibration ─────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_scores = model(clip_val, dino_val, forensic_val).numpy()

    final_tau = calibrate_threshold(
        torch.tensor(final_scores),
        y_val,
        alpha=alpha
    )
    final_metrics = compute_metrics(final_scores, y_val.numpy(), final_tau)

    print("\n" + "=" * 50)
    print("FINAL METRICS (best checkpoint)")
    for k, v in final_metrics.items():
        print(f"  {k:<15}: {v}")
    print("=" * 50)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":    best_state,
        "tau":           final_tau,
        "alpha":         alpha,
        "metrics":       final_metrics,
        "clip_dim":      clip_tr.shape[1],
        "dino_dim":      dino_tr.shape[1],
        "forensic_dim":  forensic_tr.shape[1],
    }, output_path)

    print(f"\nSaved model → {output_path}")
    print(f"Threshold τ = {final_tau:.4f}  (FPR ≤ {alpha*100:.0f}%)")
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   type=str,   default="feature_cache/features.npz")
    parser.add_argument("--val",        type=str,   default=None)
    parser.add_argument("--out",        type=str,   default="outputs/fusion_detector.pt")
    parser.add_argument("--alpha",      type=float, default=0.01)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable feature augmentation")
    parser.add_argument("--aug-noise",  type=float, default=0.05,
                        help="Noise std for feature augmentation")
    args = parser.parse_args()

    train(
        features_path=args.features,
        val_path=args.val,
        output_path=args.out,
        alpha=args.alpha,
        epochs=args.epochs,
        lr=args.lr,
        augment=not args.no_augment,
        aug_noise=args.aug_noise,
    )
