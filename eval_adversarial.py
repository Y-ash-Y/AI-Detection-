"""
Adversarial robustness evaluation for PMSA fusion detector.
Manual FGSM and PGD implementation — no external attack library needed.

Research question:
  Since forensic features (DCT/PRNU) are non-differentiable and held
  fixed during attack, do gradient-based attacks on the semantic streams
  (CLIP/DINOv2) succeed in evading the full detector?

Usage:
    python3 eval_adversarial.py \
        --model outputs/fusion_detector_mixed.pt \
        --fake  data/kaggle_deepfake/real_vs_fake/real-vs-fake/test/fake \
        --out   outputs/adversarial_results.json \
        --n     100
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from image.models.clip_encoder       import CLIPEncoder
from image.models.dino_encoder       import DINOEncoder
from image.models.forensic_features  import extract_forensic_features
from image.models.fusion_model       import FusionDetector

DEVICE = torch.device("cpu")  # CPU only — stable gradients, no MPS issues

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0, 1]
])


# ── Differentiable scaler ──────────────────────────────────────────────────
class TorchScaler(nn.Module):
    def __init__(self, sklearn_scaler):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(
                sklearn_scaler.mean_, dtype=torch.float32))
        self.register_buffer(
            "std",  torch.tensor(
                sklearn_scaler.scale_, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)


# ── Differentiable encoder wrappers ────────────────────────────────────────
class CLIPEncoderCPU(nn.Module):
    def __init__(self, clip_enc):
        super().__init__()
        self.model     = clip_enc.model.to(DEVICE)
        self.device    = DEVICE

    def forward(self, x):
        """x: (B, 3, H, W) in [0,1] on CPU"""
        import torchvision.transforms.functional as TF
        if x.shape[-1] != 224:
            x = TF.resize(x, [224, 224])
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
        std  = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
        x = (x - mean) / std
        feat = self.model.encode_image(x.to(torch.float16)).to(torch.float32)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat


class DINOEncoderCPU(nn.Module):
    def __init__(self, dino_enc):
        super().__init__()
        self.model  = dino_enc.model.to(DEVICE)
        self.device = DEVICE

    def forward(self, x):
        """x: (B, 3, H, W) in [0,1] on CPU"""
        import torchvision.transforms.functional as TF
        if x.shape[-1] != 518:
            x = TF.resize(x, [518, 518])
        mean = torch.tensor(
            [0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor(
            [0.229, 0.224, 0.225]).view(1,3,1,1)
        x = (x - mean) / std
        return self.model(x)


# ── End-to-end differentiable detector ────────────────────────────────────
class E2EDetector(nn.Module):
    """
    Full differentiable pipeline for adversarial attack.

    Gradient path: pixel → CLIP/DINO → TorchScaler → FusionMLP → logit

    Forensic features held FIXED (non-differentiable numpy pipeline).
    This tests: can pixel-space attacks defeat a multi-stream detector
    when one stream (forensic) is gradient-obfuscated?
    """
    def __init__(self, clip_cpu, dino_cpu, fusion,
                 sc_clip, sc_dino, forensic_fixed):
        super().__init__()
        self.clip      = clip_cpu
        self.dino      = dino_cpu
        self.fusion    = fusion
        self.sc_clip   = TorchScaler(sc_clip)
        self.sc_dino   = TorchScaler(sc_dino)
        self.register_buffer("forensic", forensic_fixed)

    def forward(self, x):
        clip_f = self.clip(x)
        dino_f = self.dino(x)
        clip_s = self.sc_clip(clip_f)
        dino_s = self.sc_dino(dino_f)
        return self.fusion(clip_s, dino_s, self.forensic)


# ── Manual FGSM ────────────────────────────────────────────────────────────
def fgsm_attack(model, img, eps):
    """
    Fast Gradient Sign Method.
    Minimise logit (push fake score below threshold).
    """
    x = img.clone().detach().requires_grad_(True)
    logit = model(x)
    loss  = logit.mean()   # maximise → detected; we MINIMISE → evade
    loss.backward()
    with torch.no_grad():
        perturb = eps * x.grad.sign()
        adv = torch.clamp(x - perturb, 0.0, 1.0)
    return adv.detach()


# ── Manual PGD ─────────────────────────────────────────────────────────────
def pgd_attack(model, img, eps, alpha, steps):
    """
    Projected Gradient Descent (multi-step FGSM).
    """
    adv = img.clone().detach()
    adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
    adv = torch.clamp(adv, 0.0, 1.0)

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        logit = model(adv)
        loss  = logit.mean()
        loss.backward()
        with torch.no_grad():
            adv = adv - alpha * adv.grad.sign()
            delta = torch.clamp(adv - img, -eps, eps)
            adv   = torch.clamp(img + delta, 0.0, 1.0)

    return adv.detach()


# ── Main evaluation ────────────────────────────────────────────────────────
def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    fusion = FusionDetector(
        clip_dim     = ckpt["clip_dim"],
        dino_dim     = ckpt["dino_dim"],
        forensic_dim = ckpt["forensic_dim"],
    )
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()
    tau         = ckpt["tau"]
    sc_clip     = joblib.load(ckpt["scaler_clip"])
    sc_dino     = joblib.load(ckpt["scaler_dino"])
    sc_forensic = joblib.load(ckpt["scaler_forensic"])
    return fusion, tau, sc_clip, sc_dino, sc_forensic


def run(model_path, fake_dir, out_path, n_images,
        epsilons, pgd_steps):

    print("Loading model and encoders...")
    fusion, tau, sc_clip, sc_dino, sc_forensic = load_model(model_path)
    fusion = fusion.to(DEVICE).eval()

    raw_clip = CLIPEncoder(device=DEVICE)
    raw_dino = DINOEncoder(device=DEVICE)
    clip_cpu = CLIPEncoderCPU(raw_clip)
    dino_cpu = DINOEncoderCPU(raw_dino)

    fake_paths = sorted(Path(fake_dir).glob("*.jpg"))[:n_images]
    if not fake_paths:
        fake_paths = sorted(Path(fake_dir).glob("*.png"))[:n_images]
    print(f"Images: {len(fake_paths)}")

    # Pre-extract forensic features (non-differentiable, done once)
    print("\nExtracting forensic features...")
    forensic_list, valid = [], []
    for p in tqdm(fake_paths):
        try:
            forensic_list.append(extract_forensic_features(str(p)))
            valid.append(p)
        except Exception:
            pass

    forensic_np = sc_forensic.transform(np.stack(forensic_list))
    forensic_t  = torch.from_numpy(forensic_np).float()

    results = {"n_images": len(valid), "tau": float(tau)}

    # ── Baseline ────────────────────────────────────────────
    print("\nBaseline (clean)...")
    baseline_scores = []
    for i, p in enumerate(tqdm(valid)):
        img = TRANSFORM(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            e2e = E2EDetector(
                clip_cpu, dino_cpu, fusion,
                sc_clip, sc_dino, forensic_t[i:i+1])
            score = e2e(img).item()
        baseline_scores.append(score)

    baseline_dr = float(np.mean(
        [s > tau for s in baseline_scores]))
    results["baseline_detection_rate"] = baseline_dr
    results["baseline_mean_score"]     = float(np.mean(baseline_scores))
    print(f"  Detection rate : {baseline_dr:.4f}")
    print(f"  Mean score     : {np.mean(baseline_scores):.4f}  "
          f"(tau={tau:.4f})")

    # ── Attack loop ─────────────────────────────────────────
    for eps in epsilons:
        key = f"eps_{int(eps*255):03d}"
        results[key] = {}

        for atk_name in ["FGSM", "PGD"]:
            print(f"\n{atk_name} ε={eps*255:.0f}/255...")
            det, adv_scores = [], []

            for i, p in enumerate(tqdm(valid)):
                img = TRANSFORM(
                    Image.open(p).convert("RGB")).unsqueeze(0)

                e2e = E2EDetector(
                    clip_cpu, dino_cpu, fusion,
                    sc_clip, sc_dino,
                    forensic_t[i:i+1]).eval()

                try:
                    if atk_name == "FGSM":
                        adv = fgsm_attack(e2e, img, eps)
                    else:
                        adv = pgd_attack(
                            e2e, img, eps,
                            alpha=eps/4, steps=pgd_steps)

                    with torch.no_grad():
                        adv_score = e2e(adv).item()
                    adv_scores.append(adv_score)
                    det.append(adv_score > tau)
                except Exception as ex:
                    adv_scores.append(baseline_scores[i])
                    det.append(baseline_scores[i] > tau)

            dr = float(np.mean(det))
            results[key][atk_name] = {
                "detection_rate"   : dr,
                "evasion_rate"     : 1.0 - dr,
                "mean_adv_score"   : float(np.mean(adv_scores)),
                "epsilon"          : float(eps),
                "epsilon_over_255" : float(eps * 255),
            }
            print(f"  Detection rate : {dr:.4f}  "
                  f"(Evasion: {1-dr:.4f})")
            print(f"  Mean adv score : {np.mean(adv_scores):.4f}")

    # ── Save + print ─────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    print("\n" + "=" * 58)
    print("ADVERSARIAL ROBUSTNESS SUMMARY")
    print(f"  Baseline detection rate : {baseline_dr:.4f}")
    print(f"  Threshold τ             : {tau:.4f}")
    print("-" * 58)
    print(f"{'Attack':<8} {'ε/255':<8} {'Detection':<12} "
          f"{'Evasion':<10} {'Avg Score':<10}")
    print("-" * 58)
    for k, v in results.items():
        if not k.startswith("eps_"):
            continue
        for aname, m in v.items():
            print(f"{aname:<8} {m['epsilon_over_255']:<8.0f} "
                  f"{m['detection_rate']:<12.4f} "
                  f"{m['evasion_rate']:<10.4f} "
                  f"{m['mean_adv_score']:<10.4f}")
    print("=" * 58)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="outputs/fusion_detector_mixed.pt")
    p.add_argument("--fake",       required=True)
    p.add_argument("--out",        default="outputs/adversarial_results.json")
    p.add_argument("--n",          type=int,   default=100)
    p.add_argument("--epsilons",   nargs="+",  type=float,
                   default=[2/255, 4/255, 8/255])
    p.add_argument("--pgd-steps",  type=int,   default=10)
    args = p.parse_args()

    run(args.model, args.fake, args.out, args.n,
        args.epsilons, args.pgd_steps)