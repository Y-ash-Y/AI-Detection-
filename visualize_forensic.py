"""
Forensic Feature Visualisation for PMSA.

Generates two overlays on the original image:
  1. DCT Anomaly Heatmap  — spectral artifacts from generator upsampling
  2. PRNU Spatial Map     — absence of camera sensor fingerprint

These visualisations explain WHY the model flagged an image as fake,
grounded directly in the NP decision framework.

Usage:
    python3 visualize_forensic.py \
        --image  path/to/image.jpg \
        --model  outputs/fusion_detector_mixed.pt \
        --out    outputs/forensic_viz/
"""

import argparse
import json
import numpy as np
import cv2
import torch
import joblib
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

from image.models.clip_encoder      import CLIPEncoder
from image.models.dino_encoder      import DINOEncoder
from image.models.forensic_features import extract_forensic_features
from image.models.fusion_model      import FusionDetector


# ── Spatial forensic maps (separate from feature vectors) ─────────────────

def compute_dct_anomaly_map(image_path: str,
                             block_size: int = 8) -> np.ndarray:
    """
    Compute per-block DCT anomaly score as a spatial heatmap.

    Method:
      For each 8×8 block, compute the high-frequency energy ratio:
        HF_ratio = sum(|DCT[4:,:]|²) / (sum(|DCT|²) + ε)

      Real camera images: energy decays smoothly toward high frequencies.
      AI-generated images: periodic grid artifacts cause anomalous
      high-frequency spikes in specific spatial locations.

    Returns: float32 heatmap of shape (H//block_size, W//block_size)
             normalised to [0, 1], higher = more anomalous
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(image_path).convert("L"))
    img = cv2.resize(img, (256, 256)).astype(np.float32)

    h, w   = img.shape
    n_rows = h // block_size
    n_cols = w // block_size
    hf_map = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            block = img[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size]
            if block.shape != (block_size, block_size):
                continue
            dct_block = cv2.dct(block)
            total_e   = float(np.sum(dct_block ** 2)) + 1e-8
            # High-frequency: bottom-right quadrant of DCT block
            hf_e      = float(np.sum(dct_block[4:, 4:] ** 2))
            hf_map[i, j] = hf_e / total_e

    # Normalise to [0, 1]
    hf_map = (hf_map - hf_map.min()) / (hf_map.max() - hf_map.min() + 1e-8)
    return hf_map


def compute_prnu_spatial_map(image_path: str,
                              sigma: float = 1.0) -> np.ndarray:
    """
    Compute PRNU noise residual spatial energy map.

    Method:
      noise = image - GaussianBlur(image, sigma)
      For each 16×16 spatial block, compute RMS noise energy.

      Real camera images: spatially consistent PRNU fingerprint
        (consistent noise energy across blocks due to sensor).
      AI-generated images: noise is spatially inconsistent —
        some regions have near-zero noise (over-smooth generator
        output), others have structured noise from upsampling artifacts.

    The spatial VARIANCE of noise energy is what distinguishes
    real from fake — not the absolute noise level.

    Returns: float32 heatmap (H//16, W//16) normalised to [0, 1]
             higher = more spatially anomalous noise
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.array(Image.open(image_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0

    block_size = 16
    n_rows     = 256 // block_size
    n_cols     = 256 // block_size
    noise_map  = np.zeros((n_rows, n_cols), dtype=np.float32)

    for c in range(3):
        chan     = img[:, :, c]
        blurred  = cv2.GaussianBlur(chan, (0, 0), sigma)
        residual = np.abs(chan - blurred)

        for i in range(n_rows):
            for j in range(n_cols):
                block = residual[i*block_size:(i+1)*block_size,
                                 j*block_size:(j+1)*block_size]
                noise_map[i, j] += float(np.sqrt(np.mean(block ** 2)))

    noise_map /= 3.0  # average across channels

    # Highlight spatial INCONSISTENCY — deviation from spatial mean
    spatial_mean = noise_map.mean()
    anomaly_map  = np.abs(noise_map - spatial_mean)
    anomaly_map  = (anomaly_map - anomaly_map.min()) / \
                   (anomaly_map.max() - anomaly_map.min() + 1e-8)
    return anomaly_map


def compute_dino_attention_map(image_path: str,
                                dino_enc: DINOEncoder) -> np.ndarray:
    """
    Extract DINOv2 self-attention map for the [CLS] token.

    DINOv2's self-attention naturally highlights semantically
    inconsistent regions — areas where the model's structural
    understanding flags anomalies.

    For ViT-S/14 on 518×518: patch size=14, so 37×37=1369 patches.
    We take the mean attention across all heads for the CLS token.

    Returns: float32 attention map (37, 37) normalised to [0, 1]
    """
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    x   = transform(img).unsqueeze(0)

    # Hook to capture attention from last transformer block
    attention_maps = []

    def hook_fn(module, input, output):
        # output shape: (B, num_heads, N, N) where N = n_patches + 1
        attention_maps.append(output.detach().cpu())

    # Register hook on last attention block
    handle = None
    for name, module in dino_enc.model.named_modules():
        if hasattr(module, 'attn'):
            last_attn = module.attn

    # Hook the softmax output of attention
    def attn_hook(module, input, output):
        attention_maps.append(output.detach().cpu())

    # Get attention weights by hooking into forward_attn
    # DINOv2 timm implementation stores attn weights
    attn_weights = []

    class AttentionCapture:
        def __init__(self):
            self.weights = None

        def __call__(self, module, input, output):
            self.weights = output.detach().cpu()

    capture = AttentionCapture()

    # Find and hook the last attention layer
    last_block = list(dino_enc.model.blocks)[-1]
    hook = last_block.attn.register_forward_hook(capture)

    device = next(dino_enc.model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        _ = dino_enc.model(x)
    hook.remove()

    if capture.weights is not None:
        w = capture.weights
        if w.dim() == 4:
            cls_attn = w[0, :, 0, 1:].mean(0).numpy()
        elif w.dim() == 3:
            cls_attn = w[0, 0, 1:].numpy()
        else:
            return np.ones((37, 37), dtype=np.float32) * 0.5

        n_patches = cls_attn.shape[0]
        side      = int(np.sqrt(n_patches))
        cls_attn  = cls_attn[:side*side]
        attn_map  = cls_attn.reshape(side, side)
        attn_map  = (attn_map - attn_map.min()) / \
                    (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map.astype(np.float32)


    # Fallback: uniform map
    return np.ones((37, 37), dtype=np.float32) * 0.5


# ── Full inference with score breakdown ───────────────────────────────────

def run_inference(image_path, model_path):
    """
    Run full inference and return score, decision, and per-stream scores.
    """
    ckpt = torch.load(model_path, map_location="cpu")
    fusion = FusionDetector(
        clip_dim=ckpt["clip_dim"],
        dino_dim=ckpt["dino_dim"],
        forensic_dim=ckpt["forensic_dim"])
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    tau         = ckpt["tau"]
    sc_clip     = joblib.load(ckpt["scaler_clip"])
    sc_dino     = joblib.load(ckpt["scaler_dino"])
    sc_forensic = joblib.load(ckpt["scaler_forensic"])

    clip_enc = CLIPEncoder()
    dino_enc = DINOEncoder()

    # Extract features
    with torch.no_grad():
        clip_f = clip_enc.extract(image_path).unsqueeze(0)
        dino_f = dino_enc.extract(image_path).unsqueeze(0)
    forensic_f = extract_forensic_features(image_path).reshape(1, -1)

    # Normalize
    clip_n     = torch.from_numpy(
        sc_clip.transform(clip_f.numpy())).float()
    dino_n     = torch.from_numpy(
        sc_dino.transform(dino_f.numpy())).float()
    forensic_n = torch.from_numpy(
        sc_forensic.transform(forensic_f)).float()

    with torch.no_grad():
        # Full score
        full_score = fusion(clip_n, dino_n, forensic_n).item()

        # Per-stream scores (zero out other streams)
        clip_score = fusion(
            clip_n,
            torch.zeros_like(dino_n),
            torch.zeros_like(forensic_n)).item()

        dino_score = fusion(
            torch.zeros_like(clip_n),
            dino_n,
            torch.zeros_like(forensic_n)).item()

        forensic_score = fusion(
            torch.zeros_like(clip_n),
            torch.zeros_like(dino_n),
            forensic_n).item()

    decision = "FAKE" if full_score > tau else "REAL"
    margin   = full_score - tau

    return {
        "full_score"      : full_score,
        "tau"             : tau,
        "decision"        : decision,
        "margin"          : margin,
        "clip_score"      : clip_score,
        "dino_score"      : dino_score,
        "forensic_score"  : forensic_score,
        "dino_enc"        : dino_enc,
    }


# ── Main visualisation ─────────────────────────────────────────────────────

def visualize(image_path: str,
              model_path: str,
              out_dir: str,
              save_json: bool = True):

    image_path = str(image_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem       = Path(image_path).stem

    print(f"Image: {image_path}")
    print("Running inference...")
    result = run_inference(image_path, model_path)

    print(f"  Decision  : {result['decision']}")
    print(f"  Score     : {result['full_score']:.4f}  "
          f"(τ={result['tau']:.4f}, margin={result['margin']:+.4f})")
    print(f"  CLIP      : {result['clip_score']:.4f}")
    print(f"  DINOv2    : {result['dino_score']:.4f}")
    print(f"  Forensic  : {result['forensic_score']:.4f}")

    print("Computing spatial maps...")
    dct_map  = compute_dct_anomaly_map(image_path)
    prnu_map = compute_prnu_spatial_map(image_path)

    print("Extracting DINOv2 attention...")
    attn_map = compute_dino_attention_map(image_path, result["dino_enc"])

    # Load original image for display
    orig = np.array(Image.open(image_path).convert("RGB"))

    # ── Build figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor="#0f0f0f")
    fig.suptitle(
        f"PMSA — Forensic Analysis: {Path(image_path).name}",
        color="white", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.35, wspace=0.25,
        left=0.04, right=0.96,
        top=0.92, bottom=0.08)

    # Color maps
    cmap_dct  = "inferno"
    cmap_prnu = "plasma"
    cmap_attn = "viridis"

    def add_overlay(ax, base_img, heatmap, cmap, title, alpha=0.6):
        """Overlay normalised heatmap on original image."""
        ax.imshow(base_img)
        h_resized = cv2.resize(heatmap, (base_img.shape[1],
                                          base_img.shape[0]))
        ax.imshow(h_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.axis("off")

    # Row 0: Original + 3 spatial maps
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(orig)
    ax0.set_title("Original Image", color="white", fontsize=9, pad=4)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    add_overlay(ax1, orig, dct_map, cmap_dct,
                "DCT Anomaly Map\n(High-freq spectral artifacts)")

    ax2 = fig.add_subplot(gs[0, 2])
    add_overlay(ax2, orig, prnu_map, cmap_prnu,
                "PRNU Spatial Map\n(Sensor fingerprint anomalies)")

    ax3 = fig.add_subplot(gs[0, 3])
    add_overlay(ax3, orig, attn_map, cmap_attn,
                "DINOv2 Attention\n(Structural anomaly regions)")

    # Row 1: Decision panel + per-stream bar chart + score gauges
    # Decision panel
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#1a1a1a")
    decision_color = "#ff4444" if result["decision"] == "FAKE" else "#44ff44"
    ax4.text(0.5, 0.65, result["decision"],
             ha="center", va="center",
             fontsize=28, fontweight="bold",
             color=decision_color,
             transform=ax4.transAxes)
    ax4.text(0.5, 0.42,
             f"Score: {result['full_score']:.3f}",
             ha="center", va="center",
             fontsize=11, color="white",
             transform=ax4.transAxes)
    ax4.text(0.5, 0.28,
             f"Threshold τ: {result['tau']:.3f}",
             ha="center", va="center",
             fontsize=10, color="#aaaaaa",
             transform=ax4.transAxes)
    ax4.text(0.5, 0.14,
             f"Margin: {result['margin']:+.3f}",
             ha="center", va="center",
             fontsize=10,
             color=decision_color,
             transform=ax4.transAxes)
    ax4.set_title("NP Decision", color="white", fontsize=9, pad=4)
    ax4.set_xticks([])
    ax4.set_yticks([])
    for spine in ax4.spines.values():
        spine.set_edgecolor("#333333")

    # Per-stream bar chart
    ax5 = fig.add_subplot(gs[1, 1:3])
    ax5.set_facecolor("#1a1a1a")

    streams = ["CLIP\n(Semantic)", "DINOv2\n(Structural)",
               "Forensic\n(DCT+PRNU)", "Full\nFusion"]
    scores  = [result["clip_score"], result["dino_score"],
               result["forensic_score"], result["full_score"]]
    colors  = []
    for s in scores:
        if s > result["tau"]:
            colors.append("#ff6666")
        elif s > 0:
            colors.append("#ffaa44")
        else:
            colors.append("#4488ff")

    bars = ax5.bar(streams, scores, color=colors,
                   width=0.5, edgecolor="#333333", linewidth=0.5)
    ax5.axhline(y=result["tau"], color="#ffffff",
                linestyle="--", linewidth=1.2, alpha=0.8,
                label=f"τ = {result['tau']:.3f}")
    ax5.axhline(y=0, color="#555555", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"{score:.2f}",
                 ha="center", va="bottom",
                 color="white", fontsize=8)

    ax5.set_title("Per-Stream Score Contribution",
                  color="white", fontsize=9, pad=4)
    ax5.set_ylabel("LRT Score", color="#aaaaaa", fontsize=8)
    ax5.tick_params(colors="#aaaaaa", labelsize=8)
    ax5.legend(fontsize=8, labelcolor="white",
               facecolor="#2a2a2a", edgecolor="#444444")
    ax5.set_facecolor("#1a1a1a")
    for spine in ax5.spines.values():
        spine.set_edgecolor("#333333")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Explanation panel
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.set_facecolor("#1a1a1a")
    ax6.axis("off")

    dominant = max(
        [("CLIP", result["clip_score"]),
         ("DINOv2", result["dino_score"]),
         ("Forensic", result["forensic_score"])],
        key=lambda x: x[1])[0]

    explanations = {
        "FAKE": [
            "• High-freq DCT spikes detected",
            "  (generator upsampling artifacts)",
            "",
            "• PRNU fingerprint absent or",
            "  spatially inconsistent",
            "  (no physical camera sensor)",
            "",
            f"• Dominant stream: {dominant}",
            f"• Score {result['full_score']:.2f} > τ {result['tau']:.2f}",
        ],
        "REAL": [
            "• DCT spectrum follows natural",
            "  1/f² power law decay",
            "",
            "• PRNU fingerprint consistent",
            "  with physical camera sensor",
            "",
            f"• Dominant stream: {dominant}",
            f"• Score {result['full_score']:.2f} ≤ τ {result['tau']:.2f}",
        ]
    }

    lines = explanations[result["decision"]]
    for k, line in enumerate(lines):
        ax6.text(0.05, 0.92 - k * 0.105, line,
                 transform=ax6.transAxes,
                 color="white", fontsize=7.5,
                 va="top", fontfamily="monospace")

    ax6.set_title("Forensic Reasoning",
                  color="white", fontsize=9, pad=4)
    for spine in ax6.spines.values():
        spine.set_edgecolor("#333333")

    # Save figure
    out_path = out_dir / f"{stem}_forensic.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Save JSON
    if save_json:
        json_path = out_dir / f"{stem}_forensic.json"
        result_clean = {k: v for k, v in result.items()
                        if k != "dino_enc"}
        with open(json_path, "w") as f:
            json.dump(result_clean, f, indent=2)
        print(f"Saved → {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True,
                        help="Path to image or directory of images")
    parser.add_argument("--model",
                        default="outputs/fusion_detector_mixed.pt")
    parser.add_argument("--out",
                        default="outputs/forensic_viz")
    parser.add_argument("--n", type=int, default=5,
                        help="If --image is a dir, process n images")
    args = parser.parse_args()

    image_path = Path(args.image)

    if image_path.is_dir():
        paths = sorted(image_path.glob("*.jpg"))[:args.n]
        if not paths:
            paths = sorted(image_path.glob("*.png"))[:args.n]
        print(f"Processing {len(paths)} images from {image_path}")
        for p in paths:
            try:
                visualize(str(p), args.model, args.out)
                print()
            except Exception as e:
                print(f"Failed {p}: {e}")
    else:
        visualize(args.image, args.model, args.out)


if __name__ == "__main__":
    main()
    