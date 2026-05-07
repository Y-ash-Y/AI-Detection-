# PMSA — Probabilistic Media Synthesis & Authentication
### Image Detection Pipeline

---

## Mathematical Foundation

This system implements a **Neyman-Pearson optimal detector**:

```
H0: x is REAL    H1: x is FAKE

Optimal test:
    Λ(x) = p(x | fake) / p(x | real)

Decision:
    log Λ(x) > τ  →  FAKE
    log Λ(x) ≤ τ  →  REAL
```

The fusion model is a **learned surrogate** for `log Λ(x)`.  
The threshold `τ` is calibrated post-training to guarantee **FPR ≤ α** on real images.

---

## Feature Streams

| Stream   | Model              | Dim  | Captures                          |
|----------|--------------------|------|-----------------------------------|
| CLIP     | ViT-B/32           | 512  | Semantic, architecture-agnostic   |
| DINOv2   | ViT-S/14           | 384  | Structural, geometric             |
| Forensic | DCT + Noise        | 1024 | Spectral artifacts, sensor noise  |
| **Total**|                    | **1920** |                               |

---

## Repo Structure

```
pmsa/
├── image/
│   ├── models/
│   │   ├── device.py           # MPS/CUDA/CPU device selector
│   │   ├── clip_encoder.py     # CLIP ViT-B/32 feature extractor
│   │   ├── dino_encoder.py     # DINOv2 ViT-S/14 feature extractor
│   │   ├── forensic_features.py# DCT + noise residual features
│   │   ├── fusion_model.py     # LRT surrogate MLP (FusionDetector)
│   │   └── detector_utils.py   # NP calibration + metrics
│   ├── pipeline/
│   │   ├── feature_extraction.py  # Single-image feature pipeline
│   │   ├── build_features.py      # Cache features to .npz (run once)
│   │   └── inference.py           # Single-image prediction
│   ├── training/
│   │   └── train_fusion.py        # Train FusionDetector + calibrate τ
│   ├── eval/
│   │   ├── evaluate.py            # Full metrics + ROC curve
│   │   └── robustness.py          # Perturbation robustness suite
│   ├── data/
│   │   └── dataset_loader.py      # CIFAKE + generic loader
│   └── configs/
│       └── config.yaml
├── feature_cache/               # .npz files (git-ignored)
├── outputs/                     # Saved models + results
├── data/
│   ├── real/                    # Real images
│   └── fake/                    # AI-generated images
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Install

```bash
cd pmsa
pip install -e .
pip install -r requirements.txt
```

### 2. Get CIFAKE dataset

```python
# Option A: Kaggle CLI
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
unzip cifake-real-and-ai-generated-synthetic-images.zip -d data/cifake

# Option B: HuggingFace
from datasets import load_dataset
ds = load_dataset("aychang/cifake")
```

Expected structure after download:
```
data/cifake/
    train/
        REAL/   (50,000 images)
        FAKE/   (50,000 images)
    test/
        REAL/   (10,000 images)
        FAKE/   (10,000 images)
```

### 3. Build feature cache (run once, ~20-30 min on M4)

```bash
# Training set
python -m image.pipeline.build_features \
    --cifake data/cifake \
    --split  train \
    --out    feature_cache/cifake_train.npz

# Test set
python -m image.pipeline.build_features \
    --cifake data/cifake \
    --split  test \
    --out    feature_cache/cifake_test.npz
```

### 4. Train

```bash
python -m image.training.train_fusion \
    --features feature_cache/cifake_train.npz \
    --val      feature_cache/cifake_test.npz \
    --out      outputs/fusion_detector.pt \
    --alpha    0.01 \
    --epochs   30
```

Expected output:
```
Epoch  30 | Train Loss=0.0821 | Val Loss=0.1043 | AUC=0.9912 | TPR@1%FPR=0.9754 | τ=1.2341
```

### 5. Evaluate

```bash
python -m image.eval.evaluate \
    --features feature_cache/cifake_test.npz \
    --model    outputs/fusion_detector.pt \
    --out      outputs/eval_results.json
```

### 6. Robustness test

```bash
python -m image.eval.robustness \
    --image_dir data/cifake/test/FAKE \
    --model     outputs/fusion_detector.pt \
    --out       outputs/robustness.json
```

### 7. Single image inference

```bash
python -m image.pipeline.inference \
    --image path/to/image.jpg \
    --model outputs/fusion_detector.pt
```

Output:
```
========================================
Image  : path/to/image.jpg
Score  : 1.8432
τ      : 1.2341  (α=0.01)
Margin : +0.6091
Result : FAKE
========================================
```

---

## Metrics Reported

| Metric       | Description                                      |
|--------------|--------------------------------------------------|
| AUC          | Area under ROC curve                             |
| TPR @ FPR=1% | Detection rate at 1% false alarm (primary metric)|
| TPR @ FPR=5% | Detection rate at 5% false alarm                 |
| EER          | Equal Error Rate                                 |
| Accuracy     | Binary accuracy at threshold τ                   |
| FPR @ τ      | Actual false positive rate achieved              |

---

## Notes on Mac M4

- Device auto-selected: MPS > CUDA > CPU
- Feature caching is critical — run `build_features.py` once, then all
  training/eval runs in seconds
- If MPS causes issues with certain ops, set `PYTORCH_ENABLE_MPS_FALLBACK=1`

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m image.training.train_fusion ...
```
