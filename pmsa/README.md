# PMSA — Probabilistic Media Synthesis & Authentication

⚡ A quant-grade AI project blending **deepfake detection** and **diffusion-latent watermarking**, with statistical rigor (Neyman–Pearson testing, GLRT thresholds) and robustness analysis.

---

## 🚀 Features
- **Detector (Neyman–Pearson surrogate):**
  - Forensic features: residual high-pass + block-DCT bands.
  - Trained with **α-controlled thresholds** to fix false alarm rates.
  - Metrics: ROC, AUC, EER, TPR@FPR=α.
  - Robustness sweeps under JPEG, noise, crop.

- **Watermark (Diffusion latent GLRT):**
  - Spread-spectrum code embedding into latent noise.
  - Generalized Likelihood Ratio Test (GLRT) for detection.
  - Analytically derived α-level thresholds.
  - Robustness sweeps under JPEG, noise, crop.

- **Synthetic structured dataset**:
  - “Real” = smooth Gaussian textures.
  - “Fake” = down/up sampling aliasing, block quantization, checkerboard artifacts.
  - Swap in **FaceForensics++** or **DFDC** for real data.

---

## 📂 Project Structure
    pmsa/
    README.md
    requirements.txt
    .gitignore
    pmsa/
    init.py
    data/
    datasets.py # toy dataset
    structured_dataset.py # synthetic dataset with artifacts
    models/
    detector.py # LRT surrogate
    features.py # residual + DCT feature extractors
    watermark.py # diffusion latent watermark
    training/
    train_detector.py # training loop (robustness augmentations)
    eval/
    eval_detector.py # ROC/AUC eval
    eval_attacks.py # detector robustness under attacks
    eval_watermark_attacks.py # watermark robustness under attacks
    utils/
    logging.py
    metrics.py
    attacks.py
    tests/
    test_watermark.py

