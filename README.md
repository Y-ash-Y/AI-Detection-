# PMSA v2 — Probabilistic Media Synthesis & Authentication

A **calibrated, generator-invariant** detector for synthetic images.

> **Thesis.** A detector's Neyman-Pearson false-alarm guarantee (FPR ≤ α) depends
> only on the *real* image distribution, so it is recoverable under shift via cheap
> per-domain recalibration on **unlabeled real images** — even when detection power
> (driven by the *fake* generator) is not. See [docs/research_statement.md](docs/research_statement.md).

This is v2. v1 (`pmsa_image_pipeline`) is archived as the baseline reference —
its zero-shot failure (FPR 1.0% → 14.4%, TPR@1% 0.939 → 0.008) is the motivating
result. See [reference/v1_baseline.md](reference/v1_baseline.md).

---

## Why this design

Frozen backbones + cached features. The GPU that holds the data (P100 on Kaggle)
does **one** extraction pass per dataset, writing `npz` caches. Every experiment
after that — baselines, fusion, calibration, the shift matrix — trains in minutes
on the laptop (MPS). Compute-shaped from the start.

```
images ──(P100, once)──► feature_cache/*.npz ──(M4, minutes)──► everything
        extract_backbone                         probe / fusion / calibrate / shift
```

## Three streams (all frozen)

| Stream    | Backbone (v2)        | Dim | Captures                         | vs v1 |
|-----------|----------------------|-----|----------------------------------|-------|
| Semantic  | CLIP ViT-L/14        | 768 | architecture-agnostic semantics  | was B/32 |
| Structural| DINOv2 ViT-B/14      | 768 | geometry / structure             | was S/14 |
| Artifact  | NPR (handcrafted)    | 44  | upsampling pixel-relations        | replaces DCT/PRNU |

NPR (Neighboring Pixel Relations) replaces v1's thumbnail-DCT/fake-PRNU forensic
stream, which barely transferred (0.06 detection alone, see v1 ablation).

## Layout

```
pmsa/
  backbones/    frozen extractors + registry (clip_l14, dino_b14, npr)
  features/     npz cache (FeatureSet) + the heavy extraction pass
  data/         Manifest (LOGO + three-way splits) + GenImage scanner
  models/       LinearProbe (UnivFD baseline) + FusionDetector (per-stream decomp)
  calibration/  NP thresholds (empirical + split-conformal) + Calibrator   ← contribution
  eval/         metrics w/ bootstrap CIs + the shift-experiment matrix      ← contribution
  utils/        seed, device
scripts/        01 extract → 02 baseline → 03 fusion → 04 score → 05 shift
configs/        default.yaml + experiments/
tests/          calibration + metrics + cache (run NOW, no data/GPU needed)
reference/      v1_baseline.md (the naive-approach rows)
docs/           research_statement.md (Phase 0 output)
```

## Quickstart

```bash
pip install -e ".[dev]"        # core + pytest
pytest -q                      # the contribution is tested without any data

# Heavy deps only where extraction runs (Kaggle/P100):
pip install -e ".[extract]"
```

## The phase map

- **P0 — read** (UnivFD, GenImage, AIDE/Chameleon, NPR, conformal). Output:
  [docs/research_statement.md](docs/research_statement.md).
- **P1 — extract** (run on Kaggle/P100):
  ```bash
  python scripts/01_extract_features.py \
      --genimage-root /kaggle/input/genimage \
      --generators stable_diffusion_v_1_4 sdxl midjourney \
      --per-class-limit 10000 --split train --tag train --device cuda
  ```
  Add real-domain-shift pools and in-the-wild fakes:
  `--real-dir ffhq:/path/ffhq chameleon:/path/cham_real`
  `--fake-dir nano_banana:/path/nb flux:/path/flux`
- **P2 — baseline** (the floor; LOGO, on the laptop):
  ```bash
  python scripts/02_train_baseline.py --backbone clip_l14 --tag train
  ```
- **P3 — fusion**:
  ```bash
  python scripts/03_train_fusion.py --tag train --held-out sdxl
  ```
- **P4 — calibrate + shift matrix** (the contribution):
  ```bash
  python scripts/04_score_and_calibrate.py --model fusion \
      --checkpoint outputs/fusion_logo_sdxl_seed0.pt --tag test --base-domain imagenet
  python scripts/05_shift_experiment.py --scores outputs/scores_fusion_test.npz \
      --base-domain imagenet --base-generator sdxl
  ```
- **P5 — robustness** (BPDA-corrected adversarial), in-the-wild eval, write-up.

## The table that doesn't exist yet

`05_shift_experiment.py` produces the 2×2 the literature is missing:

|              | generator fixed        | generator shifted       |
|--------------|------------------------|-------------------------|
| real fixed   | A in-dist              | B power drops, FPR holds |
| real shifted | C FPR breaks→recal     | D both, recal restores FPR |

If real-only recalibration restores FPR ≤ α even partially in C/D, that's the
finding — a workshop paper with a real result.

## Honesty notes (carried from v1's lessons)
- Every metric ships with a bootstrap CI and ≥3 seeds.
- The baseline (UnivFD) is step one; v2 must beat it or the complexity is unearned.
- Calibration and test reals are always disjoint.
- Expect 65–75% on Chameleon. That's the frontier, not failure.
```
