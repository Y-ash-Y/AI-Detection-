# v1 (pmsa_image_pipeline) — the naive-approach baseline row

Archived numbers from v1 (`~/Downloads/pmsa_image_pipeline`). These are the
"naive approach" reference rows for v2 tables — what we have to beat, and the
failure that motivates the v2 thesis. **Do not delete v1.**

## v1 architecture
- CLIP **ViT-B/32** (512-d) — v2 upgrades to **ViT-L/14** (768-d)
- DINOv2 **ViT-S/14** (384-d) — v2 upgrades to **ViT-B/14** (768-d)
- Forensic: DCT thumbnails + fake-PRNU noise (1024-d) — v2 replaces with **NPR**
- Fusion -> learned log-likelihood-ratio surrogate, NP threshold at FPR alpha=0.01

## In-distribution (trained on CIFAKE, tested on CIFAKE)
| metric        | value  |
|---------------|--------|
| AUC           | 0.9972 |
| TPR @ 1% FPR  | 0.939  |
| TPR @ 5% FPR  | 0.9894 |
| EER           | 0.0244 |
| accuracy      | 0.9646 |
| FPR @ tau     | 0.010  |

## Zero-shot (trained on CIFAKE, tested on Kaggle 140k real-vs-fake)
| metric        | value  |  vs in-dist        |
|---------------|--------|--------------------|
| AUC           | 0.5343 | 0.997  -> 0.534 (coin flip) |
| TPR @ 1% FPR  | 0.008  | 0.939  -> 0.008 (power collapse) |
| EER           | 0.475  |                    |
| accuracy      | 0.5082 |                    |
| **FPR @ tau** | **0.144** | **0.010 -> 0.144 (guarantee broken)** |

## The two tangled failures (the v2 spine)
1. **FPR blew up 1.0% -> 14.4%.** Cause = the *real* domain changed
   (CIFAR-10 objects -> FFHQ-ish faces). tau is a function of real images only,
   so this is recoverable by per-domain recalibration on real images — no fakes.
2. **Power collapsed (TPR@1% 0.939 -> 0.008).** Cause = the *fake* generator
   changed (CIFAKE/SD -> StyleGAN2). This needs generator-invariant features.

These are different failure modes with different fixes. v1 conflated them.
v2's contribution is to separate them and show the false-alarm guarantee is
recoverable under shift even when power is not.

## Robustness (v1, in-distribution) — degrades hard under perturbation
detection rate: none 0.936 / jpeg50 0.838 / jpeg20 0.586 / noise0.10 0.262 /
blur7 0.444 / crop20 0.576.

## Adversarial (v1) — fully broken under white-box attack
At eps=2/255: FGSM evasion 0.99, PGD evasion 1.00. (v2 must use BPDA-corrected
eval; the forensic stream's gradients made v1's numbers optimistic.)

## Stream ablation (v1, on a shifted set)
clip_only 0.58 / dino_only 0.62 / forensic_only 0.06 / **full 0.92** detection.
Takeaway carried into v2: the forensic stream alone barely transferred — hence
replacing DCT/PRNU with NPR. Fusion >> any single stream survives as a finding.
