# PMSA v2 — Research Statement (Phase 0 output, draft)

> Fill this in as you complete the Phase 0 reading. The skeleton below encodes the
> thesis; replace the TODOs with your defended positions.

## Problem
On in-the-wild frontier images, published SOTA synthetic-image detection sits at
~66–72% accuracy; many methods are at chance. The open problem is the gap between
in-distribution benchmarks and deployment under shift — not squeezing the last
point on a saturated benchmark.

## Thesis
A synthetic-image detector's Neyman-Pearson **false-alarm guarantee (FPR ≤ α)
depends only on the real-image distribution**, because the threshold τ is
calibrated on real scores alone. Therefore:

- Under **generator shift** (new fake model, same real domain): the FPR guarantee
  **holds** with the original τ; only detection power degrades.
- Under **real-domain shift** (new camera/scene distribution): the guarantee
  **breaks**, but it is **recoverable** by recalibrating τ on *unlabeled real
  images from the new domain* — no fakes required.

This reframes detection robustness: "the false-alarm guarantee is recoverable
under shift even when detection power is not." That separation is the contribution.

## Why it's winnable
The field's best public answer is barely above guessing on frontier fakes. The
win condition is not 99% — it is a rigorous, calibrated detector with an honest
generalization story and one real finding about guarantees under shift.

## Plan (see README for the phase map)
- P0 reading: UnivFD, GenImage, AIDE/Chameleon, NPR, conformal prediction primer.
- P1 data + cached frozen features (CLIP-L/14, DINOv2-B/14, NPR).
- P2 honest baseline: UnivFD linear probe, LOGO protocol, on Chameleon too.
- P3 PMSA v2 fusion (semantic + structural + NPR artifact) w/ stream decomposition.
- P4 conformal NP calibration + the shift matrix (the contribution).
- P5 robustness (BPDA-corrected), in-the-wild eval, write-up.

## Phase 0 reading notes (TODO)
- **UnivFD (Ojha 2023):** TODO — why frozen CLIP generalizes; what it misses.
- **GenImage:** TODO — protocol, LOGO, known leakage caveats.
- **AIDE / Chameleon:** TODO — the in-the-wild reality check; why methods collapse.
- **NPR (Tan 2024):** TODO — what upsampling artifact it captures; transfer story.
- **Conformal (Angelopoulos & Bates):** TODO — exchangeability assumption and
  whether real-domain recalibration data satisfies it.

## Expectation setting
v2 numbers on Chameleon will likely land 65–75% like everyone else's. That is the
frontier, not failure. The deliverable is calibration + an honest finding.
