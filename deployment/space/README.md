---
title: PMSA Image Forensics
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.18.0
python_version: '3.12'
app_file: space_app.py
pinned: false
license: mit
short_description: Custom SigLIP and forensic fusion image detector
models:
  - google/siglip-base-patch16-224
preload_from_hub:
  - google/siglip-base-patch16-224 config.json,preprocessor_config.json,model.safetensors 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed
---

# PMSA — image forensic detector

A portfolio/research demonstration of a custom trained detection head combining
Google's frozen SigLIP features with deterministic spectral, color and residual
statistics. Google pretrained the backbone; this project implemented the detector
heads, feature fusion, calibration and evaluation workflow.

This release uses the fixed fresh-full fusion seed 0 and its unchanged broad-real
calibrator. It is not a newly trained model or a promoted experimental candidate.
The versioned release manifest binds the serving files and identifies the source
bundle. The public split receipt is sanitized; private split identities are not
included.

## Interpret results cautiously

- Scores are not calibrated AI probabilities. No reliable confidence percentage
  is claimed.
- A detected signal is not proof of AI generation; absence of a signal does not
  prove camera origin. Real-photo false alarms and generated-image misses remain.
- The near-threshold uncertain label is heuristic, not a coverage guarantee.
- Calibration targets a reference real domain. Its false-alarm target does not
  automatically survive domain shift.
- Metadata keyword hints are not verified C2PA or SynthID evidence and never
  override the model score.
- Logos, ordinary edits, screenshots, generative edits and unfamiliar models may
  be misclassified. Video detection is not supported.

Do not use this demo to accuse people, authenticate legal evidence or make
automatic moderation decisions. No universal or state-of-the-art detection claim
is made.

## Privacy

The application does not train on or publish uploaded images. Gradio temporarily
caches uploads, with scheduled cleanup; hosting infrastructure may retain logs.
This is not a zero-retention guarantee. Do not upload sensitive personal images
to a public demo. No personal photos, feature caches or interview materials are
distributed in this repository.

## Source and credits

[Project source](https://github.com/Y-ash-Y/AI-Detection-).
Credits: Google SigLIP; GenImage, COCO, FFHQ, StyleGAN2; Chameleon/AIDE authors.
Dataset access and licenses remain separate from this serving-code release.
