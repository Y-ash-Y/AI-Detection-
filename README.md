# PMSA — AI Image Detector

PMSA is a computer vision project that flags signs of AI generation in still
images. The Gradio app shows a verdict, the score relative to its decision
threshold, and optional diagnostics for what contributed to the score.

[Live demo](https://huggingface.co/spaces/Xcalivier/ai-generated-detector) ·
[Evaluation notebook](notebooks/kaggle_fresh_evaluation.ipynb) ·
[CI](https://github.com/Y-ash-Y/AI-Detection-/actions)

## What I built

- **Two-stream detector.** Frozen Google SigLIP embeddings capture image content.
  A custom 66-feature stream measures color, frequency, and noise residuals.
  A trained fusion head combines the two and reports each stream's contribution.
- **Calibration and evaluation.** A separate set of real images sets the
  decision threshold. Group-aware splits keep known related images together,
  and evaluation reports both detection and false alarms at the frozen threshold.
- **Deployment.** A Gradio app on Hugging Face Spaces with upload validation,
  integrity checks for model artifacts, and offline regression tests.

## Benchmark

On **26,017 Chameleon image representatives**, fusion averaged **0.911 ROC AUC**,
**64.7% detection**, and **8.3% false alarms** across three seeds. The threshold
targeted 5% false alarms on a separate real calibration set; that target did
not transfer to Chameleon. These results are benchmark-specific. The app score
is not a calibrated probability or proof of how an image was made. Chameleon
was previously inspected; see [evaluation details](reference/EVALUATION.md) for
the full comparison, historical experiments and limits. Metadata hints are not
verified C2PA or SynthID evidence.

## Run locally

From the repository root in a Python 3.11 or 3.12 environment, check the
pipeline without datasets or pretrained weights:

```bash
python -m pip install -e '.[dev,app]'
python -m pytest -q
python examples/offline_smoke.py
```

For lightweight checks, use `.[dev]`; the optional app/neural tests may skip.
The [offline example](examples/README.md) uses synthetic fixtures and does not
measure detector accuracy.

For local inference, download the complete
[Space model bundle](https://huggingface.co/spaces/Xcalivier/ai-generated-detector/tree/main/model),
install the [Space runtime dependencies](deployment/space/requirements.txt) in a
compatible environment, and set `PMSA_BUNDLE` to that directory before running
`python app.py`. The [deployment guide](deployment/README.md) explains bundle
pairing, local dependency options and release staging.

## Repository map

| Where | What to find there |
| --- | --- |
| [`pmsa/`](pmsa/README.md) | Detector, data and feature contracts, heads, calibration and evaluation |
| [`scripts/`](scripts/README.md) | Data prep, extraction, training, evaluation and release commands |
| [`notebooks/`](notebooks/README.md) | Kaggle evaluation and targeted recovery/parity workflows |
| [`configs/`](configs/README.md) | Default, experiment and pinned evaluation settings |
| [`examples/`](examples/README.md) | Offline synthetic example |
| [`tests/`](tests/README.md) | Regression coverage and CI instructions |
| [`deployment/`](deployment/README.md) | Local serving and minimal Space staging |
| [`reference/`](reference/README.md) | Benchmark context and historical baselines |
| [`feature_cache/`](feature_cache/README.md), [`outputs/`](outputs/README.md) | Local workspace conventions; contents ignored |

For a research run, start with the [workflow guide](scripts/README.md) or the
[fresh Kaggle notebook](notebooks/kaggle_fresh_evaluation.ipynb). Generic
dependencies are defined in [`pyproject.toml`](pyproject.toml); the root
requirements files are install shortcuts. Pinned evaluation and standalone
serving requirements are intentionally separate.

Local `docs/`, photos, feature caches, experiment results and model artifacts
are not required for the source checkout. Do not treat feature caches as
disposable interpreter caches or publish private paths, splits or predictions.

[Inference](pmsa/inference.py) · [Training](pmsa/training.py) ·
[Calibration](pmsa/calibration/conformal.py) · [MIT License](LICENSE)

Google pretrained SigLIP. Training data includes GenImage, FFHQ, StyleGAN2
images, and COCO; Chameleon is used for evaluation. Dataset licenses are separate
from the code license.
