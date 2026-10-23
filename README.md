# AI-Generated Image Detector

PMSA is a still-image detection portfolio/research system: frozen vision features,
a custom forensic statistics stream, trained detector heads, and real-only
threshold calibration. It explores generalization and false alarms under domain
shift—not universal proof that an image is real or AI-generated.

[Live demo](https://huggingface.co/spaces/Xcalivier/ai-generated-detector) ·
[CI](https://github.com/Y-ash-Y/AI-Detection-/actions) ·
[Architecture](#ownership-and-architecture)

The minimal hosted release was verified on 2026-10-23 with a synthetic HTTP upload,
invalid-file rejection and released-file integrity checks. It serves the fixed
fresh-full fusion seed 0 reference, not an experimental adaptation candidate.
The latest interface combines upload and preview in one panel, with optional
developer diagnostics. The UI release did not change the trained model.

## Ownership and architecture

Google's pretrained SigLIP supplies frozen semantic features; it was not pretrained
in this project. The project implements dataset assembly, a 66-dimensional
spectral/color/residual stream, trained fusion heads, score decomposition,
calibration and evaluation. An optional third-party classifier adapter is a
credited comparison baseline, never an automatic application fallback.

The inference path is:

1. Decode a bounded still image and apply the recorded EXIF/RGB/JPEG policy.
2. Extract the same features and processor settings used for training.
3. Standardize features with fit-only statistics; obtain a probe/fusion score.
4. Apply a separately fitted real-only threshold.
5. Show a cautious verdict, signed contributions and unverified metadata hints.

The app does **not** display sigmoid scores as calibrated AI probabilities.
Its single image box changes from a drop target into a preview after upload,
while preserving the original file for inference. A plain-language verdict has optional
signed score contributions and raw diagnostics in a collapsed Developer analysis panel.
A visual raw-score indicator shows the threshold—not a confidence percentage.
Hosted checks covered a synthetic image upload, invalid-file rejection and
score agreement with local inference—not new accuracy or privacy certification.
Metadata keyword scanning does not verify C2PA or SynthID and never overrides
the forensic score. Logos and other computer-made graphics are not automatically
generative AI; edited photographs may have mixed origins.

## Evidence status

Fresh metadata-bearing extraction, three-seed head evaluation and sampled
serving-parity checks have completed. Chameleon was previously viewed, so this is
reproducible benchmark evaluation, not a newly blinded holdout or a production
release. Category-matched data collection has restarted; no new training or
confirmation result is claimed from that preparation.

| Fresh full run: Chameleon, 26,017 representatives | Mean AUC | FPR at frozen threshold | TPR at frozen threshold |
|---|---:|---:|---:|
| SigLIP linear probe | 0.8697 | 6.11% | 45.92% |
| Forensic linear probe | 0.5969 | 5.14% | 3.01% |
| SigLIP nonlinear head | 0.9032 | 8.65% | 63.28% |
| SigLIP + forensic fusion | 0.9105 | 8.25% | 64.66% |

These are means across seeds 0/1/2, not confidence intervals or best-seed results.
Thresholds used separate broad calibration reals at target FPR 5%, not Chameleon
test scores. That target did **not** transfer. AUC measures ranking quality, not
accuracy or confidence for one image. At these frozen thresholds, fusion detected
about 65 of 100 generated images and falsely flagged about 8 of 100 real images
on this benchmark. These rates do not establish performance on current GPT images.
Full per-image evidence and personal collections are not included in this repository.
The hosted release was a separate explicit deployment; no model was retrained or
selected automatically from these results.

### Earlier results — separate experiments

The following are **historical notebook results**, not the fresh full run above.
Legacy recovered caches lack original image hashes and extraction metadata.

| Historical Chameleon result | SigLIP probe | Fusion |
|---|---:|---:|
| AUC | 0.8511 | 0.8986 |
| TPR at test-derived 1% FPR | 0.1430 | 0.2517 |
| TPR at test-derived 5% FPR | 0.3755 | 0.5013 |
| Realized FPR at deployed threshold | 0.0630 | 0.1020 |
| TPR at deployed threshold | 0.4153 | 0.6613 |

The deployed target was 5% FPR: fusion's observed **10.2%** means that target did
not transfer to Chameleon. AUC is a ranking metric, not accuracy or a per-image
probability. JPEG re-encoding does not eliminate acquisition/source bias.

Historical 100-fake frontier collections reported zero-shot TPR of 30.0% for
GPT-Image and 57.3% for Nano Banana at the experiment's 1% target. With 50 labeled
adaptation examples the reported means were 82.7% and 90.7%. These are small-sample
experiments; adapted models are not zero-shot, seed spread is not a confidence
interval, and the collections do not represent every current generator/version.

No SOTA or production-reliability claim is made. Small curated frontier
collections, previously inspected benchmarks and reused development examples
must not be treated as independent evidence of universal generator detection.

### Recovered-cache evaluation (exploratory)

Twelve local head fits (four ablations, three seeds) now use common independent
group splits. On 26,017 retained Chameleon feature representatives, mean AUC was
0.8584 for a SigLIP linear probe, 0.9092 for a SigLIP-only nonlinear head, and
0.9147 for fusion. Fusion's false-alarm rate was 7.91% at a frozen threshold
targeting 5%, so strong ranking did not establish threshold transfer. A separate
2,000-real target calibration supplement reduced false alarms toward the target
but cost detection power. Target-domain threshold adaptation is not a zero-shot
threshold result and does not fix poor score ordering.

Missing legacy extraction provenance still prevents certifying image-to-app
equivalence. Chameleon was previously viewed, and these are not newly blinded
or production-release results. These exploratory results are separate from the
fresh metadata-bearing evaluation and the fixed hosted model.

## Reliability safeguards

- Independent group-disjoint fit / validation / real calibration / test partitions.
- Exact-image hash and optional related-image grouping; mixed-label groups fail explicitly.
- Disjoint real groups in the updated leave-one-generator-out experiments.
- Recorded preprocessing, processor mode, backbone revision and extraction versions.
- Integrity-bound model/calibrator/split bundles; no silent checkpoint substitution.
- Safe weights-only loading for new fusion checkpoints; explicit trusted legacy opt-in.
- Pre-model upload decoding and pixel/frame limits; explicit artifact-schema,
  finite-weight and encoder/head compatibility errors.
- Separate ranking metrics and frozen-threshold FPR/TPR with intervals.
- External observed-overlap and stream/configuration checks.
- Offline regression tests and CPU CI configuration.

These are software safeguards, **not evidence of an accuracy increase**.
Near-duplicate auditing, prompt-family curation and domain representativeness still
need human review. Bundle hashes provide integrity, not publisher authentication.

## Quick start: tests

```bash
python -m pip install -e '.[dev,app]'
python -m pytest -q
python examples/offline_smoke.py
```

Tests require no datasets or pretrained-model downloads. The app extra includes
Gradio and the neural libraries to exercise all groups. For lightweight tests use
`.[dev]`; optional groups skip without their dependencies. The synthetic example
creates disposable noise images with arbitrary fixture labels: it exercises the
forensic image-to-verdict path, **not accuracy**, and leaves no usable model.
CI installs CPU Torch; local checks do not imply GitHub CI or the hosted demo ran.

The main implementation is in [inference](pmsa/inference.py),
[fusion heads](pmsa/models/fusion.py), [calibration](pmsa/calibration/conformal.py)
and [shared preprocessing](pmsa/preprocessing.py).

## Reproduce or preview

The source includes runnable entry points for each stage:

| Stage | Entry point |
|---|---|
| Fresh Kaggle evaluation | [Notebook](notebooks/kaggle_fresh_evaluation.ipynb) |
| Feature extraction | [01_extract_features.py](scripts/01_extract_features.py) |
| Independent head comparison | [10_evaluate_recovered.py](scripts/10_evaluate_recovered.py) |
| Raw-data audit | [13_audit_raw_data.py](scripts/13_audit_raw_data.py) |
| Serving-parity checks | [15_check_serving_parity.py](scripts/15_check_serving_parity.py) |
| Reviewed data preparation | [17_prepare_data.py](scripts/17_prepare_data.py) |
| Minimal Space release staging | [19_prepare_space.py](scripts/19_prepare_space.py) |

Run a script with `--help` for its required inputs. Datasets, caches and model
weights are not bundled with a fresh Git checkout. The fixed serving model files
are published separately in the [Space model directory](https://huggingface.co/spaces/Xcalivier/ai-generated-detector/tree/main/model).
Local inference requires a complete trusted bundle with matching model,
calibrator, split receipt and bundle metadata; do not mix files from different runs.

After creating a complete trusted v3 bundle:

```bash
python -m pip install -r requirements-app.txt
PMSA_BUNDLE=outputs/fusion_v3_01 python app.py
```

For an existing artifact, generic latest dependencies may not match its extractor.
Use its recorded versions; the isolated fresh-full reference is
`configs/requirements-photo-eval.txt`. A mismatch fails intentionally rather than
silently changing features. Installing source/running fixtures is separate from
running a particular trained bundle.

The default app expects a v3 bundle. Local retired v2 artifacts may be archived;
remaining legacy checkpoints require explicit legacy configuration because pairing
and preprocessing cannot be inferred reliably from filenames. No model training
or live redeployment is automatically triggered.

Default extraction now selects **SigLIP + forensic**. CLIP, DINOv2 and NPR source
modules remain available for historical comparisons without downloading their
weights on import. Use `--config configs/legacy_v2.yaml` to select the old streams
explicitly; old notebooks are historical records, not the current workflow.
Feature caches and experiment records are not disposable interpreter caches:
retain the current ones to avoid unnecessary extraction and preserve evidence.

## Layout

```text
pmsa/
  backbones/      frozen encoders and deterministic forensic features
  data/           manifests, grouping and disjoint splits
  features/       extraction metadata, hashes and aligned feature caches
  models/         trained linear probe and fusion heads
  calibration/    real-only empirical/conformal thresholds
  eval/           ranking and operating-point metrics
  training.py     independent four-part training protocol
  evaluation.py   external frozen-threshold evaluation and overlap checks
  artifacts.py    artifact pairing, checksums and runtime validation
  preprocessing.py shared image policy
  inference.py    cautious trusted-artifact prediction
  provenance.py   unverified metadata hints only
scripts/          extraction, training, evaluation and historical experiments
examples/         disposable offline synthetic pipeline smoke
tests/            offline regression and tiny synthetic integration tests
app.py            Gradio interface; temporary file handling and diagnostics
deployment/       minimal Hugging Face Space templates
configs/          experiment settings
reference/        historical v1 baseline
feature_cache/    local caches; ignored
outputs/          local models and results; ignored
```

The public project overview is maintained in this README. The `docs/` directory
contains local personal/reference materials, is Git-ignored, and is not required
to run the application or tests. Private images, study materials, feature caches
and local experiment artifacts are not part of the public repository.

## Credits

Google SigLIP; GenImage, FFHQ, COCO, StyleGAN2; Chameleon/AIDE benchmark authors;
the original source authors of licensed frontier collections. This repository's
heads and evaluation pipeline are distinct from backbone pretraining and external
baseline weights. Respect dataset access restrictions and licenses.
