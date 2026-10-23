# AI-Generated Image Detector

PMSA is a still-image detection portfolio/research system: frozen vision features,
a custom forensic statistics stream, trained detector heads, and real-only
threshold calibration. It explores generalization and false alarms under domain
shift—not universal proof that an image is real or AI-generated.

[Live demo](https://huggingface.co/spaces/Xcalivier/ai-generated-detector) ·
[CI](https://github.com/Y-ash-Y/AI-Detection-/actions) ·
[Architecture](docs/ARCHITECTURE.md)

The minimal hosted release was verified on 2026-10-23 with a synthetic HTTP upload,
invalid-file rejection and released-file integrity checks. It serves the fixed
fresh-full fusion seed 0 reference, not an experimental adaptation candidate.
See the [deployment receipt](docs/evidence/deployment_summary.json).

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
Its interface pairs an upload preview and plain-language verdict with optional
signed score contributions and raw diagnostics in a collapsed Developer analysis panel.
A visual raw-score indicator shows the threshold—not a confidence percentage.
The [UI release receipt](docs/evidence/ui_release_summary.json) records verified
hosted checks; the model and calibrator are unchanged.
See [what the benchmark numbers mean](docs/BENCHMARK_GUIDE.md) for an honest
assessment and a prioritized improvement plan.
Metadata keyword scanning does not verify C2PA or SynthID and never overrides
the forensic score. Logos and other computer-made graphics are not automatically
generative AI; edited photographs may have mixed origins.

## Evidence status

Fresh metadata-bearing extraction, three-seed head evaluation and sampled
serving-parity checks have completed. Chameleon was previously viewed, so this is
reproducible benchmark evaluation, not a newly blinded holdout or a production
release. The new-image collection phase is deferred.

| Fresh full run: Chameleon, 26,017 representatives | Mean AUC | FPR at frozen threshold | TPR at frozen threshold |
|---|---:|---:|---:|
| SigLIP linear probe | 0.8697 | 6.11% | 45.92% |
| Forensic linear probe | 0.5969 | 5.14% | 3.01% |
| SigLIP nonlinear head | 0.9032 | 8.65% | 63.28% |
| SigLIP + forensic fusion | 0.9105 | 8.25% | 64.66% |

These are means across seeds 0/1/2, not confidence intervals or best-seed results.
Thresholds used separate broad calibration reals at target FPR 5%, not Chameleon
test scores. That target did **not** transfer. The [aggregate evidence snapshot](docs/evidence/fresh_full_summary.json)
records reviewed archive identity and scope; full per-image evidence stays private.
See [engineering status](docs/ENGINEERING_STATUS.md) for completed and open gates.
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

See the [model card](docs/MODEL_CARD.md) for exact reported intervals, limitations,
ownership, privacy and evidence status. No SOTA or production-reliability claim is made.

### Recovered-cache evaluation (exploratory)

Twelve local head fits (four ablations, three seeds) now use common independent
group splits. On 26,017 retained Chameleon feature representatives, mean AUC was
0.8584 for a SigLIP linear probe, 0.9092 for a SigLIP-only nonlinear head, and
0.9147 for fusion. Fusion's false-alarm rate was 7.91% at a frozen threshold
targeting 5%, so strong ranking did not establish threshold transfer. A separate
2,000-real target calibration supplement reduced false alarms toward the target
but cost detection power. See the model card for exact results and qualifications.

Missing legacy extraction provenance still prevents certifying image-to-app
equivalence. Chameleon was previously viewed, and these are not newly blinded
or production-release results. The complete local evidence is preserved under
`outputs/recovered_eval_v3_01/`; existing deployment files remain unchanged.

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

See [architecture and rationale](docs/ARCHITECTURE.md) for the pipeline, equations,
module contracts, calibration assumptions and design trade-offs.

## Reproduce or preview

The [reproduction guide](docs/REPRODUCIBILITY.md) covers:
legacy-cache exploratory experiments, fresh extraction on Kaggle, missing frontier
cache recovery, independent training/evaluation, and an explicit legacy app preview.

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
docs/             public model card and reproduction guide
configs/          experiment settings
reference/        historical v1 baseline
feature_cache/    local caches; ignored
outputs/          local models and results; ignored
```

Private interview books, image collections, caches and checkpoints stay out of Git.
The app does not train on or publish uploads; Gradio temporarily caches them.
Hosting/log retention is a separate concern—avoid sensitive uploads to public demos.

## Credits

Google SigLIP; GenImage, FFHQ, COCO, StyleGAN2; Chameleon/AIDE benchmark authors;
the original source authors of licensed frontier collections. This repository's
heads and evaluation pipeline are distinct from backbone pretraining and external
baseline weights. Respect dataset access restrictions and licenses.
