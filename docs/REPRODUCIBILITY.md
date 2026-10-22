# Reproduction and recovery workflow

## What this reliability pass does not do

It does not replace historical checkpoints, recover missing evaluation caches,
retrain SigLIP, or update the live Space. Private interview materials remain ignored.
Do not erase originals or reuse output names while investigating a run.

## Offline checks

```bash
python -m pip install -e '.[dev,app]'
python -m pytest -q
python examples/offline_smoke.py
```

The app extra includes the neural libraries and Gradio. Lightweight `.[dev]` skips
optional groups; CI installs CPU Torch to exercise them. Tests use synthetic
fixtures and temporary files, not benchmark images. The offline example extracts
forensic features from disposable generated noise, fits a tiny fixture head,
calibrates on disjoint class-zero fixture rows, restores an integrity-bound bundle
and checks a held-out image-to-verdict path. Every picture is synthetic; those
labels are test classes, not real/AI ground truth. No foundation weights are
downloaded and no usable model is retained. It is not an accuracy demonstration.
CI configuration is provided; a passing local test run does not prove that GitHub CI
has run. The Gradio app additionally needs `requirements-app.txt`.

## Recovery: old caches remain useful, but exploratory

Recovered `siglip_diversetrain.npz`, `forensic_diversetrain.npz`, and the matching
Chameleon caches can be used for inexpensive head experiments. Missing original
image hashes and extraction configuration cannot be reconstructed from embeddings.
Do not invent them or mark old caches as v3. Put copies in your local cache directory;
retain the originals and keep all caches outside Git.

An optional single CPU probe experiment:

```bash
python scripts/train_deploy.py --tag diversetrain --model probe --backbone siglip \
  --alpha 0.05 --device cpu --exploratory-legacy --out outputs/exploratory_probe_01
python scripts/eval_testset.py --deploy-dir outputs/exploratory_probe_01 \
  --model probe --backbone siglip --test-tag chameleontest --exploratory-legacy \
  --out outputs/exploratory_probe_chameleon_01
```

This creates an explicit `NOT_DEPLOYABLE.txt` marker. Results remain exploratory;
feature-identity grouping does not establish image-level uniqueness or correct
preprocessing. Probe joblib files must be trusted.

## Completed recovered-cache comparison

The recovered four NPZs in Downloads were evaluated without copying them into the
repository or replacing deployment artifacts. `scripts/10_evaluate_recovered.py`
records the experiment plan, audits identities and runs shared-split three-seed
ablations; `scripts/11_validate_evaluation.py` reconciles manifests, operating
counts and paired bootstrap comparisons; `scripts/12_recalibrate_evaluation.py`
performs separate target-real calibration without training a head or using fake
test labels for calibration.

The initial nine fits compare semantic linear, forensic linear and fusion heads.
Three supplemental SigLIP-only nonlinear fits control for nonlinear capacity.
All twelve runs are reported, not the best seed. The complete local evidence is
`outputs/recovered_eval_v3_01/REPORT.md`, including commands for a repeat run with
new output paths. Feature metadata remains missing, so these outputs carry
`NOT_DEPLOYABLE.txt` and do not certify an image-to-app release. The raw frontier
images were not re-extracted against the unknown legacy training configuration.

## Fresh extraction (separate, optional compute phase)

The supplied fresh full-run archive and later serving-parity archive have already
been reviewed. The commands below reproduce that workflow for a **new** run; they
are not instructions to rerun completed training just to use the software.
Public aggregate scope is in `docs/evidence/fresh_full_summary.json`; private
per-image records remain under ignored outputs. See `ENGINEERING_STATUS.md`.

For a complete smoke-to-full Kaggle handoff, use
[`FRESH_EVALUATION.md`](FRESH_EVALUATION.md) and
`notebooks/kaggle_fresh_evaluation.ipynb`. The source archive excludes personal
materials; raw auditing and strict fresh-cache evaluation precede any promotion.

Install `.[extract]` in the environment where raw images are available. Existing
manifests point to Kaggle input paths; do not assume they resolve on a laptop.
Keep Chameleon external/test-only. Use a new tag for every extraction; processing
and model revision settings are recorded. Default processors are explicitly slow
(`use_fast=False`) for reproducibility, so new features may differ from old ones.

```bash
python scripts/01_extract_features.py --manifest feature_cache/manifest_diverse_train.csv \
  --tag diversetrain_v3 --backbones siglip forensic --normalize-jpeg \
  --device cuda --batch-size 32 --num-workers 2
python scripts/01_extract_features.py --manifest feature_cache/manifest_chameleon_test.csv \
  --tag chameleontest_v3 --backbones siglip forensic --normalize-jpeg \
  --device cuda --batch-size 32 --num-workers 2
```

In Kaggle notebook cells prepend `!` to each command. Execute from the repository
directory after installation. For local extraction choose `--device cpu` and
`--num-workers 0` (Mac worker spawning must be tested before increasing workers).
No GPU model training is required to cache features from a frozen backbone.

The 100 GPT and 100 Nano Banana raw images in Downloads can restore the four
missing caches. Example, **not automatically executed**:

```bash
python scripts/01_extract_features.py \
  --fake-dir 'gptimage:/path/to/frontier/chatgpt_images' \
  --tag gptimage_v3 --backbones siglip forensic --normalize-jpeg \
  --device cpu --batch-size 16 --num-workers 0
python scripts/01_extract_features.py \
  --fake-dir 'nanobanana:/path/to/frontier/nano_banana' \
  --tag nanobanana_v3 --backbones siglip forensic --normalize-jpeg \
  --device cpu --batch-size 16 --num-workers 0
```

Fresh frontier caches alone do not validate the legacy training/extraction pipeline.
To compare with a v3 model, matching training and test extractor configurations
are required. Fake-only sets estimate detection rate, not AUC or FPR. Collect an
independent, representative real pool for a two-class frontier evaluation.

## Train and evaluate a v3 bundle

For an existing fresh-full artifact, reproduce its recorded extractor environment
before invoking `Detector`. `configs/requirements-photo-eval.txt` is the isolated
Python 3.12 reference; install into a new dedicated environment, not over a working
environment. Install this repository with `python -m pip install -e . --no-deps`
after installing those recorded dependencies, and install the app's Gradio extra
separately if the interface is needed. Compare `bundle.json` versions first;
generic latest dependencies for the source tests do not guarantee compatibility
with a historical bundle. Do not bypass the check or edit an old bundle's versions.

```bash
python scripts/train_deploy.py --tag diversetrain_v3 --model fusion \
  --backbones siglip forensic --alpha 0.05 --device cpu --out outputs/fusion_v3_01
python scripts/eval_testset.py --deploy-dir outputs/fusion_v3_01 \
  --test-tag chameleontest_v3 --out outputs/fusion_v3_chameleon_01
PMSA_BUNDLE=outputs/fusion_v3_01 python app.py
```

The group split is 60% fit / 15% validation / 12.5% calibration / remainder test.
Fractions refer to groups; random splits are not domain-stratified and lack of a
class fails explicitly. Training normalization uses fit rows only. Best fusion
epoch uses validation loss. Threshold uses calibration **reals only**, never test
reals. Ranking and frozen operating-point reports are separate. External evaluation
checks observed overlap against **all** development identities before deduplication.

Use distinct output directories for probe and fusion. Outputs contain configuration,
source hashes, cache hashes, split identities, predictions, environment, evaluation
and an integrity-bound bundle. Existing nonempty output directories are rejected.
Serving rejects extractor core package-version mismatch; reproduce the recorded
versions or re-extract rather than silently changing processor behavior. CPU/CUDA
wheel build suffixes on torch/torchvision are permitted to differ. Matching core
versions does not establish bitwise determinism across all hardware.

## Existing app checkpoints: explicit legacy preview only

You can retain the old checkpoint without pretending it has v3 audit records.
For your own trusted fusion checkpoint, after confirming the original extraction
used JPEG normalization, this opt-in runs a **legacy/unverified** preview:

```bash
PMSA_ALLOW_LEGACY=1 PMSA_TRUSTED_LEGACY=1 PMSA_LEGACY_NORMALIZE_JPEG=1 \
PMSA_CKPT=outputs/deploy/fusion.pt PMSA_CAL=outputs/deploy/calibrator.json python app.py
```

`PMSA_TRUSTED_LEGACY=1` enables unrestricted pickle for this old checkpoint. Never
use it on downloaded/untrusted models. JPEG policy must match the actual extraction;
set 0 only if JPEG normalization was absent. Original processor/runtime provenance
is still unknown, so this is not a validated v3 deployment. The safe default refuses
legacy artifacts instead of loading them or silently using a third-party model.

## Historical experiment scripts

Default extraction selects SigLIP + forensic. To rerun CLIP/DINO/NPR history,
pass `--config configs/legacy_v2.yaml` to extraction and fusion scripts (including
the historical extraction notebook retained in Git history). Encoder modules
remain import-compatible; old local feature/checkpoint files may have been moved
to a checksum-verified archive under `outputs/cleanup_legacy_01/`. Restore only
the desired archive members to their recorded relative paths when needed. That
archive is private and ignored by Git, not a new deployment bundle.

Scripts 04–09 are retained as research history, not certified release workflows.
Their old shift/few-shot outputs must not be presented as new audited results.
Any future few-shot revision should freeze a common untouched test pool across k,
keep related prompts/images together, use independent target real calibration,
and report seed spread separately from sampling uncertainty.

## What to save after a Kaggle run

Save a notebook version with outputs and archive the matching feature caches,
manifests, whole model bundle and evaluation directory. Download the archive and
inspect its file listing before ending the session. Store large artifacts privately
with a checksum; Git tracks source, configuration, tests and non-private docs only.
