# Next step: fresh, matching evaluation on Kaggle

## Current status

The supplied smoke/full artifacts and later sampled serving-parity results have
been reviewed. The completed full comparison uses fresh metadata-bearing caches
and four heads across three seeds. See `MODEL_CARD.md` and
`evidence/fresh_full_summary.json`; it is previously viewed benchmark reproduction,
not a newly blinded evaluation or automatically promoted model.

The remainder retains the handoff/recovery instructions for reproducibility.
Archive names refer to those historical handoffs. Use a new run identity for a
rerun; a successful old archive does not mean the current working tree is the exact
old source snapshot or that the latest template rendering was inspected.

## Handoff template and original instructions

Use `notebooks/kaggle_fresh_evaluation.ipynb`. It is a new clean notebook, not a
replacement for the earlier exploratory notebook. Code cells use checked Python
subprocess calls rather than shell magic, so no missing `!python` prefix is needed.

The updated local source handoff is `outputs/kaggle_handoff_03/pmsa-kaggle-source.zip` with
its `.zip.sha256` sidecar. It contains an explicit source allowlist, not personal
interview books, images, old caches, Git history, virtual environments or old models.
It does not push anything to Git or update the live Space. Upload the archive as
a **private** Kaggle dataset; keep the filename unchanged. Replace an already uploaded
old package with this updated version. The earlier local archive is retained.

Kaggle may unpack the ZIP automatically. Set `SOURCE_INPUT` to the attached dataset
folder, its `source/` project folder, or the actual ZIP file: all three are supported.
The new package includes `source_manifest.json`. Setup checks every declared source
file hash and refuses changed source or existing-run overwrites. This is integrity
checking, not publisher authentication; use only your trusted project package.

The local 100 GPT-Image PNGs and 100 Nano Banana JPEGs passed raw decoding and
exact encoded/decoded identity checks. No exact or annotated related identity
overlapped between those collections. This does not validate their generator
origin, rule out near-duplicates, establish diversity, or establish detection
performance. Their private audit is `outputs/frontier_raw_preflight_01/audit/`.
They are fake-only and cannot supply false-alarm rates without a trusted real pool.

Original GenImage/FFHQ/COCO/StyleGAN2/Chameleon raw directories are unavailable on
this laptop. Their recovered embeddings cannot reconstruct extraction provenance.
The full notebook must run where the raw datasets are mounted. No GPU extraction
or backbone download was performed locally for this handoff.

## Run this sequence

1. Upload/open the new notebook in Kaggle. Attach the private source-zip dataset
   and the original five image datasets already used by this project.
2. In the parameter cell, update `SOURCE_INPUT` and any changed dataset mount paths.
   Leave `MODE="smoke"`, `FULL_RUN_CONFIRMED=False`, and use a new `RUN_ID`.
   The updated default is `fresh-smoke-02` in a separate source workspace.
3. Select a compatible GPU and run the notebook top-to-bottom. A real CUDA
   operation must pass before extracting. The pipeline uses one GPU; selecting
   T4×2 does not automatically make it multi-GPU.
4. Check raw domain/class/format counts, duplicate groups, aligned feature rows,
   tests and all four model outputs. Smoke uses tiny caps, one seed, two neural
   epochs and 30 bootstrap samples: its performance is **not** a benchmark result.
5. Download each stage archive and SHA256 sidecar. Save a Kaggle notebook version
   with outputs. An archive only inside temporary working storage is not durable.
6. Once smoke is green, start a **new run**: `MODE="full"`,
   `FULL_RUN_CONFIRMED=True`, new `RUN_ID`. Run top-to-bottom. Full caps match the
   40k-scale assembly rather than silently expanding compute or changing datasets.
7. Send back the full run archive, especially `evaluation/summary.json`,
   `evaluation/results.json`, and `raw_audit/audit.json`. Do not pick and upload only
   the best seed or manually pair checkpoints with another run's calibrator.

## Safeguards and scientific scope

Manifests are enumerated in sorted order and existing output names are refused.
Raw auditing checks all selected paths, decodes images, records byte and exact RGB
hashes, rejects cross-role identity/group overlap and mixed-label groups, and writes
curated group IDs. It does not detect every recompressed, resized, near-duplicate or
shared-prompt image. Review those risks separately; do not call exact auditing a
complete dataset clearance.

Development and external images share EXIF/RGB/JPEG-q90 normalization, the recorded
resolved SigLIP revision, processor mode and package versions. Strict fresh mode
rejects incomplete metadata and differing extraction settings before fitting. Raw
auditing and metadata improve traceability, not ground-truth authenticity.

The fixed plan compares semantic linear, forensic linear, semantic nonlinear and
fusion heads, all with common group-disjoint partitions and separate real-only
calibrators. Full mode reports seeds 0/1/2 with 2,000 bootstrap samples and target
FPR 5%. Every run has a distinct integrity-bound model bundle. No model is selected
or deployed automatically. Seed spread is not a population confidence interval.

Chameleon stays outside fitting and primary calibration, but this project already
viewed it. Re-extraction is a reproducible benchmark reproduction, not a newly
blinded holdout or proof of frontier generalization. Source/domain shortcuts remain:
COCO contributes real-only scenes and StyleGAN2 contributes synthetic faces.

Do not mix freshly extracted frontier vectors with legacy training vectors whose
processor/normalization provenance is missing. After the fresh run, a separate
frontier evaluation needs matching extraction, source records, an independent real
pool and any target-real calibration separated from test data. A genuine new
deployment-domain real pool cannot be manufactured from the benchmark test set.

## Session recovery

Stage archives contain `source/` plus selected run files under `artifacts/`. This
includes caches, manifests, metadata, config, hashes, bundles, splits and predictions;
raw images are not included. Keep the matching raw datasets attached if extraction
is still incomplete. Absolute data/cache paths in the saved config may need updating
after moving the restored files, without changing the extraction metadata.

The notebook intentionally refuses to overwrite a restored run. Inspect the archive
and run only missing stages with new output destinations rather than rerunning
write cells over saved results. Recovery is not an automatic environment match:
reproduce `environment.txt` or fresh-extract consistently. Do not erase old runs.

## Validation status

Offline tests exercise raw-image identities, cross-role and mixed-label failures,
sorted scans, fresh-cache rejection, a synthetic four-head smoke with complete
bundles/shared splits, archive exclusions, source manifests, ZIP/unpacked layouts,
tampering/traversal/symlink rejection, and notebook code-cell syntax. The actual
notebook source-setup cell executes locally against a synthetic unpacked input.
Synthetic tests do not measure image detection performance. The supplied completed
run artifacts were checked separately; the current template's exact top-to-bottom
execution and viewer presentation are not established by that check. A new rerun
requires the original raw mounts and compatible environment. Local
`nbformat`/`nbclient` are unavailable; inspect a newly executed notebook in Kaggle.

## Revision-recording regression and current smoke recovery

The observed Kaggle smoke passed its 102 tests, audit and all four extraction
passes (720 development images; 160 Chameleon images), then correctly rejected
SigLIP caches with `revision: null`. The extractor had depended on the private
Transformers `config._commit_hash` field. The local fix resolves an immutable Hub
snapshot before loading the model and processor, pins both to it and validates
metadata before image hashing/extraction. Offline regression tests cover missing
private commit fields, symlinked snapshot configs and wrong/unresolved revisions.

For the already-running smoke only, paste `notebooks/kaggle_resume_pinned_smoke.py`
into ONE new cell after the failed extraction. It uses the explicitly observed
cached commit `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`, re-extracts both roles into
a NEW cache directory, leaves the original caches and source untouched, records
the override in a recovery receipt, checks metadata/alignment, and archives the
run. It does NOT invent provenance for old vectors. The override is temporary
and local to the Python extraction calls; subsequent head fitting uses the new
recorded cache/config paths. Run the four-head comparison cell next, not the
original extraction cell. Do not restart the session before preserving outputs.

This recovery cell is tested offline with simulated extraction; its actual
Kaggle GPU execution is still required. For future runs use handoff_03 in a NEW
SOURCE_ROOT and RUN_ID (e.g. `pmsa_fresh_source_v3` and `fresh-smoke-03`) rather
than modifying the older verified workspace. Do not remove the metadata guard
or manually insert a cached `main` revision into old NPZ metadata: the presence
of a current cached config alone does not prove which weights/processor produced
previously unpinned vectors. Private materials remain excluded from archives.
