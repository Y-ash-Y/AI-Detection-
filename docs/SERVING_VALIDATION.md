# Serving validation after the full evaluation

This is an engineering gate, not another model competition. Do not choose the
highest Chameleon seed, change calibrators, overwrite deploy artifacts or weaken
runtime checks to make the app run. The reviewed primary evaluation remains intact.

## Completed locally

All six safe tensor-only neural checkpoints (SigLIP MLP and fusion, seeds 0/1/2)
were integrity-checked, safely loaded and replayed on all 26,017 external cached
representatives. CPU replay matched the original saved GPU/CPU-run scores to the
recorded tolerance, with zero binary decision changes. Single-image scoring and
signed contribution sums were also checked on a deterministic sample including
threshold-adjacent rows. No joblib probe artifact was executed for this step.

These checks do not re-extract raw images or establish the app's camera-photo FPR.
Private receipts are under `outputs/serving_parity_01/`. Original Kaggle byte hashes
and source verification were reviewed separately; copies were staged in a new
ignored directory, not in the live application's deployment folder.

The original general-purpose laptop environment did NOT match the full run
(historical preflight below). A separate matching Python 3.12 environment is now
available under ignored outputs; the default app `.venv` is not automatically
changed. Do not interpret the older blocked receipt as the current isolated
environment's result, or skip guards in either environment.

| Dependency | Recorded extraction | Laptop at check |
| --- | --- | --- |
| NumPy | 2.1.3 | 2.4.6 |
| PyTorch | 2.11.0+cu128 | 2.12.0 |
| Torchvision | 0.26.0+cu128 | 0.27.0 |
| Transformers | 5.16.1 | 5.12.0 |
| Pillow | 12.3.0 | 12.2.0 |

CPU/CUDA wheel suffixes may differ under the existing compatibility rule, but the
core extractor versions must match. No installation/downgrade was attempted and
no runtime guard was bypassed. Preflight intentionally exits with status 2 here.

## Small Kaggle check: no training or full extraction rerun

1. Upload `outputs/kaggle_handoff_04/pmsa-kaggle-source.zip` as a NEW private source
   dataset and attach it to the completed full-run notebook. Keep the original
   image datasets attached. Do not replace the completed run's old source files.
2. Open `notebooks/kaggle_serving_parity.py` locally and paste its contents into
   ONE new code cell at the end of that notebook. Run only that new cell.
3. If the session restarted, first restore the already-saved full evaluation
   artifacts, keep the original image mounts and match the recorded environment.
   Update `PARITY_RUN` to the restored artifacts directory. Do not rerun training.
4. The cell processes only 16 fixed benchmark images across all six neural
   bundles and compares current serving features with original cached vectors.
   It fails if original image hashes, package versions or stream settings differ.
5. If all pass, download `pmsa-serving-parity-01.zip` AND `.zip.sha256` from Kaggle
   working storage and return them. Preserve failed receipts too; do not suppress
   failures or repeatedly tweak tolerances until they pass.

The companion cell completed on Kaggle and its downloaded `pmsa-serving-parity-02.zip`
and SHA256 sidecar were checked locally. All six bundles passed on the same 16
benchmark images, including comparison with original cached features. The source
snapshot matched the supplied parity package. This is not 96 independent images.
It checks Python inference, not the hosted Space's HTTP upload path. App-handler
tests use synthetic inputs and do not certify the live hosting environment.

## What the check compares

`scripts/15_check_serving_parity.py` verifies a trusted artifact bundle and checks
the runtime before any backbone is loaded. `pmsa/parity.py` then compares:

- Original file decoding against the app's in-memory image policy.
- Extraction-style batched preprocessing/encoding against single-image encoding.
- Optional original cache features, bound to the original raw-file SHA256 hashes.
- Model scores, threshold decisions, heuristic abstention labels and signed sums.
- Absence of an unfitted probability/confidence claim.

Numeric tolerances are explicit: feature absolute tolerance 1e-5, score absolute
tolerance 2e-4, relative tolerance 1e-4. A near-threshold decision flip still fails
even when its numerical difference is within tolerance. Failures require diagnosis,
not automatic tolerance expansion. Empty or nonfinite comparisons fail closed.

Receipts retain image byte hashes and numerical differences, not photo pixels or
personal filenames. Hashes are identities, not anonymization or proof of origin.

## Representative new real-photo confirmation: still needed

The user must confirm a dedicated pool's origin. Do not assume arbitrary Downloads
images are camera originals. Include different camera devices, subjects, lighting
and sharing/compression pipelines; keep originals/edits/crops in the same group.
Record source and processing history without unnecessary personal information.

First freeze model choice and sampling rules. Split a trusted real-photo pool into
group-disjoint calibration and test subsets BEFORE inspecting detector scores.
Use only calibration-real scores to fit a supplemental threshold. Keep the original
benchmark calibrator and results unchanged. Report frozen-threshold and adapted
threshold false alarms on the same held-out real photos, including numerator,
denominator and Wilson intervals. A real-only test cannot establish TPR or AUC;
add independent fake images for power measurements. Small personal-photo tests
are useful diagnostics but do not certify a deployment-wide 5% FPR guarantee.

Only after runtime/image parity and this separate operating evaluation should a
deployment candidate be proposed. These steps do not authorize a live deployment.

Small private camera/GPT development diagnostics were subsequently completed.
They still contain real-photo false alarms and are not a broad independent
deployment validation. Related images and repeatedly inspected examples cannot
be relabeled as new confirmation data. A targeted JPEG/resizing training pilot
was not promoted; further matched-scene data collection is currently deferred.
See `ENGINEERING_STATUS.md`; this document does not claim a new deployed model.

## Small private-photo diagnostic

`scripts/16_evaluate_real_photos.py` provides a two-stage, local-only workflow.
Prepare an immutable test plan before inspecting scores, then execute it in the
isolated environment defined by `configs/requirements-photo-eval.txt`. Keep that
environment and all personal receipts under ignored `outputs/`; never upload the
photos or publish their filenames, hashes, GPS or other metadata.

Preparation validates the six fixed tensor-only neural bundles, records image and
bundle identities, decodes the entire dedicated folder, and groups exact decoded
RGB duplicates. Difference-hash matches are only review hints: no-match does not
prove scene independence. The primary bundle is default `fusion_seed0`, fixed
before personal-photo scoring, with every other neural seed also reported. Do not
select the best personal-photo seed for deployment.

Execution checks frozen identities and extraction package versions, uses offline
Hub assets and CPU inference, and retains original calibrators. Every supplied
photo is test-only. The tool does not fit a threshold. Unreadable/multi-frame files
are explicit input failures, not successful real predictions and not part of the
readable-image FPR denominator. Report input coverage alongside false alarms.
Abstentions stay in the binary threshold denominator; report UI labels separately.

A small convenience pool cannot certify population FPR. Wilson intervals assume
independent photos, which exact hashes cannot establish. Different seeds can give
very different false alarms under camera/processing shift. A local macOS diagnostic
is not a hosted-upload test, and matching package versions alone does not establish
bitwise equivalence across OS, device and codec builds. The private receipts remain
under `outputs/real_photo_test_01/`; deployment is unchanged.
