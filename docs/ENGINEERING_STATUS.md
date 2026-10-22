# Engineering readiness — 2026-10-23

This is a source/test hardening pass, **not a new trained model or deployment**.
The matched-scene collection experiment is deferred at the user's request.

## What is in place

- Shared extraction/inference policy, pinned semantic backbone and processor
  configuration in metadata-bearing caches.
- Trusted model/calibrator/split bundles with integrity and runtime guards.
- Early rejection of unreadable, truncated, excessive-pixel and multi-frame
  uploads before loading weights.
- Explicit malformed-bundle, missing-file, unsafe-path, invalid-digest and
  nonfinite-checkpoint failures, plus encoder/head dimension checks.
- No automatic external detector fallback or unfitted confidence percentages.
- A synthetic image-to-verdict example, restore/decomposition tests and CPU CI.
- Public aggregate evidence separated from private photos, per-image results,
  caches, checkpoints and personal interview materials.

Run `python -m pytest -q` after installing `.[dev,app]`. Tests exercise software
contracts, not detection accuracy. The local full optional-dependency run passed
213 tests (including minimal Space staging and card validation). The
matching-extractor Python 3.12 run passed 200 tests, with two
optional skips and four existing scikit-learn/SciPy solver-option warnings.
The full Gradio-enabled environment exercised the app tests as part of its
full test pass.
The original hardening pass did not deploy. A subsequent explicitly authorized
release is now verified: GitHub CI passed, the Space is running, all 32 released
file hashes match, and the synthetic hosted HTTP upload/invalid-file checks pass.

Deployment is now a separate, explicitly authorized release step. A fresh matching
Python 3.12 environment passed a synthetic HTTP upload and invalid-file rejection.
The Space SDK's automatic Gradio MCP/OAuth extras require Pydantic 2.12.5; the
serving template pins that compatible set while retaining extractor versions.
Hosted verification is recorded in `evidence/deployment_summary.json`; local
success alone was not used as a hosted-deployment claim. See `DEPLOYMENT.md`.

All six retained fresh-full neural checkpoints were also safely loaded and
replayed on the same 26,017 cached external representatives. Every replay passed,
with zero binary decision changes and matching signed score decomposition on
the sampled rows. These are six model passes over the same images, not six new
independent test sets. No raw personal photos were needed. The private receipt
is `outputs/engineering_audit_20261022/cached_replay.json`; original model,
calibrator and prediction-file hashes stayed unchanged.

## Evidence status — do not mix generations

| Stage | Status | Interpretation |
|---|---|---|
| Original notebook results | Historical | Missing legacy extraction provenance |
| Recovered-cache ablations | Completed exploratory work | Not original-image provenance recovery |
| Fresh metadata-bearing full run | Completed and archive reviewed | Four heads × three seeds; previously viewed benchmark |
| Cached head and sampled raw-image parity | Completed checks | Consistency, not accuracy certification |
| Private camera/GPT development checks | Completed diagnostics | Real-photo false alarms remain |
| Training-only JPEG/resizing pilot | Completed, not promoted | No targeted-error improvement |
| Matched extension and new confirmation | Prepared, deferred | No new training or improvement claim |
| Hosted HTTP upload path | Synthetic upload and rejection verified | Not accuracy, privacy or load certification |

The [fresh-full aggregate evidence](evidence/fresh_full_summary.json) records
fusion mean Chameleon AUC 0.9105, frozen-threshold mean FPR 8.25% and TPR 64.66%
across three seeds. The 5% target did not transfer. This differs from both the
earlier 0.8986 result and later private development candidates.

## Open release decisions

1. Specify a trusted bundle, intended domain and acceptable error trade-off.
   Never promote the best-looking diagnostic seed.
2. Match recorded extractor versions or re-extract into a new run; do not bypass
   a runtime guard merely to get an interface running.
3. Obtain new confirmation data before claiming a model improvement. This phase
   is deferred, not silently marked complete.
4. If deployment is requested later, test HTTP uploads, latency, memory, logging,
   temporary retention and abuse controls in that environment separately.

No SOTA claim, universal camera/AI separation, calibrated per-image probability,
completed video system or production false-alarm SLA is established. These changes
improve engineering defensibility, not classifier discrimination under shift.
