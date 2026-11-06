# Evaluation and interpretation

[Project overview](../README.md) · [Historical v1 baseline](v1_baseline.md)

## Evidence status

The project closes with the original frozen SigLIP + forensic fusion release.
Three-seed evaluation and sampled serving-parity checks are complete. Chameleon
was previously inspected, so this is a reproducible benchmark evaluation, not
a newly blinded holdout or evidence of production reliability.

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

### Why the deployed model stayed frozen

Later development trials tested local SigLIP patches, modern training data and
real-only threshold adjustments. None demonstrated a reliable improvement in
modern-image detection without increasing false alarms or losing detection power.


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

## Ranking metric convention

New ranking reports use the available empirical false-positive budget for
`TPR@FPR`, preserving strict threshold comparisons and ties. This can differ
from older quantile-based ranking reports, especially on small sets. It does
not change the deployed threshold or its real-only calibration rule.
