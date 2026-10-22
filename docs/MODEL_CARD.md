# Model card and evidence status

## Intended use and ownership

PMSA is a still-image detection research/portfolio system. Its detection heads,
fusion of semantic and handcrafted features, and real-score threshold calibration
are project components. Google's pretrained SigLIP is a credited, frozen backbone:
this project did not pretrain it. The optional third-party detector adapter is an
explicit comparison baseline, not the application's default or silent fallback.

Use it to explore detection behavior, not to accuse people, authenticate evidence,
moderate accounts automatically, or certify whether a photograph is genuine.
It does not perform video/deepfake detection. Digitally drawn logos are not
necessarily generative AI. AI-edited photographs are a mixed-origin case outside
the current binary labeling protocol.

## Current benchmark evidence: fresh metadata-bearing full run

The fresh extraction/full head comparison completed and its archive/source
integrity was reviewed. This does not retroactively validate legacy extraction.
Chameleon was already inspected during earlier work; the run is benchmark
reproduction, not a newly blinded test or a production release.

| Head | Mean AUC | Mean frozen-threshold FPR | Mean frozen-threshold TPR |
|---|---:|---:|---:|
| SigLIP linear probe | 0.8697 | 6.11% | 45.92% |
| Forensic linear probe | 0.5969 | 5.14% | 3.01% |
| SigLIP nonlinear MLP | 0.9032 | 8.65% | 63.28% |
| SigLIP + forensic fusion | 0.9105 | 8.25% | 64.66% |

These means cover seeds 0/1/2 on the same 26,017 retained representatives
(14,863 real, 11,154 generated). Each model uses its separate real-only broad
calibration threshold at target alpha 0.05. The threshold is not fitted to
Chameleon test reals. Seed means/spread are not population confidence intervals;
per-seed intervals and full predictions remain in private run artifacts. The
[aggregate evidence snapshot](evidence/fresh_full_summary.json) records exact
means and source archive identity without personal or per-image data.

Fusion ranking is stronger than the linear probe, but false alarms still exceed
the intended 5% under shift. This is not an accuracy increase caused by the later
software hardening. Later small private development checks found persistent real
false alarms; a training-only JPEG/resize pilot did not resolve its targeted
errors and was not promoted. The proposed matched-scene confirmation phase is
deferred. No evaluated seed is automatically the hosted application's model.

## Historical evidence (separate from the fresh full run)

These numbers were recorded in earlier notebook runs. Recovered feature caches
lack extraction/version metadata and original image hashes. A separate recovered-cache
evaluation now uses the independent split protocol, as described below; it does not
make the old feature extraction auditable. These historical numbers must
not be described as results of the updated protocol, a production SLA, or SOTA.

| Chameleon, 26,033 images | SigLIP probe | SigLIP + forensic fusion |
|---|---:|---:|
| AUC | 0.8511 | 0.8986 |
| Reported bootstrap 95% AUC interval | [0.847, 0.855] | [0.895, 0.902] |
| TPR at test-derived 1% FPR threshold | 0.1430 | 0.2517 |
| TPR at test-derived 5% FPR threshold | 0.3755 | 0.5013 |
| Realized FPR at deployed threshold | 0.0630 | 0.1020 |
| TPR at deployed threshold | 0.4153 | 0.6613 |
| Accuracy at deployed threshold | 0.7131 | 0.7965 |

Chameleon contained 14,863 real and 11,170 fake examples. The fusion's deployed
target was 5% FPR; its observed 10.2% was more than twice that target. Its AUC is
a ranking result, not a 89.86% chance of correctly classifying any uploaded image.
The fusion-minus-probe AUC difference was 0.0475. This association does not prove
the forensic branch discovered a particular generator's causal fingerprint.

Historical frontier few-shot outputs used collections of 100 fake images each:

| Collection | k=0 TPR | k=10 TPR | k=25 TPR | k=50 TPR |
|---|---:|---:|---:|---:|
| GPT-Image | 0.300 ± 0.028 | 0.673 ± 0.034 | 0.787 ± 0.034 | 0.827 ± 0.009 |
| Nano Banana | 0.573 ± 0.141 | 0.793 ± 0.115 | 0.900 ± 0.033 | 0.907 ± 0.034 |

The ± values describe spread across three seeds, **not** population confidence
intervals. Sampled test subsets may differ across k and share images across seeds;
this is not a paired causal learning curve. Adapted models have seen labeled
examples from that generator and are not zero-shot. These small, curated collections
cannot establish performance on all versions, prompts, editing modes, or download
pipelines of the named products. Generator names/provenance require source records;
an image filename alone cannot verify generation origin.

## Recovered-cache evaluation (exploratory, not a released bundle)

A local evaluation fitted four heads with seeds 0/1/2 on recovered features,
using common group-separated fitting, validation, real calibration and internal
test partitions. Chameleon remained outside classifier fitting and primary
calibration, but had been viewed during earlier project development. Semantic/path
identity grouping retained 39,960 development and 26,017 external representatives;
this cannot establish raw-image or near-duplicate independence.

| Head | Mean Chameleon AUC | FPR at frozen 5%-target threshold | TPR at that threshold |
|---|---:|---:|---:|
| SigLIP linear probe | 0.8584 | 5.75% | 42.53% |
| Forensic linear probe | 0.6021 | 5.12% | 3.21% |
| SigLIP nonlinear MLP | 0.9092 | 7.85% | 64.31% |
| SigLIP + forensic fusion | 0.9147 | 7.91% | 65.64% |

The nonlinear control was added after the initial three-model comparison; no
external hyperparameter search was performed. Fusion's mean AUC gain over this
control was 0.00544, with a paired 95% percentile bootstrap interval
[0.00471, 0.00618] conditional on the three fitted heads (2,000 resamples).
Its seed AUC range was [0.9081, 0.9205]. Seed range is not a confidence interval.
The comparison with the linear probe alone exaggerates the association attributable
to the extra branch; parameter counts also differ between nonlinear architectures.

A separate supplement used 2,000 Chameleon reals for calibration and evaluated on
the remaining 12,863 reals and all 11,154 retained fakes. On the same test rows,
fusion's mean FPR moved from 8.00% to 5.45%, while TPR fell from 65.64% to 57.24%.
This is target-real adaptation, not a zero-shot threshold, and not proof of a
realized FPR always below 5%. Per-run intervals, source checksums, prediction files,
split identities and methodological qualifications are saved locally in
`outputs/recovered_eval_v3_01/REPORT.md` and its accompanying JSON evidence.
These outputs are ignored by Git and are not packaged as deployable models.

Both streams were finite and aligned, both manifests reconciled with the caches,
and no path or exact-feature overlap was observed across development and Chameleon.
Missing extraction provenance still prevents validating image-to-app equivalence.
These results do not certify SOTA, all frontier generators, or a production SLA.

## Decision and uncertainty semantics

- Higher forensic score means more AI-like according to this trained head.
- Binary rule: score > tau. Equality does not flag AI.
- The app displays `AI-generated signal`, `No AI signal detected`, or `Uncertain`.
- `No AI signal detected` does not prove camera origin.
- A ±0.25 **score-unit** band around tau is a UI abstention heuristic. It is not a
  statistical coverage guarantee, calibrated uncertainty, or universal choice of band.
- No probability calibrator is fitted. Sigmoid/softmax output must not be displayed
  as a reliable per-image AI probability. The local detector returns probability null.
- Stream contribution = learned combiner weight × head logit. Contributions plus
  the combiner bias reconstruct the score; these are not causal explanations.

## Calibration assumptions

The conformal threshold uses real-only calibration scores independent of fitting
and validation. For n exchangeable real scores it uses the ceil((n+1)(1-alpha))th
order statistic, or infinity when insufficient data makes a finite threshold
impossible. Its FPR statement is marginal over calibration and a fresh real draw.
It is **not** a high-probability guarantee on the realized FPR of each deployed
threshold, and does not automatically survive a change in real-image distribution.
Recalibration requires trusted real examples representative of the intended domain,
separate from evaluation. Mixed unlabeled uploads do not satisfy that requirement.

## Data and preprocessing risks

The original training assembly selected a balanced 40,000 examples from GenImage,
FFHQ, COCO and StyleGAN2 faces. Global class balance does not remove domain/class
confounding: COCO contributes only real images and faces have a separate generator.
JPEG re-encoding at quality 90 standardizes one step; it does **not** remove source
compression, resizing, semantic, acquisition or generator biases.

V3 records preprocessing, processor mode, backbone revision, package versions,
original file hashes and optional curator-supplied related-image identifiers.
Known related groups must not cross splits. Exact hashes cannot identify all
near-duplicates or images made from the same prompt. One representative per known
group is used in the current IID metrics protocol; mixed-label edit groups require
a separate protocol and are rejected rather than silently relabeled.

## Safety, privacy and artifact trust

The metadata scanner is a bounded keyword/EXIF hint search. It does not validate
C2PA signatures or detect an invisible SynthID watermark. Keywords can be spoofed;
content credentials can exist on camera photographs. Hints never override scores.

Uploads are not used for training or published by this application. Gradio uses
temporary uploaded files with scheduled cleanup; this is not a zero-retention
promise for hosting providers. Avoid sensitive uploads to public demos.

Bundle hashes detect changes or accidental model/calibrator swaps, **not** publisher
authenticity: an attacker can replace both files and hashes. Load only trusted
bundles. V3 fusion uses weights-only checkpoint loading; legacy unrestricted pickle
requires explicit opt-in. Joblib probes are executable pickle artifacts and must
also be trusted. Unknown or mismatched bundles fail instead of switching detectors.

## Release gates still open

Reliability tests validate software contracts, not improved detection accuracy.
Fresh metadata-bearing benchmark evaluation and sampled image-serving parity
have completed. Remaining gates include representative new confirmation data,
broader near-duplicate/prompt-family and source-bias review, target-domain
operating-rate evidence, and a separate deployment/HTTP validation if requested.
Keep historical, benchmark-reproduction and reused development results distinct;
do not silently redeploy a candidate or mark the deferred data phase complete.
