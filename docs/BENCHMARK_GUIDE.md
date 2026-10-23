# Reading the results without overclaiming

## The short answer

PMSA is a useful experimental detector with reproducible evaluation and a working
serving pipeline. Its fresh Chameleon ranking performance is encouraging. Its
false-alarm rate and missed detections do **not** justify treating it as a
definitive real-versus-AI judge, especially for personal photos or unfamiliar
generators. A better interface cannot repair that model limitation.

## Three different questions

| Question | Measure | What the current evidence says |
|---|---|---|
| Do generated images generally get higher scores than real images? | ROC AUC | Fusion mean 0.9105: good ranking on this benchmark, not 91.05% accuracy. |
| How many generated images are flagged at the chosen threshold? | True-positive rate (TPR) | Fusion mean 64.66%, roughly 65 per 100 generated images. Roughly 35 per 100 are missed. |
| How many real images are wrongly flagged at that same threshold? | False-positive rate (FPR) | Fusion mean 8.25%, roughly 8 per 100 real images. The separate calibration target was 5%. |

The denominator for TPR is generated images; the denominator for FPR is real
images. AUC evaluates ordering across thresholds. It does not specify any
particular operating threshold. None of these is the probability that a specific
uploaded image is AI-generated.

For AUC, imagine independently choosing one generated and one real example from
the evaluated population. AUC measures how often the generated example has the
higher score, with half credit for ties. An AUC near 0.91 can coexist with
unacceptable false alarms at the threshold used by an app.

These numbers are means across three fitted seeds on **26,017 retained Chameleon
representatives** (14,863 real and 11,154 generated). They are not three independent
test datasets or confidence intervals. The app serves fixed seed 0, not an
ensemble whose performance equals the three-seed mean. Chameleon was previously
inspected; this is not a newly blinded assessment. Do not merge this run with the
older 26,033-image notebook reports or use their differences as a controlled trend.

## What did fusion actually add?

The same fresh-run evidence compares four heads:

| Model | Mean AUC | Frozen-threshold FPR | Frozen-threshold TPR |
|---|---:|---:|---:|
| SigLIP linear probe | 0.8697 | 6.11% | 45.92% |
| Forensic linear probe | 0.5969 | 5.14% | 3.01% |
| SigLIP nonlinear head | 0.9032 | 8.65% | 63.28% |
| SigLIP + forensic fusion | 0.9105 | 8.25% | 64.66% |

Most of the difference from the linear SigLIP baseline is already present in the
nonlinear SigLIP head. Fusion adds about **0.0073 AUC**, **1.38 percentage points
of TPR**, and **0.40 percentage points lower FPR** relative to that nonlinear
head's observed operating point. This is a modest incremental benefit, not proof
that the forensic stream is the main source of generalization. These are
descriptive differences of means, not a statistical-significance test, and the
heads have different separately calibrated thresholds rather than equal observed
test FPR. Paired uncertainty analysis is needed before claiming a reliable gain.

The forensic probe alone is weak. A signed forensic contribution is a term in
the trained network, not evidence that the system located a physical synthesis
artifact or identified which generator made the image.

## Why “95% AI” is not available

The network produces a real-valued score, and calibration here chooses a
decision threshold from reference real images. This controls an operating rule
under its stated assumptions; it does not fit per-image posterior probabilities.

The sigmoid transform, `1 / (1 + exp(-score))`, maps a score into the interval
0–1, but the arithmetic alone does not establish calibration. Calling that number
“confidence” would recreate the earlier misleading UI. The app instead shows
the actual score, threshold and distance, with an explicit inconclusive state
near the boundary. That abstention band is heuristic, not a confidence interval.

The classifier is still probabilistic: binary cross-entropy trains its sigmoid
as a Bernoulli class-probability estimate on the training distribution. What is
absent is evidence that this estimate is calibrated for the population using the
app. “Model-estimated probability” and “validated confidence” are different
claims; real-only threshold calibration is a third, different operation.

To support a percentage later: collect representative labeled development data,
fit a separate probability-calibration model on disjoint data, and evaluate
reliability diagrams, Brier score and log loss on untouched test groups. Recheck
calibration by domain and generator. Even good in-domain calibration is not a
universal guarantee under distribution shift or a changing real/AI prevalence.

## Recommended next model work, in order

1. **Define the deployment population.** Camera photos, shared/compressed photos,
   conventional edits, AI edits and fully generated images need explicit scope
   and labels. A traditional logo is not automatically generative AI.
2. **Freeze a new evaluation collection before tuning.** Keep related photos and
   prompt variants together. Reused frontier examples are development evidence,
   not a fresh zero-shot holdout. Add modern generator versions and varied real
   sources; audit duplicates, source shortcuts, watermark and compression effects.
3. **Reduce real-photo false alarms.** Use broad, deployment-matched real data
   for development/calibration; choose acceptable false-alarm targets before
   looking at final test outcomes. Measure detection power at each target rather
   than silently moving the threshold until selected photos look right.
4. **Compare improvements fairly.** Repeat the SigLIP-only nonlinear baseline
   alongside fusion with common group splits, several seeds and paired intervals.
   Compare TPR at a common FPR as well as each frozen deployed threshold. Do not
   add another backbone without evidence its benefit exceeds cost and complexity.
5. **Only then add a probability display.** Validate it independently and monitor
   drift. A trustworthy “inconclusive” result is preferable to a fabricated 99%.

Engineering tests establish software behavior, artifact integrity and serving
consistency; they do not establish detection accuracy, fairness, privacy
certification or reliability for every current generator.

Source: [reviewed fresh-run aggregate evidence](evidence/fresh_full_summary.json).
Full per-image evidence remains private. This guide interprets existing results;
it is not a new experiment or a SOTA comparison.
