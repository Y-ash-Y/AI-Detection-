# Real-only threshold calibration

[np_threshold.py](np_threshold.py) implements empirical and conformal threshold
rules; [conformal.py](conformal.py) fits and serializes the calibrator. Fit the
threshold on separate real calibration groups, then apply it unchanged to an
evaluation split. The decision uses a strict score-above-threshold comparison.

A target false-positive rate refers to the reference real domain. It is not a
per-image probability or a promise under domain shift. Recalibrating on target
domain reals is a separate experiment, not zero-shot threshold transfer.
See [evaluation and interpretation](../../reference/EVALUATION.md).
