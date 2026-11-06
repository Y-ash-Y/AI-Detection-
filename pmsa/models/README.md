# Detector heads

[linear_probe.py](linear_probe.py) fits a standardized logistic probe on one
frozen feature stream. [fusion.py](fusion.py) fits neural stream heads and a
combiner; signed stream contributions and combiner bias account for the score.
These contributions describe the model computation, not causal explanations.

The head's expected stream order, feature dimensions and preprocessing are
part of its serving contract. Load only trusted artifacts. In particular,
legacy joblib/pickle artifacts are not safe to load from untrusted sources.
See [bundle and serving instructions](../../deployment/README.md).
