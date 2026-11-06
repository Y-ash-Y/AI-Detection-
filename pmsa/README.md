# PMSA package

The current detector uses frozen SigLIP features and deterministic forensic
features. Training, calibration, evaluation and serving share an explicit
preprocessing policy; a trained head is not interchangeable with arbitrary
features or calibration thresholds.

| Area | Responsibility |
| --- | --- |
| [Data](data/README.md) | Reviewed inputs, identities, groups and disjoint splits |
| [Backbones](backbones/README.md) | The two active feature extractors |
| [Features](features/README.md) | Extraction metadata, aligned caches and identities |
| [Models](models/README.md) | Linear probes and trained fusion heads |
| [Calibration](calibration/README.md) | Real-only decision thresholds |
| [Evaluation](eval/README.md) | Ranking metrics, operating points and shift analysis |
| [Utilities](utils/README.md) | Device selection and reproducible seeds |

[training.py](training.py) fits and calibrates separate partitions.
[evaluation.py](evaluation.py) checks external data at a frozen threshold.
[artifacts.py](artifacts.py) verifies bundle pairing and runtime compatibility;
[preprocessing.py](preprocessing.py) defines the shared image policy.
[inference.py](inference.py) runs a trusted bundle; [provenance.py](provenance.py)
reports unverified metadata hints without changing the decision.
[development.py](development.py) is for candidate development, not auto-promotion.
[parity.py](parity.py) and [photo_evaluation.py](photo_evaluation.py) support
serving checks and private frozen-model diagnostics.

Start with the [project overview](../README.md) or the [offline example](../examples/README.md).
