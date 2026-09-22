"""PMSA v2 — Probabilistic Media Synthesis & Authentication.

A calibrated, generator-invariant detector for synthetic images.

Thesis: the Neyman-Pearson false-alarm guarantee (FPR <= alpha) depends only on
the *real* image distribution, so it is recoverable under shift via cheap
per-domain recalibration on unlabeled real images — even when detection power
(driven by the *fake* generator) is not.

See README.md for the project overview and docs/report.tex for the technical report.
"""

__version__ = "0.1.0"
