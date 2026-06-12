"""The contribution: how the FPR guarantee behaves under distribution shift,
and whether real-only recalibration restores it.

The 2x2 of what can shift, relative to the calibration domain:

                    | fake generator FIXED | fake generator SHIFTED
  real domain FIXED |  (A) in-distribution |  (B) generator shift
  real domain SHIFT |  (C) real-domain     |  (D) both
                    |      shift           |

Hypotheses (PMSA v2 thesis):
  * tau depends only on the real distribution. So under (B) the FPR guarantee
    should HOLD with the original tau (real domain unchanged); only power drops.
  * Under (C)/(D) the original tau breaks the FPR guarantee, because the real
    distribution moved. But recalibrating tau on unlabeled real images from the
    NEW domain — no fakes needed — should RESTORE FPR <= alpha.

This module runs that matrix and produces the table that (as of writing) does
not exist in the literature.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np

from ..calibration import Calibrator


@dataclass
class ShiftCell:
    name: str                 # "A_in_dist", "B_gen_shift", ...
    real_domain: str
    fake_generator: str
    tpr: float                # power on this cell's fakes
    fpr_original_tau: float   # realized FPR using the calibration-domain tau
    fpr_recalibrated: float   # realized FPR after real-only recal on this domain
    tau_original: float
    tau_recalibrated: float
    recal_certified: bool
    n_real: int
    n_fake: int


def evaluate_cell(
    cell_name: str,
    base_calibrator: Calibrator,
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    recal_real_scores: np.ndarray,
    real_domain: str,
    fake_generator: str,
    alpha: float,
    method: str = "conformal",
) -> ShiftCell:
    """Evaluate one cell of the shift matrix.

    real_scores       : held-out real scores from THIS cell's real domain (test).
    fake_scores       : fake scores from THIS cell's generator (test).
    recal_real_scores : disjoint real scores from THIS domain used to refit tau
                        (the cheap, fakes-free recalibration). For an in-domain
                        cell this is just more of the same real distribution.
    """
    real_scores = np.asarray(real_scores).ravel()
    fake_scores = np.asarray(fake_scores).ravel()

    recal = Calibrator.fit(recal_real_scores, alpha, method=method,
                           real_domain=real_domain)

    return ShiftCell(
        name=cell_name,
        real_domain=real_domain,
        fake_generator=fake_generator,
        tpr=base_calibrator.tpr_on(fake_scores),
        fpr_original_tau=base_calibrator.fpr_on(real_scores),
        fpr_recalibrated=recal.fpr_on(real_scores),
        tau_original=base_calibrator.tau,
        tau_recalibrated=recal.tau,
        recal_certified=recal.certified,
        n_real=int(real_scores.size),
        n_fake=int(fake_scores.size),
    )


def run_matrix(cells: list[ShiftCell]) -> dict:
    """Bundle cells into a serializable report with the headline takeaways."""
    return {
        "cells": [asdict(c) for c in cells],
        "summary": _summarize(cells),
    }


def _summarize(cells: list[ShiftCell]) -> dict:
    def avg(key):
        vals = [getattr(c, key) for c in cells if not np.isnan(getattr(c, key))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "mean_fpr_original_tau": avg("fpr_original_tau"),
        "mean_fpr_recalibrated": avg("fpr_recalibrated"),
        "mean_tpr": avg("tpr"),
        "guarantee_restored": avg("fpr_recalibrated") <= max(c.fpr_original_tau
                                                             for c in cells),
    }


def save_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2))
