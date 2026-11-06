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
  * Under (C)/(D) the original threshold need not transfer. Recalibration uses
    known real images from the new domain. Finite-sample marginal control
    requires exchangeability, not just an observed rate below target.
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
    tpr_recalibrated: float   # power on the same fakes after recalibration
    fpr_original_tau: float   # realized FPR using the calibration-domain tau
    fpr_recalibrated: float   # realized FPR after real-only recal on this domain
    tau_original: float
    tau_recalibrated: float
    recal_certified: bool
    n_real: int
    n_fake: int
    alpha: float
    observed_recal_fpr_at_or_below_target: bool


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

    if (not real_scores.size or not fake_scores.size
            or not np.isfinite(real_scores).all() or not np.isfinite(fake_scores).all()):
        raise ValueError("Shift evaluation requires finite, nonempty real and fake test scores")
    recal = Calibrator.fit(recal_real_scores, alpha, method=method,
                           real_domain=real_domain)
    if not np.isfinite([base_calibrator.tau, recal.tau]).all():
        raise ValueError(f"Finite thresholds required; collect more real calibration groups for {real_domain}")
    recal_fpr = recal.fpr_on(real_scores)

    return ShiftCell(
        name=cell_name,
        real_domain=real_domain,
        fake_generator=fake_generator,
        tpr=base_calibrator.tpr_on(fake_scores),
        tpr_recalibrated=recal.tpr_on(fake_scores),
        fpr_original_tau=base_calibrator.fpr_on(real_scores),
        fpr_recalibrated=recal_fpr,
        tau_original=base_calibrator.tau,
        tau_recalibrated=recal.tau,
        recal_certified=recal.certified,
        n_real=int(real_scores.size),
        n_fake=int(fake_scores.size),
        alpha=alpha,
        observed_recal_fpr_at_or_below_target=recal_fpr <= alpha,
    )


def run_matrix(cells: list[ShiftCell]) -> dict:
    """Bundle cells into a serializable report with the headline takeaways."""
    if not cells:
        raise ValueError("No real-domain/generator cells to report")
    return {
        "schema_version": 2,
        "cells": [asdict(c) for c in cells],
        "summary": _summarize(cells),
        "note": "Observed target attainment is descriptive, not a population FPR guarantee.",
    }


def _summarize(cells: list[ShiftCell]) -> dict:
    # Generator cells in the same domain share their real test rows.
    real_domains = {c.real_domain: c for c in cells}.values()
    return {
        "mean_fpr_original_tau": float(np.mean([c.fpr_original_tau for c in real_domains])),
        "mean_fpr_recalibrated": float(np.mean([c.fpr_recalibrated for c in real_domains])),
        "mean_tpr": float(np.mean([c.tpr for c in cells])),
        "mean_tpr_recalibrated": float(np.mean([c.tpr_recalibrated for c in cells])),
        "all_domains_observed_recal_fpr_at_or_below_target": all(
            c.observed_recal_fpr_at_or_below_target for c in cells),
    }


def save_report(report: dict, path: str | Path) -> None:
    payload = json.dumps(report, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(payload + "\n")
