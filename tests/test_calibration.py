"""Tests for the calibration core — the contribution, so it gets real coverage.

These run NOW, no data or GPU needed: synthetic scores stand in for detector
outputs. They lock the two properties the thesis rests on:
  1. conformal tau gives finite-sample FPR <= alpha (in expectation over draws),
  2. recalibrating on real-only scores from a shifted domain restores the FPR.
"""
import numpy as np
import pytest

from pmsa.calibration import (
    Calibrator, conformal_threshold, empirical_threshold,
    min_calibration_size, realized_fpr,
)


def test_empirical_threshold_hits_alpha_on_calibration_set():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, 10000)
    tau = empirical_threshold(real, alpha=0.05)
    assert realized_fpr(real, tau) <= 0.05 + 1e-9


def test_conformal_returns_inf_when_too_few_samples():
    # need ceil(1/alpha)-1 = 99 samples for alpha=0.01
    assert min_calibration_size(0.01) == 99
    few = np.arange(50, dtype=float)
    assert conformal_threshold(few, alpha=0.01) == float("inf")


def test_conformal_marginal_guarantee_holds_in_expectation():
    """Over many calibration/test draws, P(real_test > tau) <= alpha marginally."""
    alpha = 0.1
    rng = np.random.default_rng(42)
    fprs = []
    for _ in range(300):
        cal = rng.normal(0, 1, 200)
        test = rng.normal(0, 1, 500)
        tau = conformal_threshold(cal, alpha)
        fprs.append(realized_fpr(test, tau))
    # conformal guarantees E[FPR] <= alpha; allow tiny MC slack
    assert np.mean(fprs) <= alpha + 0.01


def test_original_tau_breaks_under_real_domain_shift():
    rng = np.random.default_rng(1)
    base_real = rng.normal(0, 1, 5000)
    cal = Calibrator.fit(base_real, alpha=0.01, method="conformal",
                         real_domain="base")
    # shifted real domain: mean moves up, scores inflate
    shifted_real = rng.normal(1.5, 1, 5000)
    assert cal.fpr_on(shifted_real) > 0.05  # guarantee broken


def test_real_only_recalibration_restores_guarantee():
    rng = np.random.default_rng(2)
    base_real = rng.normal(0, 1, 5000)
    cal = Calibrator.fit(base_real, alpha=0.01, method="conformal")
    shifted_recal = rng.normal(1.5, 1, 3000)   # fakes-free recal pool
    shifted_test = rng.normal(1.5, 1, 3000)    # held-out reals, same domain
    recal = Calibrator.fit(shifted_recal, alpha=0.01, method="conformal",
                           real_domain="shifted")
    assert recal.fpr_on(shifted_test) <= 0.03  # restored near alpha


def test_generator_shift_does_not_touch_real_guarantee():
    """Changing the fake generator leaves the real distribution — and tau — valid."""
    rng = np.random.default_rng(3)
    real = rng.normal(0, 1, 5000)
    cal = Calibrator.fit(real, alpha=0.01, method="conformal")
    held_real = rng.normal(0, 1, 5000)  # same real domain, new generator elsewhere
    assert cal.fpr_on(held_real) <= 0.03


def test_calibrator_roundtrip(tmp_path):
    real = np.random.default_rng(4).normal(0, 1, 500)
    cal = Calibrator.fit(real, alpha=0.05, method="empirical", real_domain="x")
    p = tmp_path / "cal.json"
    cal.save(p)
    back = Calibrator.load(p)
    assert back.tau == cal.tau and back.real_domain == "x"
