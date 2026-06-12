#!/usr/bin/env python
"""Phase 4: the contribution. The shift matrix + real-only recalibration.

Calibrates tau on real images from a base domain, then walks the 2x2 of
{real-domain fixed/shifted} x {generator fixed/shifted}, measuring:
  - power (TPR) on each cell's fakes
  - realized FPR using the ORIGINAL tau
  - realized FPR after recalibrating tau on this domain's real images ONLY

Tests the thesis: under generator shift the FPR guarantee holds with the original
tau; under real-domain shift it breaks but real-only recalibration restores it.

Assumes you already have scores. The simplest wiring: a trained detector + a
FeatureSet whose `domain`/`source` tags define the cells. See README for the
full recipe; this script is the harness.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pmsa.config import Config
from pmsa.calibration import Calibrator
from pmsa.eval.shift_matrix import evaluate_cell, run_matrix, save_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scores", required=True,
                    help="npz with arrays: score, label, domain, source")
    ap.add_argument("--base-domain", required=True,
                    help="real domain tau is initially calibrated on")
    ap.add_argument("--base-generator", required=True,
                    help="generator considered 'in-distribution' for the base cell")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    alpha, method = cfg.calibration.alpha, cfg.calibration.method
    d = np.load(args.scores, allow_pickle=False)
    score, label = d["score"], d["label"]
    domain, source = d["domain"].astype(str), d["source"].astype(str)

    # --- fit base tau on real images of the base domain (split cal/test) ---
    base_real = score[(label == 0) & (domain == args.base_domain)]
    rng = np.random.default_rng(cfg.calibration.seed)
    perm = rng.permutation(base_real.size)
    cut = int(cfg.calibration.cal_fraction * base_real.size)
    base_cal = base_real[perm[:cut]]
    base = Calibrator.fit(base_cal, alpha, method=method,
                          real_domain=args.base_domain)
    print(f"base tau={base.tau:.4f} (n_cal={base.n_cal}, "
          f"certified={base.certified})")

    domains = sorted(set(domain))
    generators = sorted(set(source[label == 1]))
    cells = []
    for dom in domains:
        dom_real = score[(label == 0) & (domain == dom)]
        if dom_real.size == 0:
            continue
        # split this domain's reals into recal pool + held-out test reals
        p = rng.permutation(dom_real.size)
        h = dom_real.size // 2
        recal_real, test_real = dom_real[p[:h]], dom_real[p[h:]]
        for gen in generators:
            fake = score[(label == 1) & (domain == dom) & (source == gen)]
            if fake.size == 0:
                continue
            name = _cell_name(dom == args.base_domain, gen == args.base_generator)
            cells.append(evaluate_cell(
                name, base, test_real, fake, recal_real,
                real_domain=dom, fake_generator=gen,
                alpha=alpha, method=method))

    report = run_matrix(cells)
    out = Path(cfg.out_dir) / "shift_matrix.json"
    save_report(report, out)
    _print_table(cells, alpha)
    print(f"\nsaved -> {out}")


def _cell_name(real_fixed, gen_fixed):
    if real_fixed and gen_fixed:
        return "A_in_dist"
    if real_fixed and not gen_fixed:
        return "B_gen_shift"
    if not real_fixed and gen_fixed:
        return "C_real_shift"
    return "D_both_shift"


def _print_table(cells, alpha):
    print(f"\n{'cell':<13}{'domain':<12}{'gen':<16}"
          f"{'TPR':>7}{'FPR(orig)':>11}{'FPR(recal)':>12}")
    print("-" * 71)
    for c in sorted(cells, key=lambda x: x.name):
        print(f"{c.name:<13}{c.real_domain:<12}{c.fake_generator:<16}"
              f"{c.tpr:>7.3f}{c.fpr_original_tau:>11.3f}{c.fpr_recalibrated:>12.3f}")
    print(f"\ntarget alpha = {alpha}")


if __name__ == "__main__":
    main()
