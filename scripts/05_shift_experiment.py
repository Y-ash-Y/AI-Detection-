#!/usr/bin/env python
"""Phase 4: the contribution. The shift matrix + real-only recalibration.

Calibrates tau on real images from a base domain, then walks the 2x2 of
{real-domain fixed/shifted} x {generator fixed/shifted}, measuring:
  - power (TPR) on each cell's fakes
  - realized FPR using the ORIGINAL tau
  - realized FPR after recalibrating tau on this domain's real images ONLY

Use scores from a frozen detector on data excluded from fitting. Known related
images stay together; observed rates do not establish a population guarantee.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pmsa.config import Config
from pmsa.calibration import Calibrator
from pmsa.data.splits import independent_indices
from pmsa.eval.shift_matrix import evaluate_cell, run_matrix, save_report


def prepare_partitions(score, label, domain, source, groups, fraction, seed):
    score, label, domain, source, groups = map(np.asarray, (score, label, domain, source, groups))
    if (score.ndim != 1 or not len(score)
            or any(x.shape != score.shape for x in (label, domain, source, groups))
            or not np.isfinite(score).all() or not np.isin(label, [0, 1]).all()
            or any(np.any(x == "") for x in (domain, source, groups))
            or not 0 < fraction < 1):
        raise ValueError("Expected finite aligned scores, binary labels, group identities and a valid calibration fraction")
    identities = {}
    for group, dom, src in zip(groups, domain, source):
        if identities.setdefault(group, (dom, src)) != (dom, src):
            raise ValueError("Image group spans multiple domains or sources; curate before shift evaluation")
    keep = independent_indices(label, groups)
    rng = np.random.default_rng(seed)
    partitions = {}
    for dom in sorted(set(domain[keep])):
        rows = keep[(domain[keep] == dom) & (label[keep] == 0)]
        if not len(rows):
            continue
        order = rng.permutation(rows)
        cut = int(fraction * len(order))
        if not 0 < cut < len(order):
            raise ValueError(f"Need separate real calibration and test groups for {dom}")
        partitions[dom] = (order[:cut], order[cut:])
    return keep, partitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scores", required=True,
                    help="npz from script 04: score, label, domain, source, groups")
    ap.add_argument("--base-domain", required=True,
                    help="real domain tau is initially calibrated on")
    ap.add_argument("--base-generator", required=True,
                    help="generator considered 'in-distribution' for the base cell")
    ap.add_argument("--out", type=Path, default=None,
                    help="New report path; defaults to configured output directory")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    alpha, method = cfg.calibration.alpha, cfg.calibration.method
    with np.load(args.scores, allow_pickle=False) as d:
        if "groups" not in d:
            raise ValueError("Score archive lacks group identities; regenerate with script 04")
        score, label = d["score"], d["label"]
        domain, source, groups = (d[key].astype(str) for key in ("domain", "source", "groups"))
    keep, partitions = prepare_partitions(score, label, domain, source, groups,
                                           cfg.calibration.cal_fraction, cfg.calibration.seed)
    if args.base_domain not in partitions or args.base_generator not in source[label == 1]:
        raise ValueError("Base domain needs real groups and base generator must be present among fakes")
    base = Calibrator.fit(score[partitions[args.base_domain][0]], alpha, method=method,
                          real_domain=args.base_domain)
    print(f"base tau={base.tau:.4f} (n_cal={base.n_cal}, "
          f"certified={base.certified})")

    cells = []
    for dom, (cal_ix, test_ix) in partitions.items():
        fake_rows = keep[(label[keep] == 1) & (domain[keep] == dom)]
        for gen in sorted(set(source[fake_rows])):
            fake = score[fake_rows[source[fake_rows] == gen]]
            name = _cell_name(dom == args.base_domain, gen == args.base_generator)
            cells.append(evaluate_cell(
                name, base, score[test_ix], fake, score[cal_ix],
                real_domain=dom, fake_generator=gen,
                alpha=alpha, method=method))

    report = run_matrix(cells)
    report["real_partitions"] = {dom: {"calibration_groups": groups[ca].tolist(),
                                      "test_groups": groups[te].tolist()}
                                 for dom, (ca, te) in partitions.items()}
    out = args.out or Path(cfg.out_dir) / "shift_matrix.json"
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
          f"{'TPR(orig)':>11}{'TPR(recal)':>12}{'FPR(orig)':>11}{'FPR(recal)':>12}")
    print("-" * 88)
    for c in sorted(cells, key=lambda x: x.name):
        print(f"{c.name:<13}{c.real_domain:<12}{c.fake_generator:<16}"
              f"{c.tpr:>11.3f}{c.tpr_recalibrated:>12.3f}"
              f"{c.fpr_original_tau:>11.3f}{c.fpr_recalibrated:>12.3f}")
    print(f"\ntarget alpha = {alpha}")


if __name__ == "__main__":
    main()
