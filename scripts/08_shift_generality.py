#!/usr/bin/env python
"""Phase 4 generality: the FPR-break severity gradient across real domains.

Trains ONE detector on GenImage (base real domain), calibrates tau once on
held-out base reals, then measures — for each shift domain — how badly the FPR
guarantee breaks with that fixed tau, and whether real-only recalibration on that
domain restores it. Reals-only: no fakes needed for this table.

The prediction is a gradient: the further a real domain sits from the base, the
larger the break (FPR@base_tau), while FPR@recal stays ~alpha everywhere.

    python scripts/08_shift_generality.py --train-tag train \
        --base-domain imagenet --shift-tags coco,ffhq
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import FusionDetector, StreamSpec
from pmsa.calibration import Calibrator
from pmsa.utils import get_device


def _load(cfg, tag):
    sets = [FeatureSet.load(Path(cfg.feature.cache_dir) / f"{bb.name}_{tag}.npz")
            for bb in cfg.feature.backbones if bb.enabled]
    base = sets[0]
    for s in sets[1:]:
        assert np.array_equal(s.paths, base.paths), f"{tag} caches not aligned"
    return sets, base


def _score(det, sets, idx):
    return det.score([s.features[idx] for s in sets])


def run_once(cfg, device, tr_sets, tr, shift, args, seed):
    rng = np.random.default_rng(seed)
    alpha, method = cfg.calibration.alpha, cfg.calibration.method

    base_real = np.flatnonzero((tr.labels == 0) & (tr.domain == args.base_domain))
    rng.shuffle(base_real)
    n = base_real.size
    a, b = int(0.6 * n), int(0.8 * n)
    real_train, real_cal, real_test = base_real[:a], base_real[a:b], base_real[b:]
    fake_idx = np.flatnonzero(tr.labels == 1)
    train_idx = np.concatenate([fake_idx, real_train])

    det = FusionDetector([StreamSpec(s.backbone, s.dim) for s in tr_sets],
                         device=device, seed=seed)
    det.fit([s.features[train_idx] for s in tr_sets], tr.labels[train_idx],
            epochs=cfg.train.epochs, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            batch_size=cfg.train.batch_size, verbose=False)

    base_cal = Calibrator.fit(_score(det, tr_sets, real_cal), alpha, method=method,
                              real_domain=args.base_domain)
    row = {args.base_domain: {
        "fpr_base_tau": base_cal.fpr_on(_score(det, tr_sets, real_test)),
        "fpr_recal": base_cal.fpr_on(_score(det, tr_sets, real_test)),  # in-domain: same
        "is_base": True,
    }}
    for dom, (sets, fs) in shift.items():
        ridx = np.flatnonzero(fs.labels == 0)
        rng.shuffle(ridx)
        h = ridx.size // 2
        recal = Calibrator.fit(_score(det, sets, ridx[:h]), alpha, method=method,
                               real_domain=dom)
        test = _score(det, sets, ridx[h:])
        row[dom] = {
            "fpr_base_tau": base_cal.fpr_on(test),
            "fpr_recal": recal.fpr_on(test),
            "is_base": False,
        }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-tag", default="train")
    ap.add_argument("--base-domain", default="imagenet")
    ap.add_argument("--shift-tags", default="coco,ffhq",
                    help="comma list of reals-only cache tags (each = a shift domain)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = get_device()
    tr_sets, tr = _load(cfg, args.train_tag)
    shift = {}
    for tag in [t for t in args.shift_tags.split(",") if t]:
        sets, fs = _load(cfg, tag)
        shift[tag] = (sets, fs)

    runs = [run_once(cfg, device, tr_sets, tr, shift, args, seed)
            for seed in cfg.train.seeds]

    domains = [args.base_domain] + list(shift)
    alpha = cfg.calibration.alpha
    report = {"alpha": alpha, "seeds": cfg.train.seeds, "domains": {}}
    print(f"\ntarget alpha = {alpha}   seeds={cfg.train.seeds}")
    print(f"{'real domain':>14}{'FPR @ base tau':>18}{'FPR @ recal':>14}")
    print("-" * 46)
    for dom in domains:
        fb = [r[dom]["fpr_base_tau"] for r in runs]
        fr = [r[dom]["fpr_recal"] for r in runs]
        report["domains"][dom] = {
            "fpr_base_tau_mean": float(np.mean(fb)), "fpr_base_tau_std": float(np.std(fb)),
            "fpr_recal_mean": float(np.mean(fr)), "fpr_recal_std": float(np.std(fr)),
            "is_base": runs[0][dom]["is_base"],
        }
        tag = "  (base)" if runs[0][dom]["is_base"] else ""
        print(f"{dom:>14}{np.mean(fb):>13.3f}{'':5}{np.mean(fr):>11.3f}{tag}")

    out = Path(cfg.out_dir) / "shift_generality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\nFPR @ base tau should grow with shift severity; FPR @ recal ~alpha for all.")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
