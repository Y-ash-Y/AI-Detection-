#!/usr/bin/env python
"""Phase 4 core experiment: the FPR guarantee under real-domain shift.

This is the central claim of PMSA v2, demonstrated end to end:

  1. Train a fusion detector on GenImage (base real domain = imagenet).
  2. Calibrate tau on HELD-OUT base-domain reals (never seen in training).
  3. In-domain sanity: FPR <= alpha holds on more base-domain reals.
  4. Shift: apply the SAME tau to reals from a NEW domain (e.g. FFHQ faces) ->
     FPR blows past alpha. The guarantee breaks because the real distribution moved.
  5. Recovery: recalibrate tau on new-domain reals ONLY (no fakes) ->
     FPR <= alpha is restored.

No fake images from the new domain are needed for the recovery — that is the
whole point: the false-alarm guarantee is a property of the real distribution.

Prereqs (cached features):
  feature_cache/<backbone>_<train-tag>.npz   GenImage (reals + fakes), base domain
  feature_cache/<backbone>_<shift-tag>.npz   new-domain reals only (e.g. FFHQ)

    python scripts/06_real_domain_shift.py --train-tag train --shift-tag ffhq \
        --base-domain imagenet --shift-domain ffhq
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


def _load_streams(cfg, tag):
    sets = [FeatureSet.load(Path(cfg.feature.cache_dir) / f"{bb.name}_{tag}.npz")
            for bb in cfg.feature.backbones if bb.enabled]
    base = sets[0]
    for s in sets[1:]:
        assert np.array_equal(s.paths, base.paths), f"{tag} caches not aligned"
    return sets, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-tag", default="train")
    ap.add_argument("--shift-tag", default="ffhq")
    ap.add_argument("--base-domain", default="imagenet")
    ap.add_argument("--shift-domain", default="ffhq")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = get_device()
    alpha, method = cfg.calibration.alpha, cfg.calibration.method
    rng = np.random.default_rng(args.seed)

    # --- load caches -----------------------------------------------------
    tr_sets, tr = _load_streams(cfg, args.train_tag)
    sh_sets, sh = _load_streams(cfg, args.shift_tag)
    specs = [StreamSpec(s.backbone, s.dim) for s in tr_sets]

    # --- split base-domain reals into model-train / cal / test -----------
    base_real_idx = np.flatnonzero((tr.labels == 0) & (tr.domain == args.base_domain))
    rng.shuffle(base_real_idx)
    n = base_real_idx.size
    a, b = int(0.6 * n), int(0.8 * n)
    real_train, real_cal, real_test = (base_real_idx[:a], base_real_idx[a:b],
                                       base_real_idx[b:])
    fake_idx = np.flatnonzero(tr.labels == 1)

    # detector trains on ALL fakes + the model-train slice of base reals;
    # cal/test reals are held out so the FPR estimate is honest.
    train_idx = np.concatenate([fake_idx, real_train])
    y_train = tr.labels[train_idx]

    def streams(sets, idx):
        return [s.features[idx] for s in sets]

    det = FusionDetector(specs, device=device, seed=args.seed)
    det.fit(streams(tr_sets, train_idx), y_train,
            epochs=cfg.train.epochs, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            batch_size=cfg.train.batch_size, verbose=False)

    # --- scores ----------------------------------------------------------
    s_cal = det.score(streams(tr_sets, real_cal))
    s_test = det.score(streams(tr_sets, real_test))

    sh_real_idx = np.flatnonzero(sh.labels == 0)
    rng.shuffle(sh_real_idx)
    h = sh_real_idx.size // 2
    sh_cal_idx, sh_test_idx = sh_real_idx[:h], sh_real_idx[h:]
    s_shift_cal = det.score(streams(sh_sets, sh_cal_idx))
    s_shift_test = det.score(streams(sh_sets, sh_test_idx))

    # --- calibrate + measure ---------------------------------------------
    base = Calibrator.fit(s_cal, alpha, method=method, real_domain=args.base_domain)
    recal = Calibrator.fit(s_shift_cal, alpha, method=method,
                           real_domain=args.shift_domain)

    fpr_in_domain = base.fpr_on(s_test)            # should be ~alpha
    fpr_shift_orig = base.fpr_on(s_shift_test)     # should be >> alpha
    fpr_shift_recal = recal.fpr_on(s_shift_test)   # should be ~alpha (restored)

    report = {
        "alpha": alpha, "method": method, "seed": args.seed,
        "base_domain": args.base_domain, "shift_domain": args.shift_domain,
        "tau_base": base.tau, "tau_recal": recal.tau,
        "base_certified": base.certified, "recal_certified": recal.certified,
        "n_base_cal": int(real_cal.size), "n_base_test": int(real_test.size),
        "n_shift_cal": int(sh_cal_idx.size), "n_shift_test": int(sh_test_idx.size),
        "fpr_in_domain": fpr_in_domain,
        "fpr_shift_original_tau": fpr_shift_orig,
        "fpr_shift_recalibrated": fpr_shift_recal,
        "guarantee_broke": fpr_shift_orig > 2 * alpha,
        "guarantee_restored": fpr_shift_recal <= 2 * alpha,
    }
    out = Path(cfg.out_dir) / "real_domain_shift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    # --- headline --------------------------------------------------------
    print(f"\ntarget alpha = {alpha}   (tau_base={base.tau:.3f} -> "
          f"tau_recal={recal.tau:.3f})")
    print("-" * 58)
    print(f"  in-domain  ({args.base_domain:>9} reals, base tau)   "
          f"FPR = {fpr_in_domain:.3f}")
    print(f"  shifted    ({args.shift_domain:>9} reals, base tau)   "
          f"FPR = {fpr_shift_orig:.3f}   <- guarantee {'BROKE' if report['guarantee_broke'] else 'held'}")
    print(f"  recalibr.  ({args.shift_domain:>9} reals, new  tau)   "
          f"FPR = {fpr_shift_recal:.3f}   <- {'RESTORED' if report['guarantee_restored'] else 'still broken'}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
