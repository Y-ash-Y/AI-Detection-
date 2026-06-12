#!/usr/bin/env python
"""Bridge between a trained detector and the shift experiment.

Runs a detector (baseline or fusion) over a feature cache, writes a scores npz
(score/label/domain/source) that 05_shift_experiment.py consumes, and prints a
single calibrated operating point on the real-only calibration split.

    python scripts/04_score_and_calibrate.py \
        --model fusion --checkpoint outputs/fusion_logo_sdxl_seed0.pt \
        --tag test --base-domain imagenet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import LinearProbe, FusionDetector
from pmsa.calibration import Calibrator
from pmsa.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["baseline", "fusion"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="test")
    ap.add_argument("--backbone", default="clip_l14", help="baseline only")
    ap.add_argument("--base-domain", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cache = Path(cfg.feature.cache_dir)

    if args.model == "baseline":
        fs = FeatureSet.load(cache / f"{args.backbone}_{args.tag}.npz")
        probe = LinearProbe.load(args.checkpoint)
        score = probe.score(fs.features)
        base = fs
    else:
        sets = [FeatureSet.load(cache / f"{bb.name}_{args.tag}.npz")
                for bb in cfg.feature.backbones if bb.enabled]
        det = FusionDetector.load(args.checkpoint, device=get_device())
        score = det.score([s.features for s in sets])
        base = sets[0]

    out = Path(args.out or cfg.out_dir) / f"scores_{args.model}_{args.tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, score=score.astype(np.float32),
                        label=base.labels, domain=base.domain, source=base.source)

    # quick calibrated operating point on the base domain's reals
    base_real = score[(base.labels == 0) & (base.domain == args.base_domain)]
    cal = Calibrator.fit(base_real, cfg.calibration.alpha,
                         method=cfg.calibration.method, real_domain=args.base_domain)
    print(f"scores -> {out}")
    print(f"tau={cal.tau:.4f} alpha={cal.alpha} method={cal.method} "
          f"certified={cal.certified} n_cal={cal.n_cal}")


if __name__ == "__main__":
    main()
