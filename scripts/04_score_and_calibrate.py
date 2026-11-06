#!/usr/bin/env python
"""Bridge between a trained detector and the shift experiment.

Runs a detector over aligned feature caches and writes grouped scores for
05_shift_experiment.py. Any threshold printed here is an in-sample diagnostic.

    python scripts/04_score_and_calibrate.py \
        --model fusion --checkpoint outputs/fusion_logo_sdxl_seed0.pt \
        --tag test --base-domain imagenet
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.features.cache import validate_alignment
from pmsa.models import LinearProbe, FusionDetector
from pmsa.calibration import Calibrator
from pmsa.utils import get_device


def score_cache(cache, tag, checkpoint, model_type, backbone):
    if model_type == "baseline":
        fs = FeatureSet.load(cache / f"{backbone}_{tag}.npz")
        if fs.backbone != backbone:
            raise ValueError("Cache backbone name does not match requested probe stream")
        probe = LinearProbe.load(checkpoint)
        return fs, probe.score(fs.features)
    det = FusionDetector.load(checkpoint, device=get_device())
    names = [spec.name for spec in det.specs]
    sets = [FeatureSet.load(cache / f"{name}_{tag}.npz") for name in names]
    if [fs.backbone for fs in sets] != names:
        raise ValueError("Cache backbone names do not match checkpoint stream order")
    validate_alignment(sets)
    return sets[0], det.score([fs.features for fs in sets])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["baseline", "fusion"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="test")
    ap.add_argument("--backbone", choices=["siglip", "forensic"], default="siglip", help="baseline only")
    ap.add_argument("--base-domain", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cache = Path(cfg.feature.cache_dir)

    base, score = score_cache(cache, args.tag, args.checkpoint, args.model, args.backbone)
    if base.content_hashes is None or np.any(base.content_hashes == ""):
        raise ValueError("Shift scores require image identities; re-extract with the current cache format")
    groups = base.groups()
    base_real = score[(base.labels == 0) & (base.domain == args.base_domain)]
    cal = Calibrator.fit(base_real, cfg.calibration.alpha,
                         method=cfg.calibration.method, real_domain=args.base_domain)
    out = Path(args.out or cfg.out_dir) / f"scores_{args.model}_{args.tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as f:
        np.savez_compressed(f, score=score, label=base.labels, domain=base.domain,
                            source=base.source, groups=groups)
    print(f"scores -> {out}")
    print(f"In-sample reference only: tau={cal.tau:.4f} alpha={cal.alpha} "
          f"method={cal.method} n_rows={cal.n_cal}. Use script 05 for held-out rates.")


if __name__ == "__main__":
    main()
