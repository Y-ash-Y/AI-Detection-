#!/usr/bin/env python
"""Phase 3: train the PMSA v2 fusion detector (CLIP + DINOv2 + NPR), LOGO protocol.

Loads the aligned per-backbone caches, trains the fusion net on the laptop (MPS),
and reports metrics + per-stream decomposition on the held-out generator.

    python scripts/03_train_fusion.py --tag train --held-out sdxl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import FusionDetector, StreamSpec
from pmsa.eval import full_report
from pmsa.utils import get_device


def _load_streams(cfg, tag):
    sets = []
    for bb in cfg.feature.backbones:
        if not bb.enabled:
            continue
        sets.append(FeatureSet.load(
            Path(cfg.feature.cache_dir) / f"{bb.name}_{tag}.npz"))
    # alignment check
    base = sets[0]
    for s in sets[1:]:
        assert np.array_equal(s.paths, base.paths), "caches not aligned"
    return sets, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="train")
    ap.add_argument("--held-out", required=True)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = get_device()
    sets, base = _load_streams(cfg, args.tag)
    specs = [StreamSpec(s.backbone, s.dim) for s in sets]

    g = args.held_out
    train_mask = (base.labels == 0) | (base.source != g)
    test_mask = (base.labels == 0) | (base.source == g)

    def streams(mask):
        return [s.features[mask] for s in sets]

    ytr = base.labels[train_mask]
    yte = base.labels[test_mask]

    agg = {}
    for seed in cfg.train.seeds:
        det = FusionDetector(specs, device=device, seed=seed)
        det.fit(streams(train_mask), ytr,
                epochs=cfg.train.epochs, lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                batch_size=cfg.train.batch_size, verbose=False)
        scores = det.score(streams(test_mask))
        rep = full_report(scores, yte, cfg.eval.fpr_targets,
                          cfg.eval.bootstrap_n, cfg.eval.ci, seed)
        agg[seed] = rep
        print(f"[seed {seed}] AUC={rep['auc']['point']:.4f} "
              f"TPR@1%={rep['tpr_at_0.01']['point']:.4f}")
        det.save(Path(cfg.out_dir) / f"fusion_logo_{g}_seed{seed}.pt")

    out = Path(cfg.out_dir) / f"fusion_logo_{g}.json"
    out.write_text(json.dumps(agg, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
