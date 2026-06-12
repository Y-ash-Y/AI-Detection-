#!/usr/bin/env python
"""Phase 2: the honest baseline. UnivFD linear probe on frozen CLIP, LOGO protocol.

This number is the floor. Reports AUC / EER / TPR@FPR with bootstrap CIs for each
held-out generator. Runs in seconds on cached features.

    python scripts/02_train_baseline.py --tag train --held-out sdxl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import LinearProbe
from pmsa.eval import full_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backbone", default="clip_l14")
    ap.add_argument("--tag", default="train")
    ap.add_argument("--held-out", default=None,
                    help="generator to leave out; if omitted, loop over all")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    fs = FeatureSet.load(Path(cfg.feature.cache_dir) / f"{args.backbone}_{args.tag}.npz")
    generators = sorted(set(fs.source[fs.labels == 1]))
    held = [args.held_out] if args.held_out else generators

    results = {}
    for g in held:
        train_mask = (fs.labels == 0) | (fs.source != g)
        test_mask = (fs.labels == 0) | (fs.source == g)
        Xtr, ytr = fs.features[train_mask], fs.labels[train_mask]
        Xte, yte = fs.features[test_mask], fs.labels[test_mask]

        probe = LinearProbe(seed=cfg.seed).fit(Xtr, ytr)
        scores = probe.score(Xte)
        rep = full_report(scores, yte, cfg.eval.fpr_targets,
                          cfg.eval.bootstrap_n, cfg.eval.ci, cfg.seed)
        results[g] = rep
        print(f"[LOGO held-out={g}] AUC={rep['auc']['point']:.4f} "
              f"TPR@1%={rep['tpr_at_0.01']['point']:.4f}")
        probe.save(Path(cfg.out_dir) / f"baseline_{args.backbone}_logo_{g}.pkl")

    out = Path(cfg.out_dir) / f"baseline_{args.backbone}_logo.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
