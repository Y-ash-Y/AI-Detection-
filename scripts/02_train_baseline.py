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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import LinearProbe
from pmsa.eval import full_report
from pmsa.data.splits import logo_split, independent_indices


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
    fs = fs.select(independent_indices(fs.labels, fs.groups()))
    generators = sorted(set(fs.source[fs.labels == 1]))
    held = [args.held_out] if args.held_out else generators

    out = Path(cfg.out_dir) / f"baseline_{args.backbone}_logo_v3.json"
    targets = [out, *[Path(cfg.out_dir) / f"baseline_{args.backbone}_logo_v3_{g}.pkl" for g in held]]
    if any(p.exists() for p in targets):
        raise FileExistsError("LOGO outputs exist; choose a new out_dir in the config")
    if any(Path(g).name != g or g in (".", "..") for g in held):
        raise ValueError("Generator names must be filename-safe")

    results = {}
    for g in held:
        train_mask, test_mask = logo_split(fs.labels, fs.source, fs.groups(), g, cfg.seed)
        Xtr, ytr = fs.features[train_mask], fs.labels[train_mask]
        Xte, yte = fs.features[test_mask], fs.labels[test_mask]

        probe = LinearProbe(seed=cfg.seed).fit(Xtr, ytr)
        scores = probe.score(Xte)
        rep = full_report(scores, yte, cfg.eval.fpr_targets,
                          cfg.eval.bootstrap_n, cfg.eval.ci, cfg.seed)
        rep["protocol"] = "disjoint-logo-v3"
        rep["exploratory_legacy"] = fs.content_hashes is None
        rep["train_paths"] = fs.paths[train_mask].tolist()
        rep["test_paths"] = fs.paths[test_mask].tolist()
        results[g] = rep
        print(f"[LOGO held-out={g}] AUC={rep['auc']['point']:.4f} "
              f"TPR@1%={rep['tpr_at_0.01']['point']:.4f}")
        probe.save(Path(cfg.out_dir) / f"baseline_{args.backbone}_logo_v3_{g}.pkl")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
