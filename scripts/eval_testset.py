#!/usr/bin/env python
"""Honest evaluation of a trained model on a held-out test set (e.g. Chameleon).

Scores cached test features with your deployed model + calibrator and reports the
numbers that matter: AUC, TPR@FPR (with bootstrap CIs), and — at your deployed
threshold — the realized false-alarm rate and detection rate. This is the
in-the-wild credibility number; expect ~65-75% accuracy on Chameleon (SOTA range).

    python scripts/eval_testset.py --backbone siglip --test-tag chameleontest
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import LinearProbe
from pmsa.calibration import Calibrator
from pmsa.eval import full_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--deploy-dir", default="outputs/deploy")
    ap.add_argument("--backbone", default="siglip", help="probe backbone cache to score")
    ap.add_argument("--test-tag", default="chameleontest")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cache = Path(cfg.feature.cache_dir)
    fs = FeatureSet.load(cache / f"{args.backbone}_{args.test_tag}.npz")
    probe = LinearProbe.load(Path(args.deploy_dir) / "probe.pkl")
    cal = Calibrator.load(Path(args.deploy_dir) / "calibrator.json")

    scores = probe.score(fs.features)
    rep = full_report(scores, fs.labels, cfg.eval.fpr_targets,
                      cfg.eval.bootstrap_n, cfg.eval.ci, cfg.seed)

    real, fake = scores[fs.labels == 0], scores[fs.labels == 1]
    fpr = float(np.mean(real > cal.tau)) if real.size else float("nan")
    tpr = float(np.mean(fake > cal.tau)) if fake.size else float("nan")
    acc = float(np.mean((scores > cal.tau) == (fs.labels == 1)))

    print(f"\n=== {args.test_tag} ({len(fs)} imgs: {int((fs.labels==0).sum())} real / "
          f"{int((fs.labels==1).sum())} fake) ===")
    print(f"AUC        {rep['auc']['point']:.4f} "
          f"[{rep['auc']['lo']:.3f}, {rep['auc']['hi']:.3f}]")
    print(f"TPR@1%FPR  {rep['tpr_at_0.01']['point']:.4f}")
    print(f"TPR@5%FPR  {rep['tpr_at_0.05']['point']:.4f}")
    print(f"\nAt deployed tau={cal.tau:.3f} (alpha={cal.alpha}):")
    print(f"  realized FPR (false alarms on real) {fpr:.4f}")
    print(f"  detection rate (TPR on fake)        {tpr:.4f}")
    print(f"  accuracy                            {acc:.4f}")
    print("\n(in-the-wild ~65-75% AUC = SOTA range = success; report this honestly)")


if __name__ == "__main__":
    main()
