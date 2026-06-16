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
from pmsa.models import LinearProbe, FusionDetector
from pmsa.calibration import Calibrator
from pmsa.eval import full_report
from pmsa.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--deploy-dir", default="outputs/deploy")
    ap.add_argument("--model", choices=["probe", "fusion"], default="probe")
    ap.add_argument("--backbone", default="siglip", help="probe backbone cache to score")
    ap.add_argument("--test-tag", default="chameleontest")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cache = Path(cfg.feature.cache_dir)
    deploy = Path(args.deploy_dir)

    if args.model == "fusion":
        det = FusionDetector.load(deploy / "fusion.pt", device=get_device())
        sets = [FeatureSet.load(cache / f"{s.name}_{args.test_tag}.npz")
                for s in det.specs]
        labels = sets[0].labels
        scores = det.score([s.features for s in sets])
    else:
        fs = FeatureSet.load(cache / f"{args.backbone}_{args.test_tag}.npz")
        labels = fs.labels
        scores = LinearProbe.load(deploy / "probe.pkl").score(fs.features)

    cal = Calibrator.load(deploy / "calibrator.json")
    rep = full_report(scores, labels, cfg.eval.fpr_targets,
                      cfg.eval.bootstrap_n, cfg.eval.ci, cfg.seed)

    real, fake = scores[labels == 0], scores[labels == 1]
    fpr = float(np.mean(real > cal.tau)) if real.size else float("nan")
    tpr = float(np.mean(fake > cal.tau)) if fake.size else float("nan")
    acc = float(np.mean((scores > cal.tau) == (labels == 1)))

    print(f"\n=== {args.test_tag} [{args.model}] ({len(labels)} imgs: "
          f"{int((labels==0).sum())} real / {int((labels==1).sum())} fake) ===")
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
