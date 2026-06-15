#!/usr/bin/env python
"""Train the single DEPLOYMENT detector for the demo app (not LOGO — uses all data).

Trains the fusion detector on every generator + reals in the cached train features,
fits the calibrator on held-out reals, and saves both to outputs/deploy/. Runs on
the laptop (MPS/CPU) in minutes from cached npz — no GPU extraction needed.

    python scripts/train_deploy.py --tag train --alpha 0.05
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# When running this script directly, Python's sys.path[0] is the `scripts/`
# directory which prevents importing the top-level `pmsa` package. Ensure the
# project root is on `sys.path` so `import pmsa` works.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import FusionDetector, StreamSpec
from pmsa.calibration import Calibrator
from pmsa.eval import full_report
from pmsa.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="train")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="target false-alarm rate for the deployed threshold")
    ap.add_argument("--out", default="outputs/deploy")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = get_device()
    cache = Path(cfg.feature.cache_dir)
    sets = [FeatureSet.load(cache / f"{bb.name}_{args.tag}.npz")
            for bb in cfg.feature.backbones if bb.enabled]
    base = sets[0]
    for s in sets[1:]:
        assert np.array_equal(s.paths, base.paths), "caches not aligned"
    specs = [StreamSpec(s.backbone, s.dim) for s in sets]

    # split: train / val (early stop) / calibration (real-only) / test
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(base))
    n = len(idx)
    tr, va, te = idx[:int(.7 * n)], idx[int(.7 * n):int(.8 * n)], idx[int(.8 * n):]

    def streams(ix):
        return [s.features[ix] for s in sets]

    det = FusionDetector(specs, device=device, seed=cfg.seed)
    det.fit(streams(tr), base.labels[tr], streams(va), base.labels[va],
            epochs=cfg.train.epochs, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            batch_size=cfg.train.batch_size, verbose=True)

    # honest held-out report
    te_scores = det.score(streams(te))
    rep = full_report(te_scores, base.labels[te], cfg.eval.fpr_targets,
                      cfg.eval.bootstrap_n, cfg.eval.ci, cfg.seed)
    print(f"\nheld-out: AUC={rep['auc']['point']:.4f} "
          f"TPR@1%={rep['tpr_at_0.01']['point']:.4f} "
          f"TPR@5%={rep['tpr_at_0.05']['point']:.4f}")

    # calibrate threshold on held-out REAL scores
    real_te = te[base.labels[te] == 0]
    cal = Calibrator.fit(det.score(streams(real_te)), args.alpha,
                         method=cfg.calibration.method, real_domain="mixed")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det.save(out / "fusion.pt")
    cal.save(out / "calibrator.json")
    print(f"\nsaved detector -> {out}/fusion.pt")
    print(f"saved calibrator (alpha={args.alpha}, tau={cal.tau:.3f}) -> {out}/calibrator.json")
    print("ready for: python app.py")


if __name__ == "__main__":
    main()
