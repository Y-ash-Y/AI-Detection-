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
    ap.add_argument("--model", choices=["probe", "fusion"], default="probe",
                    help="probe = robust single-backbone linear probe (recommended); "
                         "fusion = CLIP+DINOv2+NPR (stronger in-dist, more fragile)")
    ap.add_argument("--backbone", default="clip_l14",
                    help="probe-mode backbone cache to train on (e.g. clip_l14, siglip)")
    ap.add_argument("--backbones", nargs="*", default=None,
                    help="fusion-mode backbone caches (e.g. siglip forensic)")
    ap.add_argument("--out", default="outputs/deploy")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = get_device()
    cache = Path(cfg.feature.cache_dir)

    # split indices on the probe backbone's cache (all caches are aligned)
    probe_fs = FeatureSet.load(cache / f"{args.backbone}_{args.tag}.npz")
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(probe_fs))
    n = len(idx)
    tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    labels = probe_fs.labels
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.model == "probe":
        from pmsa.models import LinearProbe
        probe = LinearProbe(seed=cfg.seed).fit(probe_fs.features[tr], labels[tr])
        te_scores = probe.score(probe_fs.features[te])
        real_scores = probe.score(probe_fs.features[te][labels[te] == 0])
        probe.save(out / "probe.pkl")
        (out / "probe_backbone.txt").write_text(args.backbone)  # remember for inference
        saved = out / "probe.pkl"
    else:
        names = args.backbones or [b.name for b in cfg.feature.backbones if b.enabled]
        sets = [FeatureSet.load(cache / f"{name}_{args.tag}.npz") for name in names]
        for s in sets[1:]:
            assert np.array_equal(s.paths, sets[0].paths), "caches not aligned"
        specs = [StreamSpec(s.backbone, s.dim) for s in sets]
        va = tr[int(.875 * len(tr)):]; tr2 = tr[:int(.875 * len(tr))]

        def streams(ix):
            return [s.features[ix] for s in sets]

        det = FusionDetector(specs, device=device, seed=cfg.seed)
        det.fit(streams(tr2), labels[tr2], streams(va), labels[va],
                epochs=cfg.train.epochs, lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                batch_size=cfg.train.batch_size, verbose=True)
        te_scores = det.score(streams(te))
        real_scores = det.score([s.features[te][labels[te] == 0] for s in sets])
        det.save(out / "fusion.pt")
        saved = out / "fusion.pt"

    rep = full_report(te_scores, labels[te], cfg.eval.fpr_targets,
                      cfg.eval.bootstrap_n, cfg.eval.ci, cfg.seed)
    print(f"\n[{args.model}] held-out: AUC={rep['auc']['point']:.4f} "
          f"TPR@1%={rep['tpr_at_0.01']['point']:.4f} "
          f"TPR@5%={rep['tpr_at_0.05']['point']:.4f}")

    cal = Calibrator.fit(real_scores, args.alpha,
                         method=cfg.calibration.method, real_domain="mixed")
    cal.save(out / "calibrator.json")
    print(f"saved model -> {saved}")
    print(f"saved calibrator (alpha={args.alpha}, tau={cal.tau:.3f}) -> {out}/calibrator.json")
    print(f"ready for: PMSA_CKPT={saved} python app.py")


if __name__ == "__main__":
    main()
