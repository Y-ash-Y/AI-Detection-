#!/usr/bin/env python
"""Phase 4 follow-up: how many labeled in-domain fakes does it take to recover power?

06 showed that under real-domain shift, recalibrating tau on unlabeled reals
restores FPR <= alpha but NOT detection power. This script answers the natural
next question: power is recoverable with *labeled* in-domain fakes — how many?

For each budget k (number of labeled new-domain fakes), we:
  - adapt the fusion detector on a balanced mix of base data + new-domain reals +
    k new-domain fakes (the only scarce, expensive ingredient),
  - keep the FPR guarantee enforced by recalibrating tau on HELD-OUT new-domain
    reals (FPR <= alpha for free, abundant),
  - measure recovered TPR on a HELD-OUT new-domain fake test set at that tau.

k=0 is the reals-only baseline (no labeled fakes). The output is a TPR-vs-k curve
with FPR pinned at alpha throughout: the price of power, in labeled examples.

    python scripts/07_few_shot_recovery.py --train-tag train --shift-tag ffhq \
        --shift-fake-tag facesfake --k-list 0,10,25,50,100,250,500,1000
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
from pmsa.eval import roc_auc
from pmsa.utils import get_device


def _load(cfg, tag):
    sets = [FeatureSet.load(Path(cfg.feature.cache_dir) / f"{bb.name}_{tag}.npz")
            for bb in cfg.feature.backbones if bb.enabled]
    base = sets[0]
    for s in sets[1:]:
        assert np.array_equal(s.paths, base.paths), f"{tag} caches not aligned"
    return sets, base


def _resize(idx, n):
    """Tile/truncate an index array to length n (oversampling for balance)."""
    if idx.size == 0 or n == 0:
        return np.empty(0, dtype=int)
    return np.resize(idx, n)


def _stack(groups):
    """groups: list of (sets, idx, label). Returns per-stream arrays + labels."""
    n_streams = len(groups[0][0])
    per_stream = [np.vstack([g[0][si].features[g[1]] for g in groups if g[1].size])
                  for si in range(n_streams)]
    labels = np.concatenate([np.full(g[1].size, g[2]) for g in groups if g[1].size])
    return per_stream, labels


def run_once(cfg, device, tr_sets, tr, sh_sets, sh, fk_sets, fk, k_list, B, seed):
    rng = np.random.default_rng(seed)
    alpha, method = cfg.calibration.alpha, cfg.calibration.method
    specs = [StreamSpec(s.backbone, s.dim) for s in tr_sets]

    # base blocks (subsampled to B per class)
    base_reals = np.flatnonzero(tr.labels == 0)
    base_fakes = np.flatnonzero(tr.labels == 1)
    rng.shuffle(base_reals); rng.shuffle(base_fakes)
    base_reals, base_fakes = base_reals[:B], base_fakes[:B]

    # face reals: adapt / cal / test
    fr = np.flatnonzero(sh.labels == 0); rng.shuffle(fr)
    n = fr.size
    fr_adapt, fr_cal, fr_test = fr[:n // 2], fr[n // 2:3 * n // 4], fr[3 * n // 4:]

    # face fakes: fixed test set + budget pool
    ff = np.flatnonzero(fk.labels == 1); rng.shuffle(ff)
    half = ff.size // 2
    ff_test, ff_budget = ff[:half], ff[half:]

    out = {}
    for k in k_list:
        adapt_fakes = ff_budget[:k]
        groups = [
            (tr_sets, base_reals, 0),
            (tr_sets, base_fakes, 1),
            (sh_sets, _resize(fr_adapt, B), 0),       # abundant face reals
        ]
        if k > 0:
            groups.append((fk_sets, _resize(adapt_fakes, B), 1))  # k unique fakes, oversampled

        Xs, y = _stack(groups)
        det = FusionDetector(specs, device=device, seed=seed)
        det.fit(Xs, y, epochs=cfg.train.epochs, lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                batch_size=cfg.train.batch_size, verbose=False)

        # enforce FPR on held-out face reals, measure recovered TPR on held-out fakes
        cal = Calibrator.fit(det.score([s.features[fr_cal] for s in sh_sets]),
                             alpha, method=method, real_domain="shift")
        fpr = cal.fpr_on(det.score([s.features[fr_test] for s in sh_sets]))
        tpr = cal.tpr_on(det.score([s.features[ff_test] for s in fk_sets]))
        auc = roc_auc(
            np.concatenate([det.score([s.features[fr_test] for s in sh_sets]),
                            det.score([s.features[ff_test] for s in fk_sets])]),
            np.concatenate([np.zeros(fr_test.size), np.ones(ff_test.size)]))
        out[k] = {"tpr": tpr, "fpr": fpr, "auc": auc, "tau": cal.tau,
                  "n_unique_fakes": int(adapt_fakes.size)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-tag", default="train")
    ap.add_argument("--shift-tag", default="ffhq")
    ap.add_argument("--shift-fake-tag", default="facesfake")
    ap.add_argument("--k-list", default="0,10,25,50,100,250,500,1000")
    ap.add_argument("--base-per-class", type=int, default=4000)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = get_device()
    k_list = [int(x) for x in args.k_list.split(",")]
    tr_sets, tr = _load(cfg, args.train_tag)
    sh_sets, sh = _load(cfg, args.shift_tag)
    fk_sets, fk = _load(cfg, args.shift_fake_tag)

    runs = [run_once(cfg, device, tr_sets, tr, sh_sets, sh, fk_sets, fk,
                     k_list, args.base_per_class, seed) for seed in cfg.train.seeds]

    report = {"alpha": cfg.calibration.alpha, "seeds": cfg.train.seeds, "k": {}}
    print(f"\ntarget alpha = {cfg.calibration.alpha}   seeds={cfg.train.seeds}")
    print(f"{'k (labeled fakes)':>18}{'TPR (recovered)':>20}{'FPR':>14}{'face AUC':>12}")
    print("-" * 64)
    for k in k_list:
        tpr = [r[k]["tpr"] for r in runs]
        fpr = [r[k]["fpr"] for r in runs]
        auc = [r[k]["auc"] for r in runs]
        report["k"][k] = {
            "tpr_mean": float(np.mean(tpr)), "tpr_std": float(np.std(tpr)),
            "fpr_mean": float(np.mean(fpr)), "fpr_std": float(np.std(fpr)),
            "auc_mean": float(np.mean(auc)), "auc_std": float(np.std(auc)),
        }
        print(f"{k:>18}{np.mean(tpr):>13.3f}+/-{np.std(tpr):.3f}"
              f"{np.mean(fpr):>9.3f}{np.mean(auc):>12.3f}")

    out = Path(cfg.out_dir) / "few_shot_recovery.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nFPR should stay ~alpha at every k (guarantee held); TPR is the recovery curve.")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
