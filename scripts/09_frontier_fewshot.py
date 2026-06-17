#!/usr/bin/env python
"""Frontier few-shot: how well does the detector handle a BRAND-NEW generator
(e.g. GPT-Image) that was never in training — zero-shot, then after k examples?

For each budget k (number of labeled new-generator fakes), adapt the fusion
detector on base diverse data + k new fakes, keep the false-alarm rate pinned by
calibrating tau on held-out base reals, and measure detection on a HELD-OUT set of
new-generator fakes (disjoint from the k used to adapt).

  k=0  = zero-shot: the new generator is unseen -> the honest baseline.
  k>0  = few-shot:  detection recovers as you add a handful of examples.

The new generator's images need no matched reals — we reuse the base real class.
Prereqs (cached features for each backbone):
  feature_cache/<bb>_<train-tag>.npz       base diverse data (reals + fakes)
  feature_cache/<bb>_<new-fake-tag>.npz    new-generator fakes only

    python scripts/09_frontier_fewshot.py --train-tag diversetrain \
        --new-fake-tag gptimage --backbones siglip forensic \
        --k-list 0,10,25,50 --new-name GPT-Image
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import FusionDetector, StreamSpec
from pmsa.calibration import Calibrator
from pmsa.eval import roc_auc
from pmsa.utils import get_device


def _load(cfg, names, tag):
    sets = [FeatureSet.load(Path(cfg.feature.cache_dir) / f"{n}_{tag}.npz") for n in names]
    for s in sets[1:]:
        assert np.array_equal(s.paths, sets[0].paths), f"{tag} caches not aligned"
    return sets


def _resize(idx, n):
    if idx.size == 0 or n == 0:
        return np.empty(0, dtype=int)
    return np.resize(idx, n)


def _stack(groups):
    n_streams = len(groups[0][0])
    per_stream = [np.vstack([g[0][si].features[g[1]] for g in groups if g[1].size])
                  for si in range(n_streams)]
    labels = np.concatenate([np.full(g[1].size, g[2]) for g in groups if g[1].size])
    return per_stream, labels


def run_once(cfg, device, base_sets, base, new_sets, k_list, B, seed):
    rng = np.random.default_rng(seed)
    alpha, method = cfg.calibration.alpha, cfg.calibration.method
    specs = [StreamSpec(s.backbone, s.dim) for s in base_sets]

    base_reals = np.flatnonzero(base.labels == 0)
    base_fakes = np.flatnonzero(base.labels == 1)
    rng.shuffle(base_reals); rng.shuffle(base_fakes)
    # base reals: model-train / cal / test
    a, b = int(.6 * base_reals.size), int(.8 * base_reals.size)
    real_train, real_cal, real_test = base_reals[:a], base_reals[a:b], base_reals[b:]
    base_train_fakes = base_fakes[:B]

    # new-generator fakes: held-out test + adaptation budget pool (disjoint)
    new_all = np.arange(len(new_sets[0]))
    rng.shuffle(new_all)
    half = new_all.size // 2
    new_test, new_budget = new_all[:half], new_all[half:]

    out = {}
    for k in k_list:
        groups = [
            (base_sets, _resize(real_train, B), 0),
            (base_sets, base_train_fakes, 1),
        ]
        if k > 0:
            groups.append((new_sets, _resize(new_budget[:k], B), 1))
        Xs, y = _stack(groups)
        det = FusionDetector(specs, device=device, seed=seed)
        det.fit(Xs, y, epochs=cfg.train.epochs, lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                batch_size=cfg.train.batch_size, verbose=False)

        cal = Calibrator.fit(det.score([s.features[real_cal] for s in base_sets]),
                             alpha, method=method, real_domain="base")
        real_s = det.score([s.features[real_test] for s in base_sets])
        new_s = det.score([s.features[new_test] for s in new_sets])
        out[k] = {
            "tpr": float(np.mean(new_s > cal.tau)),
            "fpr": float(np.mean(real_s > cal.tau)),
            "auc": roc_auc(np.concatenate([real_s, new_s]),
                           np.concatenate([np.zeros(real_s.size), np.ones(new_s.size)])),
            "n_unique": int(min(k, new_budget.size)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-tag", default="diversetrain")
    ap.add_argument("--new-fake-tag", default="gptimage")
    ap.add_argument("--backbones", nargs="*", default=["siglip", "forensic"])
    ap.add_argument("--k-list", default="0,10,25,50")
    ap.add_argument("--base-per-class", type=int, default=4000)
    ap.add_argument("--new-name", default="new-generator")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = get_device()
    base_sets = _load(cfg, args.backbones, args.train_tag)
    new_sets = _load(cfg, args.backbones, args.new_fake_tag)
    base = base_sets[0]
    k_list = [int(x) for x in args.k_list.split(",")]

    runs = [run_once(cfg, device, base_sets, base, new_sets, k_list,
                     args.base_per_class, seed) for seed in cfg.train.seeds]

    report = {"alpha": cfg.calibration.alpha, "seeds": cfg.train.seeds,
              "new_generator": args.new_name, "k": {}}
    print(f"\n=== few-shot adaptation to {args.new_name} "
          f"({len(new_sets[0])} fakes) | alpha={cfg.calibration.alpha} seeds={cfg.train.seeds} ===")
    print(f"{'k (labeled)':>12}{'detection (TPR)':>18}{'FPR':>9}{'AUC':>9}")
    print("-" * 48)
    for k in k_list:
        tpr = [r[k]["tpr"] for r in runs]
        fpr = [r[k]["fpr"] for r in runs]
        auc = [r[k]["auc"] for r in runs]
        report["k"][k] = {"tpr_mean": float(np.mean(tpr)), "tpr_std": float(np.std(tpr)),
                          "fpr_mean": float(np.mean(fpr)), "auc_mean": float(np.mean(auc))}
        tag = "  <- zero-shot" if k == 0 else ""
        print(f"{k:>12}{np.mean(tpr):>11.3f}+/-{np.std(tpr):.3f}"
              f"{np.mean(fpr):>9.3f}{np.mean(auc):>9.3f}{tag}")

    out = Path(cfg.out_dir) / f"frontier_fewshot_{args.new_fake_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nFPR stays ~alpha at every k; TPR is the recovery from zero-shot.")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
