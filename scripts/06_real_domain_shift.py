#!/usr/bin/env python
"""Phase 4 core experiment: the FPR guarantee under real-domain shift, and what it
costs in detection power.

The full PMSA v2 claim has two halves:
  (1) the NP false-alarm guarantee (FPR <= alpha) is RECOVERABLE under real-domain
      shift by recalibrating tau on new-domain reals only (no fakes), and
  (2) detection POWER may NOT be recoverable the same way.

This script demonstrates both. It:
  - trains a fusion detector on GenImage (base real domain = imagenet),
  - calibrates tau on HELD-OUT base-domain reals,
  - measures FPR in-domain (sanity), on shifted reals with the base tau (breaks),
    and on shifted reals after real-only recalibration (restored),
  - and, if a new-domain FAKE set is given, measures detection power on the new
    domain at BOTH tau values: raising tau to fix FPR may sacrifice TPR. That gap
    is the "power is not recoverable" half of the thesis.

Runs over multiple seeds (cfg.train.seeds) and reports mean +/- std.

Prereqs (cached features):
  feature_cache/<bb>_<train-tag>.npz   GenImage reals+fakes (base domain)
  feature_cache/<bb>_<shift-tag>.npz   new-domain reals only (e.g. FFHQ)
  feature_cache/<bb>_<shift-fake-tag>.npz   new-domain fakes (optional, e.g. StyleGAN2)

    python scripts/06_real_domain_shift.py --train-tag train --shift-tag ffhq \
        --shift-fake-tag facesfake --base-domain imagenet --shift-domain ffhq
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
from pmsa.eval import roc_auc, bootstrap_ci
from pmsa.utils import get_device


def _load_streams(cfg, tag):
    sets = [FeatureSet.load(Path(cfg.feature.cache_dir) / f"{bb.name}_{tag}.npz")
            for bb in cfg.feature.backbones if bb.enabled]
    base = sets[0]
    for s in sets[1:]:
        assert np.array_equal(s.paths, base.paths), f"{tag} caches not aligned"
    return sets, base


def _streams(sets, idx):
    return [s.features[idx] for s in sets]


def run_once(cfg, device, tr_sets, tr, sh_sets, sh, fk_sets, args, seed):
    rng = np.random.default_rng(seed)
    alpha, method = cfg.calibration.alpha, cfg.calibration.method

    # split base-domain reals: model-train / cal / test
    base_real_idx = np.flatnonzero((tr.labels == 0) & (tr.domain == args.base_domain))
    rng.shuffle(base_real_idx)
    n = base_real_idx.size
    a, b = int(0.6 * n), int(0.8 * n)
    real_train, real_cal, real_test = base_real_idx[:a], base_real_idx[a:b], base_real_idx[b:]
    fake_idx = np.flatnonzero(tr.labels == 1)

    train_idx = np.concatenate([fake_idx, real_train])
    det = FusionDetector([StreamSpec(s.backbone, s.dim) for s in tr_sets],
                         device=device, seed=seed)
    det.fit(_streams(tr_sets, train_idx), tr.labels[train_idx],
            epochs=cfg.train.epochs, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            batch_size=cfg.train.batch_size, verbose=False)

    s_cal = det.score(_streams(tr_sets, real_cal))
    s_test = det.score(_streams(tr_sets, real_test))
    s_fake_base = det.score(_streams(tr_sets, fake_idx))  # base-domain fakes (power ref)

    sh_real = np.flatnonzero(sh.labels == 0)
    rng.shuffle(sh_real)
    h = sh_real.size // 2
    s_shift_cal = det.score(_streams(sh_sets, sh_real[:h]))
    s_shift_test = det.score(_streams(sh_sets, sh_real[h:]))

    base = Calibrator.fit(s_cal, alpha, method=method, real_domain=args.base_domain)
    recal = Calibrator.fit(s_shift_cal, alpha, method=method, real_domain=args.shift_domain)

    out = {
        "tau_base": base.tau, "tau_recal": recal.tau,
        "recal_certified": recal.certified,
        "fpr_in_domain": base.fpr_on(s_test),
        "fpr_shift_original_tau": base.fpr_on(s_shift_test),
        "fpr_shift_recalibrated": recal.fpr_on(s_shift_test),
        "tpr_in_domain": float(np.mean(s_fake_base > base.tau)),  # base power @ base tau
    }

    # --- power on the NEW domain (if fakes provided) ---------------------
    if fk_sets is not None:
        fk = fk_sets[0][1]  # FeatureSet for labels (all fakes)
        fk_streams_sets = fk_sets[1]
        fake_idx_face = np.flatnonzero(fk.labels == 1)
        s_face_fake = det.score(_streams(fk_streams_sets, fake_idx_face))
        out["tpr_face_base_tau"] = float(np.mean(s_face_fake > base.tau))
        out["tpr_face_recal_tau"] = float(np.mean(s_face_fake > recal.tau))
        # separability on faces: real (ffhq test) vs fake (faces)
        face_scores = np.concatenate([s_shift_test, s_face_fake])
        face_labels = np.concatenate([np.zeros(s_shift_test.size),
                                      np.ones(s_face_fake.size)])
        out["face_auc"] = roc_auc(face_scores, face_labels)
        out["_face_scores"] = face_scores      # for a single-seed bootstrap CI
        out["_face_labels"] = face_labels
    return out


def _agg(runs, key):
    vals = [r[key] for r in runs]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-tag", default="train")
    ap.add_argument("--shift-tag", default="ffhq")
    ap.add_argument("--shift-fake-tag", default=None,
                    help="cache of new-domain fakes (e.g. facesfake); enables power eval")
    ap.add_argument("--base-domain", default="imagenet")
    ap.add_argument("--shift-domain", default="ffhq")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = get_device()
    tr_sets, tr = _load_streams(cfg, args.train_tag)
    sh_sets, sh = _load_streams(cfg, args.shift_tag)
    fk_sets = None
    if args.shift_fake_tag:
        fk_streams, fk = _load_streams(cfg, args.shift_fake_tag)
        fk_sets = ((None, fk), fk_streams)

    runs = [run_once(cfg, device, tr_sets, tr, sh_sets, sh, fk_sets, args, seed)
            for seed in cfg.train.seeds]

    alpha = cfg.calibration.alpha
    report = {"alpha": alpha, "seeds": cfg.train.seeds,
              "base_domain": args.base_domain, "shift_domain": args.shift_domain}
    for k in ["tau_base", "tau_recal", "fpr_in_domain", "fpr_shift_original_tau",
              "fpr_shift_recalibrated", "tpr_in_domain"]:
        m, s = _agg(runs, k)
        report[k] = {"mean": m, "std": s}
    if fk_sets is not None:
        for k in ["face_auc", "tpr_face_base_tau", "tpr_face_recal_tau"]:
            m, s = _agg(runs, k)
            report[k] = {"mean": m, "std": s}
        ci = bootstrap_ci(roc_auc, runs[0]["_face_scores"], runs[0]["_face_labels"],
                          cfg.eval.bootstrap_n, cfg.eval.ci, cfg.seed)
        report["face_auc_ci_seed0"] = {"point": ci.point, "lo": ci.lo, "hi": ci.hi}

    out_path = Path(cfg.out_dir) / "real_domain_shift.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    _print(report, fk_sets is not None)
    print(f"\nsaved -> {out_path}")


def _print(r, have_fakes):
    def ms(k):
        return f"{r[k]['mean']:.3f}+/-{r[k]['std']:.3f}"
    a = r["alpha"]
    print(f"\ntarget alpha = {a}   seeds={r['seeds']}")
    print(f"tau: base {ms('tau_base')}  ->  recal {ms('tau_recal')}")
    print("-" * 64)
    print("FALSE-ALARM GUARANTEE (FPR)")
    print(f"  in-domain   ({r['base_domain']:>9} reals, base tau)  {ms('fpr_in_domain')}")
    print(f"  shifted     ({r['shift_domain']:>9} reals, base tau)  {ms('fpr_shift_original_tau')}  <- breaks")
    print(f"  recalibr.   ({r['shift_domain']:>9} reals, new  tau)  {ms('fpr_shift_recalibrated')}  <- restored")
    if have_fakes:
        print("\nDETECTION POWER (TPR)")
        print(f"  in-domain   ({r['base_domain']:>9} fakes, base tau)  {ms('tpr_in_domain')}")
        print(f"  new domain  ({r['shift_domain']:>9} fakes, base tau)  {ms('tpr_face_base_tau')}")
        print(f"  new domain  ({r['shift_domain']:>9} fakes, recal tau) {ms('tpr_face_recal_tau')}  <- power after FPR fix")
        print(f"  separability (face AUC)                    {ms('face_auc')}")
        drop = r['tpr_face_base_tau']['mean'] - r['tpr_face_recal_tau']['mean']
        print(f"\n  => fixing FPR on the new domain costs {drop:+.3f} TPR there.")


if __name__ == "__main__":
    main()
