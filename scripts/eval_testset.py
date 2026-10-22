#!/usr/bin/env python
"""Evaluate external caches with a frozen model and threshold; never fit on test."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.artifacts import verify_bundle
from pmsa.config import Config
from pmsa.features import FeatureSet
from pmsa.models import LinearProbe, FusionDetector
from pmsa.calibration import Calibrator
from pmsa.evaluation import evaluate_cached


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--deploy-dir", default="outputs/deploy_v3")
    ap.add_argument("--model", choices=["probe", "fusion"], default="fusion",
                    help="Legacy only; v3 bundle supplies model type and backbone order")
    ap.add_argument("--backbone", default="siglip", help="Legacy probe only")
    ap.add_argument("--test-tag", default="chameleontest_v3")
    ap.add_argument("--out", required=True, help="New directory for report and predictions")
    ap.add_argument("--exploratory-legacy", action="store_true")
    ap.add_argument("--trusted-legacy", action="store_true",
                    help="Allow unrestricted pickle for a TRUSTED old fusion checkpoint")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    deploy, out = Path(args.deploy_dir), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Evaluation output exists; choose a new --out")
    bundle = verify_bundle(deploy) if (deploy / "bundle.json").exists() else None
    if not bundle and not args.exploratory_legacy:
        raise ValueError("Missing artifact bundle; legacy evaluation must be explicitly exploratory")
    kind = bundle["model_type"] if bundle else args.model
    if kind == "fusion":
        model = FusionDetector.load(deploy / "fusion.pt", device="cpu",
                                    trusted_legacy=args.trusted_legacy and args.exploratory_legacy)
        names = [s.name for s in model.specs]
    else:
        model = LinearProbe.load(deploy / "probe.pkl")  # Only trusted joblib files.
        names = list(bundle["streams"]) if bundle else [args.backbone]
    sets = [FeatureSet.load(Path(cfg.feature.cache_dir) / f"{name}_{args.test_tag}.npz") for name in names]
    splits = json.loads((deploy / "splits.json").read_text()) if (deploy / "splits.json").exists() else {}
    report, scores, labels, paths = evaluate_cached(
        model, sets, Calibrator.load(deploy / "calibrator.json"), bundle,
        splits.get("development_ids", {}), exploratory_legacy=args.exploratory_legacy,
        bootstrap_n=cfg.eval.bootstrap_n, ci=cfg.eval.ci, seed=cfg.seed,
        fpr_targets=cfg.eval.fpr_targets)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(out / "predictions.npz", scores=scores, labels=labels, paths=paths)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
