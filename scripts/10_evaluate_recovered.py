#!/usr/bin/env python
"""Predeclared local comparison from recovered, legacy real-data feature caches.

No backbone downloads or fine-tuning. Three seeds, fit/validation/calibration/test
group separation, shared ablation splits, and frozen-threshold external evaluation.
Every result remains exploratory because extraction/hash provenance is absent.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.artifacts import sha256_file, verify_bundle, validate_stream_metadata
from pmsa.calibration import Calibrator
from pmsa.config import Config
from pmsa.data.splits import independent_indices
from pmsa.evaluation import evaluate_cached, check_external_overlap
from pmsa.features import FeatureSet
from pmsa.features.cache import validate_alignment
from pmsa.models import FusionDetector, LinearProbe
from pmsa.training import train


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def feature_fingerprints(fs):
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in fs.features]


def audit_sets(sets, tag):
    validate_alignment(sets)
    base = sets[0]
    ix = independent_indices(base.labels, base.groups())
    return {"tag": tag, "original_rows": len(base), "representatives": len(ix),
            "removed_feature_identity_repeats": len(base)-len(ix),
            "original_labels": {str(k): v for k, v in Counter(map(int, base.labels)).items()},
            "representative_labels": {str(k): v for k, v in Counter(map(int, base.labels[ix])).items()},
            "by_domain_label": {f"{d}:{label}": count for (d, label), count in
                                Counter(zip(map(str, base.domain), map(int, base.labels))).items()},
            "source_counts": dict(Counter(map(str, base.source))),
            "unique_paths": len(set(base.paths)),
            "streams": {s.backbone: {"shape": list(s.features.shape), "dtype": str(s.features.dtype),
                                     "finite": bool(np.isfinite(s.features).all()),
                                     "extraction_metadata_present": bool(s.metadata),
                                     "image_hashes_present": s.content_hashes is not None}
                        for s in sets}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--require-metadata", action="store_true",
                    help="Fresh-cache mode: reject legacy data and verify complete bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-tag", default="diversetrain")
    ap.add_argument("--test-tag", default="chameleontest")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--models", default="siglip_probe,forensic_probe,fusion")
    ap.add_argument("--reason", default="Initial predeclared comparison")
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache_dir)
    if out.exists():
        raise FileExistsError("Evaluation directory exists; choose a new --out")
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or min(args.epochs, args.bootstrap, args.cpu_threads) < 1:
        raise ValueError("Distinct seeds and positive run settings required")
    import torch
    torch.set_num_threads(args.cpu_threads)
    names = ["siglip", "forensic"]
    files = [cache / f"{bb}_{tag}.npz" for tag in (args.train_tag, args.test_tag) for bb in names]
    before = {str(p): sha256_file(p) for p in files}
    preserved = sorted(Path("outputs/deploy").glob("*"))
    before.update({str(p): sha256_file(p) for p in preserved if p.is_file()})
    base_sets = [FeatureSet.load(cache / f"{bb}_{args.train_tag}.npz") for bb in names]
    test_sets = [FeatureSet.load(cache / f"{bb}_{args.test_tag}.npz") for bb in names]
    if args.require_metadata:
        for sets in (base_sets, test_sets):
            validate_alignment(sets)
            validate_stream_metadata({s.backbone: s.metadata for s in sets})
            for bb, fs in zip(names, sets):
                if fs.backbone != bb or fs.content_hashes is None or not np.all(fs.content_hashes != ""):
                    raise ValueError("Fresh evaluation requires matching backbones and original image hashes")
        for development, external in zip(base_sets, test_sets):
            if development.metadata != external.metadata:
                raise ValueError("Development/external extraction settings differ")
    audit = {"training": audit_sets(base_sets, args.train_tag),
             "external": audit_sets(test_sets, args.test_tag),
             "source_sha256": {p.name: before[str(p)] for p in files}}
    check_external_overlap(test_sets[0], {k: getattr(base_sets[0], k).tolist()
        if getattr(base_sets[0], k) is not None else []
        for k in ("paths", "content_hashes", "related_groups")})
    # Detect observed exact-embedding overlap across caches, beyond distinct paths.
    overlaps = {bb: len(set(feature_fingerprints(tr)) & set(feature_fingerprints(te)))
                for bb, tr, te in zip(names, base_sets, test_sets)}
    audit["exact_feature_overlap_by_stream"] = overlaps
    if overlaps["siglip"]:
        raise ValueError("Observed semantic feature overlap between development and external test")
    available = {"siglip_probe": ("probe", ["siglip"]),
                 "forensic_probe": ("probe", ["forensic"]),
                 "fusion": ("fusion", names),
                 "siglip_mlp": ("fusion", ["siglip"])}
    chosen = args.models.split(",")
    if len(set(chosen)) != len(chosen) or any(n not in available for n in chosen):
        raise ValueError("Unknown or repeated model selection")
    plans = [(name, *available[name]) for name in chosen]
    cfg = Config.load(args.config)
    cfg.feature.cache_dir = str(cache)
    cfg.train.epochs, cfg.eval.bootstrap_n = args.epochs, args.bootstrap
    out.mkdir(parents=True)
    write_json(out / "audit.json", audit)
    write_json(out / "preserved_artifacts.json", before)
    write_json(out / "plan.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds, "models": plans, "epochs": args.epochs, "bootstrap_resamples": args.bootstrap,
        "alpha": args.alpha, "shared_group_reference": "siglip", "device": "cpu",
        "selection_policy": "No hyperparameter selection using external scores; report all seeds.",
        "scope": ("Fresh-cache evaluation; independent raw-data/serving review still required."
                  if args.require_metadata else "Exploratory recovered-feature evaluation, not a verified deployable release."),
        "require_metadata": args.require_metadata,
        "reason": args.reason})
    print(json.dumps(audit, indent=2), flush=True)
    results = []
    reference_splits = {}
    for seed in seeds:
        cfg.seed = seed
        for name, kind, streams in plans:
            start = time.perf_counter()
            run = out / f"{name}_seed{seed}"
            print(f"START {name} seed={seed}", flush=True)
            training = train(cfg, streams, args.train_tag, run, kind, args.alpha, "cpu", not args.require_metadata,
                             group_reference=base_sets[0])
            splits = json.loads((run / "splits.json").read_text())
            current = {k: v["original_indices"] for k, v in splits["partitions"].items()}
            if seed in reference_splits and current != reference_splits[seed]:
                raise ValueError("Ablation split mismatch")
            reference_splits[seed] = current
            model = (FusionDetector.load(run / "fusion.pt") if kind == "fusion" else
                     LinearProbe.load(run / "probe.pkl"))
            selected_sets = [test_sets[names.index(bb)] for bb in streams]
            report, scores, labels, paths = evaluate_cached(
                model, selected_sets, Calibrator.load(run / "calibrator.json"),
                verify_bundle(run) if args.require_metadata else None,
                splits["development_ids"], exploratory_legacy=not args.require_metadata,
                bootstrap_n=args.bootstrap, ci=cfg.eval.ci, seed=seed,
                group_reference=test_sets[0])
            write_json(run / "external_evaluation.json", report)
            np.savez_compressed(run / "external_predictions.npz", scores=scores, labels=labels, paths=paths)
            # Independent metric cross-check (does not share the rank-AUC helper).
            from sklearn.metrics import roc_auc_score, confusion_matrix
            if not np.isclose(roc_auc_score(labels, scores), report["ranking"]["auc"]["point"], atol=1e-12):
                raise ValueError("Independent AUC cross-check failed")
            matrix = confusion_matrix(labels, scores > report["operating"]["tau"], labels=[0, 1])
            write_json(run / "cross_check.json", {"sklearn_auc": float(roc_auc_score(labels, scores)),
                "confusion_matrix_real_fake": matrix.tolist(),
                "shared_splits_verified": True})
            elapsed = time.perf_counter()-start
            item = {"model": name, "seed": seed, "training": training,
                    "external": report, "seconds": elapsed}
            results.append(item)
            write_json(out / "results.json", results)
            print(f"DONE {name} seed={seed} AUC={report['ranking']['auc']['point']:.4f} "
                  f"FPR={report['operating']['fpr']['point']:.4f} "
                  f"TPR={report['operating']['tpr']['point']:.4f} seconds={elapsed:.1f}", flush=True)
    summary = {}
    for name, _, _ in plans:
        runs = [r for r in results if r["model"] == name]
        summary[name] = {metric: {"mean": float(np.mean(values)), "seed_std": float(np.std(values)),
                                  "min": float(min(values)), "max": float(max(values))}
            for metric, values in {
                "auc": [r["external"]["ranking"]["auc"]["point"] for r in runs],
                "fpr": [r["external"]["operating"]["fpr"]["point"] for r in runs],
                "tpr": [r["external"]["operating"]["tpr"]["point"] for r in runs],
                "tpr_at_1pct": [r["external"]["ranking"]["tpr_at_0.01"]["point"] for r in runs],
                "tpr_at_5pct": [r["external"]["ranking"]["tpr_at_0.05"]["point"] for r in runs],
            }.items()}
    write_json(out / "summary.json", {"n_seeds": len(seeds), "models": summary,
        "scope": ("Fresh metadata-bearing caches; seed spread is not a population confidence interval."
                  if args.require_metadata else "Exploratory legacy caches; seed spread is not a population confidence interval.")})
    for path, digest in before.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Original artifact changed: {path}")
    write_json(out / "completed.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                                        "original_artifacts_unchanged": True, "runs": len(results)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
