"""Auditable training from caches; independent fit/validation/calibration/test."""
import json
from pathlib import Path

import numpy as np

from .artifacts import save_bundle, sha256_file, validate_stream_metadata
from .calibration import Calibrator
from .calibration.np_threshold import min_calibration_size
from .data.splits import grouped_split, independent_indices
from .eval import full_report
from .eval.metrics import operating_report
from .features import FeatureSet
from .features.cache import validate_alignment
from .models import FusionDetector, LinearProbe, StreamSpec


def train(cfg, names, tag, out, model_type="probe", alpha=.05, device="cpu", exploratory_legacy=False,
          group_reference=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Run output already exists; choose a NEW --out")
    if len(set(names)) != len(names) or (model_type == "probe" and len(names) != 1):
        raise ValueError("Invalid stream names")
    if model_type not in ("probe", "fusion"):
        raise ValueError("Unknown model type")
    files = [Path(cfg.feature.cache_dir) / f"{name}_{tag}.npz" for name in names]
    sets = [FeatureSet.load(p) for p in files]
    validate_alignment(sets)
    for name, fs in zip(names, sets):
        if fs.backbone != name:
            raise ValueError("Cache backbone name mismatch")
    complete = all(s.metadata.get("schema_version") == 1 and "image_policy" in s.metadata
                   and s.content_hashes is not None and np.all(s.content_hashes != "") for s in sets)
    if not complete and not exploratory_legacy:
        raise ValueError("Legacy cache metadata missing. Re-extract using a new tag, or use --exploratory-legacy (not deployable).")
    if complete:
        validate_stream_metadata({s.backbone: s.metadata for s in sets})
    base = sets[0]
    if group_reference is not None:
        # Ablations must share the same image groups, not each stream's accidental
        # legacy feature collisions. Alignment includes labels and identities.
        validate_alignment([group_reference, *sets])
    reference = group_reference if group_reference is not None else base
    original_n = len(base)
    # Preserve ALL members for external overlap detection, even excluded duplicates.
    development_ids = {k: getattr(base, k).tolist() if getattr(base, k) is not None else []
                       for k in ("paths", "content_hashes", "related_groups")}
    groups = reference.groups()
    selected = independent_indices(base.labels, groups)
    sets = [s.select(selected) for s in sets]
    base, groups = sets[0], groups[selected]
    parts = dict(zip(("train", "validation", "calibration", "test"), grouped_split(groups, seed=cfg.seed)))
    for name, ix in parts.items():
        if set(np.unique(base.labels[ix])) != {0, 1}:
            raise ValueError(f"Split {name} lacks a class; use a larger independently sampled dataset")
    tr, va, ca, te = (parts[n] for n in ("train", "validation", "calibration", "test"))
    real_cal = ca[base.labels[ca] == 0]
    if len(real_cal) < min_calibration_size(alpha):
        raise ValueError("Insufficient independent real calibration examples for finite threshold")
    def streams(ix):
        return [s.features[ix] for s in sets]
    if model_type == "probe":
        model = LinearProbe(seed=cfg.seed).fit(base.features[tr], base.labels[tr])
        score = lambda ix: model.score(base.features[ix])
        filename = "probe.pkl"
    else:
        model = FusionDetector([StreamSpec(s.backbone, s.dim) for s in sets], device, cfg.seed)
        model.fit(streams(tr), base.labels[tr], streams(va), base.labels[va],
                  epochs=cfg.train.epochs, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
                  batch_size=cfg.train.batch_size)
        score = lambda ix: model.score(streams(ix))
        filename = "fusion.pt"
    cal = Calibrator.fit(score(real_cal), alpha, cfg.calibration.method, "mixed")
    if not np.isfinite(cal.tau):
        raise ValueError("Finite threshold required for deployment")
    scores = score(te)
    report = {
        "protocol": "independent-groups-v3", "exploratory_legacy": not complete,
        "seed": cfg.seed, "original_rows": original_n, "independent_representatives": len(base),
        "ranking": full_report(scores, base.labels[te], cfg.eval.fpr_targets,
                               cfg.eval.bootstrap_n, cfg.eval.ci, cfg.seed),
        "operating": operating_report(scores, base.labels[te], cal.tau, cfg.eval.ci),
        "notes": ["Ranking TPR@FPR uses test-derived thresholds; operating rates use the frozen calibrator.",
                  "Conformal FPR control is marginal under exchangeable reals, not a guarantee under arbitrary shift.",
                  "Exact hashes/annotations cannot exclude unannotated near-duplicates.",
                  "Legacy feature fingerprints cannot establish image-level uniqueness or preprocessing provenance."],
    }
    split_manifest = {"seed": cfg.seed, "development_ids": development_ids,
                      "grouping": {"reference_backbone": reference.backbone,
                                   "basis": "content_hashes" if reference.content_hashes is not None else "legacy_feature_fingerprints"},
                      "cache_sha256": {p.name: sha256_file(p) for p in files},
                      "partitions": {name: {"original_indices": selected[ix].tolist(),
                                            "groups": groups[ix].tolist(),
                                            "n_real": int((base.labels[ix] == 0).sum()),
                                            "n_fake": int((base.labels[ix] == 1).sum())}
                                     for name, ix in parts.items()}}
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / filename)
    cal.save(out / "calibrator.json")
    for name, value in (("splits", split_manifest), ("evaluation", report), ("config", cfg.to_dict())):
        (out / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False))
    root = Path(__file__).resolve().parents[1]
    (out / "source_hashes.json").write_text(json.dumps({str(p.relative_to(root)): sha256_file(p)
        for p in sorted([*root.glob("pmsa/**/*.py"), *root.glob("scripts/*.py"), root / "pyproject.toml"])}, indent=2))
    np.savez_compressed(out / "test_predictions.npz", scores=scores, labels=base.labels[te],
                        paths=base.paths[te])
    if complete:
        save_bundle(out, filename, model_type, {s.backbone: s.metadata for s in sets})
    else:
        (out / "NOT_DEPLOYABLE.txt").write_text("Exploratory legacy-cache result only. Re-extract v3 features before deployment.\n")
    return report
