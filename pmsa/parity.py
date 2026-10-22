"""Serving parity checks, not performance evaluation or model selection.

Cached replay can run without a backbone. Image parity requires the recorded
extraction runtime and never bypasses Detector's compatibility checks.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import check_runtime, sha256_file, verify_bundle, validate_stream_metadata
from .calibration import Calibrator
from .features.cache import validate_alignment
from .inference import Detector, decision_label
from .models import FusionDetector
from .preprocessing import load_image


def compare_arrays(expected, actual, *, atol=1e-5, rtol=1e-4):
    """Report tolerances explicitly; reject empty, mismatched or nonfinite data."""
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Finite nonnegative tolerances required")
    a, b = np.asarray(expected), np.asarray(actual)
    valid = (a.shape == b.shape and a.size > 0
             and np.isfinite(a).all() and np.isfinite(b).all())
    return {"passed": bool(valid and np.allclose(a, b, atol=atol, rtol=rtol)),
            "expected_shape": list(a.shape), "actual_shape": list(b.shape),
            "max_absolute_error": float(np.max(np.abs(a-b))) if valid else None,
            "atol": atol, "rtol": rtol}


def runtime_preflight(bundle):
    try:
        check_runtime(bundle["streams"])
    except ValueError as exc:
        return {"compatible": False, "reason": str(exc)}
    return {"compatible": True, "reason": "Recorded extraction package versions match."}


def replay_cached_scores(directory, sets, predictions, *, sample_n=64):
    """Replay a trusted tensor-only neural checkpoint against saved predictions.

Does not download/load a backbone or establish raw-image serving parity. Linear
joblib models are intentionally excluded from this safe tensor-load workflow.
"""
    directory = Path(directory)
    bundle = verify_bundle(directory)
    validate_stream_metadata(bundle["streams"])
    if bundle["model_type"] != "fusion":
        raise ValueError("Cached replay only accepts safe tensor neural checkpoints")
    if sample_n < 1:
        raise ValueError("sample_n must be positive")
    validate_alignment(sets)
    if [s.backbone for s in sets] != list(bundle["streams"]):
        raise ValueError("Cache stream order differs from bundle")
    for fs in sets:
        if fs.metadata != bundle["streams"][fs.backbone]:
            raise ValueError("Cache extraction metadata differs from bundle")
    with np.load(predictions, allow_pickle=False) as d:
        scores, paths, labels = d["scores"], d["paths"], d["labels"]
    lookup = {str(p): i for i, p in enumerate(sets[0].paths)}
    if len(lookup) != len(sets[0]) or len(set(paths.tolist())) != len(paths):
        raise ValueError("Ambiguous duplicate paths")
    try:
        ix = np.array([lookup[str(p)] for p in paths])
    except KeyError as exc:
        raise ValueError("Prediction image missing from cache") from exc
    if scores.shape != labels.shape or labels.shape != paths.shape or not len(scores):
        raise ValueError("Invalid prediction shapes")
    if not np.array_equal(sets[0].labels[ix], labels):
        raise ValueError("Prediction/cache labels differ")
    model = FusionDetector.load(directory / bundle["model_file"], device="cpu")
    streams = [s.features[ix] for s in sets]
    replay = model.score(streams)
    cal = Calibrator.load(directory / "calibrator.json")
    numerical = compare_arrays(scores, replay, atol=2e-4)
    binary_changes = int(np.count_nonzero((scores > cal.tau) != (replay > cal.tau)))
    det = Detector.__new__(Detector)
    det.mode, det.fusion = "fusion", model
    # Include threshold-adjacent rows, then evenly spaced rows, without cherry-picking.
    selected = np.unique(np.concatenate([
        np.argsort(np.abs(scores-cal.tau))[:min(16, sample_n)],
        np.linspace(0, len(scores)-1, min(sample_n, len(scores)), dtype=int)]))
    single, sums = [], []
    for i in selected:
        score, contributions, bias = det._score_and_streams([x[i:i+1] for x in streams])
        single.append(score)
        sums.append(sum(contributions.values()) + bias)
    single_check = compare_arrays(replay[selected], single, atol=2e-4)
    decomposition = compare_arrays(single, sums, atol=2e-4)
    label_changes = sum(decision_label(float(scores[i]), cal.tau) !=
                        decision_label(float(single[j]), cal.tau)
                        for j, i in enumerate(selected))
    passed = (numerical["passed"] and single_check["passed"] and decomposition["passed"]
              and binary_changes == 0 and label_changes == 0)
    return {"status": "passed" if passed else "failed", "rows": len(scores),
            "single_image_score_rows": len(selected), "saved_vs_replayed": numerical,
            "batch_vs_single": single_check, "contributions_plus_bias": decomposition,
            "binary_decision_changes": binary_changes, "sample_label_changes": label_changes,
            "extraction_runtime": runtime_preflight(bundle),
            "scope": "Cached neural-head replay only; no raw-image extraction, app HTTP test, or new accuracy measurement."}


def image_serving_parity(detector, paths, *, batch_size=8, reference_sets=None,
                         feature_atol=1e-5, score_atol=2e-4):
    """Compare extraction-style batches with Detector.predict's image path.

The caller must construct a real Detector, which validates its runtime first.
Optional reference caches additionally test the ORIGINAL extraction, bound by
image byte hashes and extraction metadata. Without caches this tests consistency
between the current two paths, not parity with previously extracted GPU vectors.
"""
    import torch
    from .features.extract import _ManifestDataset
    from .data.manifest import ImageRecord, Manifest
    from .preprocessing import ImagePolicy

    paths = [Path(p) for p in paths]
    if not paths or batch_size < 1:
        raise ValueError("Images and positive batch size required")
    # _ManifestDataset currently exposes only this policy switch. Do not silently
    # pretend it implements arbitrary future policy settings.
    if detector.image_policy != ImagePolicy(normalize_jpeg=detector.image_policy.normalize_jpeg):
        raise ValueError("Extraction dataset cannot reproduce this non-default image policy")
    hashes = [sha256_file(p) for p in paths]
    offline = []
    manifest = Manifest([ImageRecord(str(p), 0, "parity-unlabeled", "unknown") for p in paths])
    for bb in detector.backbones:
        ds = _ManifestDataset(manifest, bb.preprocess, detector.image_policy.normalize_jpeg)
        batches = []
        for start in range(0, len(paths), batch_size):
            tensors = [ds[i][0] for i in range(start, min(start+batch_size, len(paths)))]
            batches.append(bb.encode(torch.stack(tensors).to(detector.device)))
        offline.append(np.concatenate(batches))
    reference = None
    if reference_sets is not None:
        validate_alignment(reference_sets)
        if [s.backbone for s in reference_sets] != detector._stream_names:
            raise ValueError("Reference stream order differs")
        reference = []
        for fs in reference_sets:
            if fs.metadata != detector.bundle["streams"][fs.backbone]:
                raise ValueError("Reference extraction metadata differs")
            if fs.content_hashes is None:
                raise ValueError("Reference cache requires image hashes")
            mapping = {str(p): i for i, p in enumerate(fs.paths)}
            if len(mapping) != len(fs):
                raise ValueError("Reference paths are ambiguous")
            if any(str(p) not in mapping for p in paths):
                raise ValueError("Image missing from reference cache")
            ix = [mapping[str(p)] for p in paths]
            if fs.content_hashes[ix].tolist() != hashes:
                raise ValueError("Raw image changed since extraction")
            reference.append(fs.features[ix])
    rows = []
    for i, path in enumerate(paths):
        with Image.open(path) as original:
            serving = detector._features(original)
            verdict = detector.predict(original, image_path=path)
            prepared = load_image(path, detector.image_policy)
            from .preprocessing import prepare_image
            pixels_equal = np.array_equal(np.asarray(prepared), np.asarray(
                prepare_image(original, detector.image_policy)))
        checks = {name: compare_arrays(offline[j][i:i+1], serving[j], atol=feature_atol)
                  for j, name in enumerate(detector._stream_names)}
        score, _, _ = detector._score_and_streams([x[i:i+1] for x in offline])
        score_check = compare_arrays([score], [verdict.score], atol=score_atol)
        sum_check = compare_arrays([verdict.score],
            [sum(verdict.per_stream.values())+verdict.combiner_bias], atol=score_atol)
        ref_checks = ({name: compare_arrays(reference[j][i:i+1], serving[j], atol=feature_atol)
                       for j, name in enumerate(detector._stream_names)} if reference else {})
        label_equal = verdict.label == decision_label(score, verdict.tau, detector.uncertain_margin)
        threshold_equal = verdict.calibrated_fake == bool(score > verdict.tau)
        passed = (pixels_equal and all(c["passed"] for c in [*checks.values(), *ref_checks.values(),
                  score_check, sum_check]) and label_equal and threshold_equal and verdict.ai_probability is None)
        rows.append({"image_sha256": hashes[i], "passed": passed, "pixels_equal": pixels_equal,
                     "features": checks, "original_cache_features": ref_checks,
                     "score": score_check, "contribution_sum": sum_check,
                     "label_equal": label_equal, "threshold_decision_equal": threshold_equal})
    return {"status": "passed" if all(r["passed"] for r in rows) else "failed",
            "images": rows, "original_cache_compared": reference is not None,
            "scope": "Pipeline consistency only. Images are unlabeled; no FPR, accuracy, or authenticity claim. No raw pixels or filenames saved."}
