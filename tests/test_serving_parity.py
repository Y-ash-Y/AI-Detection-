"""Synthetic engineering checks; not image-detector performance evidence."""
import json

import numpy as np
import pytest
from PIL import Image
from importlib.metadata import version

from pmsa.artifacts import save_bundle
from pmsa.calibration import Calibrator
from pmsa.data.manifest import ImageRecord, Manifest
from pmsa.features.extract import extract_backbone
from pmsa.inference import Detector
from pmsa.models import FusionDetector, StreamSpec
from pmsa.parity import compare_arrays, image_serving_parity, replay_cached_scores, runtime_preflight


def test_comparison_fails_closed():
    for a, b in [([], []), ([np.nan], [np.nan]), ([1], [[1]]), ([1], [2])]:
        assert not compare_arrays(a, b)["passed"]
    assert compare_arrays([1.], [1.000001])["passed"]
    with pytest.raises(ValueError):
        compare_arrays([1], [1], atol=-1)


@pytest.fixture
def forensic_bundle(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    rng = np.random.default_rng(22)
    records = []
    # Cover RGB, alpha, grayscale and EXIF orientation, without personal images.
    for i, mode in enumerate(["RGB", "RGBA", "L", "RGB"]):
        shape = (37+i, 51+i) if mode == "L" else (37+i, 51+i, 4 if mode == "RGBA" else 3)
        image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
        if i == 3:
            image.getexif()[274] = 6
        path = tmp_path / f"input{i}.png"
        image.save(path, exif=image.getexif())
        records.append(ImageRecord(str(path), i % 2, "synthetic-test", "fixture"))
    fs = extract_backbone(Manifest(records), "forensic", tmp_path / "cache.npz",
                         device="cpu", batch_size=2, num_workers=0,
                         image_size=32, normalize_jpeg=True)
    model = FusionDetector([StreamSpec("forensic", 66, 8)], seed=8)
    # Only initialize normalization; no learning or performance estimate.
    model._fit_norm([fs.features])
    model.save(tmp_path / "fusion.pt")
    Calibrator(0., .05, "empirical", 100, "synthetic-test", False).save(tmp_path / "calibrator.json")
    (tmp_path / "splits.json").write_text("{}")
    bundle = save_bundle(tmp_path, "fusion.pt", "fusion", {"forensic": fs.metadata})
    det = Detector(tmp_path / "fusion.pt", tmp_path / "calibrator.json")
    return tmp_path, fs, det, bundle


def test_image_batch_and_original_cache_parity(forensic_bundle):
    directory, fs, det, _ = forensic_bundle
    report = image_serving_parity(det, fs.paths, batch_size=3, reference_sets=[fs])
    assert report["status"] == "passed"
    assert report["original_cache_compared"]
    assert len(report["images"]) == 4
    assert str(directory) not in json.dumps(report)
    assert "accuracy" not in report


def test_actual_app_handler_matches_offline_features(forensic_bundle, monkeypatch):
    pytest.importorskip("gradio")
    from app import classify
    directory, fs, det, _ = forensic_bundle
    monkeypatch.setattr("app._load", lambda: det)
    for i, path in enumerate(fs.paths):
        expected, _, _ = det._score_and_streams([fs.features[i:i+1]])
        _, diagnostics, details = classify(str(path))
        assert np.isclose(diagnostics["Score (not a probability)"], expected)
        assert diagnostics["Binary threshold exceeded"] == bool(expected > det.calibrator.tau)
        assert "not causal" in details


def test_changed_reference_image_rejected(forensic_bundle):
    _, fs, det, _ = forensic_bundle
    Image.new("RGB", (8, 8)).save(fs.paths[0])
    with pytest.raises(ValueError, match="changed since extraction"):
        image_serving_parity(det, fs.paths, reference_sets=[fs])


def test_serving_feature_drift_detected(forensic_bundle, monkeypatch):
    _, fs, det, _ = forensic_bundle
    original = det._features
    monkeypatch.setattr(det, "_features", lambda image: [x+1 for x in original(image)])
    assert image_serving_parity(det, fs.paths)["status"] == "failed"


def test_safe_neural_head_replay(forensic_bundle):
    directory, fs, det, _ = forensic_bundle
    path = directory / "predictions.npz"
    np.savez(path, scores=det.fusion.score([fs.features]), labels=fs.labels, paths=fs.paths)
    report = replay_cached_scores(directory, [fs], path, sample_n=4)
    assert report["status"] == "passed"
    assert report["binary_decision_changes"] == 0
    with np.load(path, allow_pickle=False) as data:
        scores = data["scores"] + 1
    np.savez(path, scores=scores, labels=fs.labels, paths=fs.paths)
    assert replay_cached_scores(directory, [fs], path)["status"] == "failed"


def test_mismatch_preflight_is_explicit():
    report = runtime_preflight({"streams": {"test": {"packages": {"numpy": "missing-version"}}}})
    assert not report["compatible"] and "numpy" in report["reason"]


def test_preflight_does_not_load_backbone(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("check_parity", Path(__file__).parents[1] / "scripts/15_check_serving_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "verify_bundle", lambda directory: {"streams": {"test": {"packages": {"numpy": version("numpy")}}}})
    monkeypatch.setattr(module, "Detector", lambda *a, **k: pytest.fail("No model load expected"))
    out = tmp_path / "preflight.json"
    monkeypatch.setattr("sys.argv", ["parity", "--bundle", str(tmp_path), "--preflight-only", "--out", str(out)])
    assert module.main() == 0
    with pytest.raises(FileExistsError):
        module.main()


def test_kaggle_parity_cell_syntax():
    from pathlib import Path
    path = Path(__file__).parents[1] / "notebooks/kaggle_serving_parity.py"
    compile(path.read_text(), str(path), "exec")
