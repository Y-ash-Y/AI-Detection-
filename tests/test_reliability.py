"""Offline contracts: no datasets, model downloads or expensive training needed."""
import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from importlib.metadata import version

from pmsa.artifacts import save_bundle, verify_bundle, validate_stream_metadata, check_runtime
from pmsa.calibration import Calibrator, conformal_threshold
from pmsa.config import Config
from pmsa.data.splits import group_ids, grouped_split, independent_indices, logo_split
from pmsa.evaluation import evaluate_cached, check_external_overlap
from pmsa.eval.metrics import roc_auc, eer, operating_report
from pmsa.features import FeatureSet
from pmsa.features.cache import validate_alignment
from pmsa.inference import Detector, decision_label
from pmsa.preprocessing import ImagePolicy, load_image, prepare_image
from pmsa.provenance import check_provenance


@pytest.mark.parametrize("script", sorted(p.name for p in (Path(__file__).resolve().parents[1] / "scripts").glob("*.py")))
def test_cli_help_without_install_or_data(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / script), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def metadata():
    return {"schema_version": 1, "image_policy": ImagePolicy(normalize_jpeg=True).to_dict(),
            "backbone_kwargs": {}, "packages": {"Pillow": version("Pillow"), "numpy": version("numpy")}}


def toy_set(n=400, offset=0, name="toy"):
    rng = np.random.default_rng(4)
    labels = (np.arange(n) % 2).astype(np.int8)
    features = rng.normal(size=(n, 4)).astype(np.float32)
    features[:, 0] += labels * 2
    return FeatureSet(features, labels, np.array([f"image{i+offset}.jpg" for i in range(n)]),
                      np.array(["photos"]*n), np.where(labels, "generator", "real"), name,
                      metadata(), np.array([f"{i+offset:064x}" for i in range(n)]), np.array([""]*n))


@pytest.mark.parametrize("score,tau,label", [(-.1, -1., "AI-generated signal"),
    (-1.5, -1., "No AI signal detected"), (-.9, -1., "Uncertain"), (0., 0., "Uncertain")])
def test_abstention_is_centered_on_threshold(score, tau, label):
    assert decision_label(score, tau) == label


@pytest.mark.parametrize("score,tau,margin", [(np.nan, 0., .25), (0., np.inf, .25), (0., 0., -.1)])
def test_nonfinite_decisions_rejected(score, tau, margin):
    with pytest.raises(ValueError):
        decision_label(score, tau, margin)


def test_metadata_does_not_override_forensics(tmp_path):
    path = tmp_path / "spoof.txt"
    path.write_text("c2pa openai synthid")
    det = Detector.__new__(Detector)
    det.calibrator = Calibrator(1., .05, "empirical", 100, "reference", False)
    det.uncertain_margin = .25
    det.artifact_status = "verified-bundle"
    det._features = lambda image: []
    det._score_and_streams = lambda streams: (-2., {"toy": -2.}, 0.)
    verdict = det.predict(Image.new("RGB", (3, 3)), path)
    assert verdict.provenance["found"] and not verdict.provenance["verified"]
    assert verdict.label == "No AI signal detected"
    assert verdict.ai_probability is None
    from app import render_verdict
    headline, diagnostics, detail = render_verdict(verdict)
    assert "probability:" not in headline.lower()
    assert diagnostics["Score minus threshold"] == -3.
    assert "not causal" in detail


def test_provenance_scan_is_bounded(tmp_path):
    path = tmp_path / "bounded.bin"
    path.write_bytes(b"x" * 100 + b"openai")
    assert not check_provenance(path, scan_bytes=50).found
    assert check_provenance(path, scan_bytes=200).found


def test_groups_transitive_and_row_order_invariant():
    paths = np.array(["a", "b", "c", "d"])
    hashes = np.array(["same", "same", "other", "fourth"])
    related = np.array(["", "edit-family", "edit-family", ""])
    groups = group_ids(paths, hashes, related)
    assert groups[0] == groups[1] == groups[2] != groups[3]
    order = np.array([2, 0, 3, 1])
    assert np.array_equal(group_ids(paths[order], hashes[order], related[order]), groups[order])


def test_grouped_splits_disjoint_and_reproducible():
    groups = np.repeat(np.arange(40), 3)
    a, b = grouped_split(groups, seed=7), grouped_split(groups, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(np.concatenate(a)) == len(groups)
    for i, x in enumerate(a):
        for y in a[i+1:]:
            assert not set(groups[x]) & set(groups[y])


def test_mixed_label_duplicate_rejected():
    with pytest.raises(ValueError, match="Mixed labels"):
        independent_indices([0, 1], ["original-edit", "original-edit"])


def test_logo_reals_and_target_generator_never_leak():
    labels = np.array([0]*20+[1]*10+[1]*10)
    sources = np.array(["real"]*20+["known"]*10+["target"]*10)
    groups = np.array([f"g{i}" for i in range(40)])
    train, test = logo_split(labels, sources, groups, "target")
    assert not set(groups[train]) & set(groups[test])
    assert "target" not in sources[train]
    assert set(np.unique(labels[train])) == set(np.unique(labels[test])) == {0, 1}


def test_metadata_cache_roundtrip_and_alignment(tmp_path):
    fs = toy_set(10)
    path = tmp_path / "cache.npz"
    fs.save(path)
    back = FeatureSet.load(path)
    assert back.metadata == fs.metadata
    assert np.array_equal(back.content_hashes, fs.content_hashes)
    validate_alignment([fs, back])
    back.domain[0] = "different"
    with pytest.raises(ValueError, match="domain"):
        validate_alignment([fs, back])


def test_shared_image_policy_matches_disk_and_memory(tmp_path):
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 256, (23, 37, 3), dtype=np.uint8))
    path = tmp_path / "image.png"
    image.save(path)
    policy = ImagePolicy(normalize_jpeg=True)
    assert np.array_equal(np.array(load_image(path, policy)), np.array(prepare_image(image, policy)))


def test_exif_orientation_and_pixel_limit():
    image = Image.new("RGB", (2, 3))
    image.getexif()[274] = 6
    assert prepare_image(image, ImagePolicy()).size == (3, 2)
    with pytest.raises(ValueError, match="pixel limit"):
        prepare_image(image, ImagePolicy(max_pixels=5))


def test_animated_image_rejected(tmp_path):
    path = tmp_path / "animated.gif"
    Image.new("RGB", (4, 4), "red").save(path, save_all=True,
        append_images=[Image.new("RGB", (4, 4), "blue")])
    with pytest.raises(ValueError, match="Multi-frame"):
        load_image(path, ImagePolicy())


def test_bundle_integrity_and_pairing(tmp_path):
    (tmp_path / "fusion.pt").write_bytes(b"trusted model placeholder")
    (tmp_path / "calibrator.json").write_text("{}")
    (tmp_path / "splits.json").write_text("{}")
    save_bundle(tmp_path, "fusion.pt", "fusion", {"toy": metadata()})
    verify_bundle(tmp_path, tmp_path / "fusion.pt", tmp_path / "calibrator.json")
    with pytest.raises(ValueError, match="belong"):
        verify_bundle(tmp_path, tmp_path / "other.pt")
    (tmp_path / "calibrator.json").write_text('{"tau": 100}')
    with pytest.raises(ValueError, match="integrity"):
        verify_bundle(tmp_path)


def test_unpinned_and_different_preprocessing_rejected():
    a, b = metadata(), metadata()
    b["image_policy"]["normalize_jpeg"] = False
    with pytest.raises(ValueError, match="different image"):
        validate_stream_metadata({"a": a, "b": b})
    with pytest.raises(ValueError, match="Unpinned"):
        validate_stream_metadata({"siglip": a})


def test_runtime_mismatch_rejected():
    m = metadata()
    m["packages"]["numpy"] = "not-the-installed-version"
    with pytest.raises(ValueError, match="runtime mismatch"):
        check_runtime({"toy": m})


def test_runtime_allows_cpu_cuda_build_suffix_difference(monkeypatch):
    m = metadata()
    m["packages"] = {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124"}
    monkeypatch.setattr("importlib.metadata.version", lambda p: {"torch": "2.6.0+cpu", "torchvision": "0.21.0+cpu"}[p])
    check_runtime({"toy": m})


def test_forensic_extraction_and_inference_feature_parity(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from pmsa.backbones import build_backbone
    from pmsa.data.manifest import Manifest, ImageRecord
    from pmsa.features.extract import extract_backbone
    rng = np.random.default_rng(1)
    path = tmp_path / "photo.png"
    Image.fromarray(rng.integers(0, 256, (45, 70, 3), dtype=np.uint8)).save(path)
    fs = extract_backbone(Manifest([ImageRecord(str(path), 0, "photos", "real")]),
                          "forensic", tmp_path / "forensic.npz", device="cpu",
                          batch_size=1, num_workers=0, image_size=32, normalize_jpeg=True)
    det = Detector.__new__(Detector)
    det.image_policy = ImagePolicy(**fs.metadata["image_policy"])
    det.backbones = [build_backbone("forensic", device="cpu", **fs.metadata["backbone_kwargs"])]
    with Image.open(path) as original:
        serving = det._features(original)[0]
    assert np.array_equal(serving, fs.features)
    assert fs.content_hashes[0] and fs.features.shape == (1, 66)


def test_legacy_does_not_silently_load(tmp_path):
    with pytest.raises(ValueError, match="Legacy artifacts"):
        Detector(tmp_path / "fusion.pt", tmp_path / "calibrator.json")


@pytest.mark.parametrize("alpha", [0., 1., -1., np.nan])
def test_invalid_calibration_alpha(alpha):
    with pytest.raises(ValueError):
        conformal_threshold([0., 1.], alpha)


def test_calibration_rejects_bad_scores_and_metadata():
    for scores in ([], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            conformal_threshold(scores, .05)
    with pytest.raises(ValueError):
        Calibrator(np.nan, .05, "conformal", 99, "photos", True)


@pytest.mark.parametrize("scores,labels", [([np.nan, 1], [0, 1]),
    ([0, 1], [0, 2]), ([0, 1, 2], [0, 1]), ([[0, 1]], [[0, 1]])])
def test_metric_invalid_inputs_rejected(scores, labels):
    with pytest.raises(ValueError):
        roc_auc(np.asarray(scores), np.asarray(labels))


def test_tied_auc_matches_sklearn_and_eer_matches_bruteforce():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(2)
    scores, labels = rng.integers(-2, 3, 100), np.arange(100) % 2
    assert roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))
    thresholds = np.unique(scores)
    fp = np.array([(scores[labels == 0] > t).mean() for t in thresholds])
    fn = np.array([(scores[labels == 1] <= t).mean() for t in thresholds])
    i = np.argmin(abs(fp-fn))
    assert eer(scores, labels) == pytest.approx((fp[i]+fn[i])/2)


def test_frozen_operating_point_and_single_class_rates():
    report = operating_report(np.array([0., 2., 3., 4.]), np.array([0, 0, 1, 1]), 2.)
    assert report["fpr"]["point"] == 0. and report["tpr"]["point"] == 1.
    assert report["fpr"]["hi"] > 0.  # Observing zero errors is not proof of zero risk.
    only_fake = operating_report(np.array([3.]), np.array([1]), 2.)
    assert only_fake["fpr"]["point"] is None


@pytest.mark.parametrize("key", ["paths", "content_hashes", "related_groups"])
def test_external_overlap_rejected(key):
    fs = toy_set(10)
    if key == "related_groups":
        fs.related_groups[0] = "family"
    with pytest.raises(ValueError, match=key):
        check_external_overlap(fs, {key: [getattr(fs, key)[0]]})


def test_safe_fusion_roundtrip_batching_and_signed_decomposition(tmp_path):
    pytest.importorskip("torch")
    from pmsa.models import FusionDetector, StreamSpec
    model = FusionDetector([StreamSpec("a", 3, 4), StreamSpec("b", 2, 4)])
    rng = np.random.default_rng(2)
    streams = [rng.normal(size=(20, d)).astype(np.float32) for d in (3, 2)]
    model._fit_norm(streams)
    scores = model.score(streams, batch_size=3)
    contributions = model.contributions(streams)
    bias = float(model.net.combiner.bias.detach()[0])
    assert np.allclose(sum(contributions.values())+bias, scores, atol=1e-6)
    path = tmp_path / "fusion.pt"
    model.save(path)
    restored = FusionDetector.load(path)
    assert np.allclose(restored.score(streams, batch_size=7), scores, atol=1e-6)


def test_unsafe_legacy_pickle_is_not_automatically_retried(tmp_path):
    torch = pytest.importorskip("torch")
    from pmsa.models import FusionDetector
    path = tmp_path / "old.pt"
    torch.save({"mean": [np.array([1., 2.])]}, path)
    with pytest.raises(pickle.UnpicklingError):
        FusionDetector.load(path)


def test_tiny_end_to_end_training_bundle_serving_and_external_eval(tmp_path, monkeypatch):
    from pmsa.training import train
    from pmsa.models import LinearProbe
    fs = toy_set()
    cache = tmp_path / "cache"
    fs.save(cache / "toy_dev.npz")
    cfg = Config()
    cfg.feature.cache_dir = str(cache)
    cfg.eval.bootstrap_n = 5
    out = tmp_path / "bundle"
    report = train(cfg, ["toy"], "dev", out, alpha=.1)
    assert not report["exploratory_legacy"]
    splits = json.loads((out / "splits.json").read_text())
    partitions = splits["partitions"]
    for a, x in partitions.items():
        for b, y in partitions.items():
            if a != b:
                assert not set(x["groups"]) & set(y["groups"])
    cal = Calibrator.load(out / "calibrator.json")
    assert cal.n_cal == partitions["calibration"]["n_real"]
    with pytest.raises(FileExistsError):
        train(cfg, ["toy"], "dev", out, alpha=.1)
    bundle = verify_bundle(out)
    model = LinearProbe.load(out / "probe.pkl")
    target = toy_set(40, offset=10000)
    rep, scores, labels, paths = evaluate_cached(model, [target], cal, bundle,
        splits["development_ids"], bootstrap_n=5)
    assert not rep["exploratory_legacy"] and len(scores) == 40
    assert rep["operating"]["tau"] == cal.tau
    class FakeBackbone:
        def encode(self, images):
            return target.features[:1]
    monkeypatch.setattr("pmsa.inference.build_backbone", lambda *a, **kw: FakeBackbone())
    det = Detector(out / "probe.pkl", out / "calibrator.json")
    verdict = det.predict(Image.new("RGB", (5, 5)))
    assert verdict.ai_probability is None and np.isfinite(verdict.score)
    assert verdict.score == pytest.approx(scores[0])
    wrong = copy.deepcopy(target)
    wrong.metadata["image_policy"]["normalize_jpeg"] = False
    with pytest.raises(ValueError, match="configuration differs"):
        evaluate_cached(model, [wrong], cal, bundle, splits["development_ids"], bootstrap_n=5)


def test_legacy_training_is_explicitly_exploratory(tmp_path):
    from pmsa.training import train
    fs = toy_set()
    fs.metadata, fs.content_hashes = {}, None
    fs.save(tmp_path / "toy_dev.npz")
    cfg = Config()
    cfg.feature.cache_dir = str(tmp_path)
    cfg.eval.bootstrap_n = 5
    out = tmp_path / "exploratory"
    with pytest.raises(ValueError, match="Legacy cache"):
        train(cfg, ["toy"], "dev", out, alpha=.1)
    report = train(cfg, ["toy"], "dev", out, alpha=.1, exploratory_legacy=True)
    assert report["exploratory_legacy"] and (out / "NOT_DEPLOYABLE.txt").exists()
    assert not (out / "bundle.json").exists()


def test_shared_reference_keeps_ablation_partitions_identical(tmp_path):
    from pmsa.training import train
    reference = toy_set(name="semantic")
    reference.metadata, reference.content_hashes = {}, None
    artifact = copy.deepcopy(reference)
    artifact.backbone = "artifact"
    artifact.features[2] = artifact.features[0]  # Extra same-class encoding collision.
    reference.save(tmp_path / "semantic_dev.npz")
    artifact.save(tmp_path / "artifact_dev.npz")
    cfg = Config()
    cfg.feature.cache_dir = str(tmp_path)
    cfg.eval.bootstrap_n = 3
    outputs = []
    for name in ("semantic", "artifact"):
        out = tmp_path / name
        report = train(cfg, [name], "dev", out, alpha=.1, exploratory_legacy=True,
                       group_reference=reference)
        assert report["independent_representatives"] == len(reference)
        outputs.append(json.loads((out / "splits.json").read_text())["partitions"])
    assert outputs[0] == outputs[1]
