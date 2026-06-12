"""Tests for metrics and the feature cache — both run without data or GPU."""
import numpy as np

from pmsa.eval import roc_auc, tpr_at_fpr, eer, full_report
from pmsa.features import FeatureSet, concat_streams


def test_auc_perfect_separation():
    scores = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc(scores, labels) == 1.0


def test_auc_matches_random_at_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 4000)
    labels = rng.integers(0, 2, 4000)
    assert abs(roc_auc(scores, labels) - 0.5) < 0.05


def test_tpr_at_fpr_and_eer_ranges():
    rng = np.random.default_rng(1)
    real = rng.normal(0, 1, 2000)
    fake = rng.normal(2, 1, 2000)
    scores = np.concatenate([real, fake])
    labels = np.concatenate([np.zeros(2000), np.ones(2000)])
    assert 0.0 <= tpr_at_fpr(scores, labels, 0.01) <= 1.0
    assert 0.0 <= eer(scores, labels) <= 0.5


def test_full_report_has_cis():
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(0, 1, 500), rng.normal(2, 1, 500)])
    labels = np.concatenate([np.zeros(500), np.ones(500)])
    rep = full_report(scores, labels, bootstrap_n=200)
    assert rep["auc"]["lo"] <= rep["auc"]["point"] <= rep["auc"]["hi"]


def _toy_set(n=10, d=4, backbone="a", seed=0):
    rng = np.random.default_rng(seed)
    return FeatureSet(
        features=rng.normal(0, 1, (n, d)).astype(np.float32),
        labels=(np.arange(n) % 2).astype(np.int8),
        paths=np.array([f"img{i}.png" for i in range(n)]),
        domain=np.array(["imagenet"] * n),
        source=np.array(["real" if i % 2 == 0 else "sdxl" for i in range(n)]),
        backbone=backbone,
    )


def test_cache_roundtrip(tmp_path):
    fs = _toy_set()
    p = tmp_path / "fs.npz"
    fs.save(p)
    back = FeatureSet.load(p)
    assert np.allclose(back.features, fs.features)
    assert back.backbone == "a" and len(back) == 10


def test_concat_streams_aligned():
    a, b = _toy_set(backbone="a"), _toy_set(backbone="b")
    fused = concat_streams([a, b])
    assert fused.dim == a.dim + b.dim
    assert fused.backbone == "a+b"


def test_concat_streams_rejects_misaligned():
    a = _toy_set(backbone="a", seed=0)
    b = _toy_set(backbone="b", seed=0)
    b.paths = b.paths[::-1].copy()
    try:
        concat_streams([a, b])
        assert False, "should have raised"
    except ValueError:
        pass
