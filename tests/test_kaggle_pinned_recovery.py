"""Exercise the literal recovery cell offline with simulated extraction."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from pmsa.data.manifest import ImageRecord, Manifest
from pmsa.features import FeatureSet
from pmsa.artifacts import sha256_file
from pmsa.preprocessing import ImagePolicy

ROOT = Path(__file__).resolve().parents[1]
PIN = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


def test_recovery_preserves_originals_pins_load_and_updates_next_cell(tmp_path, monkeypatch):
    import pmsa.features.extract as extraction

    run, audit = tmp_path / "run", tmp_path / "audit"
    cache = run / "feature_cache"
    cache.mkdir(parents=True)
    audit.mkdir()
    for role, n in (("train", 10), ("test", 6)):
        records = [ImageRecord(f"{role}_{i}.png", i % 2, "fixture", "fixture") for i in range(n)]
        Manifest(records).to_csv(audit / f"{role}.csv")
        for name in ("siglip", "forensic"):
            kw = {"image_size": 224 if name == "siglip" else 256}
            if name == "siglip":
                kw.update(weights="google/siglip-base-patch16-224", revision=None, use_fast=False)
            FeatureSet(np.ones((n, 4), dtype=np.float32), np.arange(n) % 2,
                np.array([r.path for r in records]), np.array([r.domain for r in records]),
                np.array([r.source for r in records]), name,
                {"schema_version": 1, "backbone_kwargs": kw,
                 "image_policy": ImagePolicy(normalize_jpeg=True).to_dict(),
                 "packages": {"fixture": "1"}}).save(cache / f"{name}_{role}_v3.npz")
    original_hashes = {p: sha256_file(p) for p in cache.glob("*.npz")}
    config = run / "config.yaml"
    config.write_text(yaml.safe_dump({"feature": {"cache_dir": str(cache), "backbones": [
        {"name": "siglip", "weights": "google/siglip-base-patch16-224", "image_size": 224},
        {"name": "forensic", "weights": "", "image_size": 256}]}}))
    config_hash = sha256_file(config)
    snapshot = tmp_path / "snapshots" / PIN / "config.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        try_to_load_from_cache=lambda *a, **kw: str(snapshot)))
    calls, stages = [], []

    def builder(name, **kw):
        calls.append((name, kw))
        return SimpleNamespace(load_config=kw)

    def simulate(manifest, name, out_path, **kw):
        bb = extraction.build_backbone(name, weights=kw["weights"], image_size=kw["image_size"])
        role = "train" if "train" in str(out_path) else "test"
        fs = FeatureSet.load(cache / f"{name}_{role}_v3.npz")
        fs.metadata = copy.deepcopy(fs.metadata)
        fs.metadata["backbone_kwargs"].update(bb.load_config)
        fs.save(out_path)

    monkeypatch.setattr(extraction, "build_backbone", builder)
    monkeypatch.setattr(extraction, "extract_backbone", simulate)
    namespace = {"MODE": "smoke", "plan": {"mode": "smoke"}, "CACHE": cache,
        "CONFIG": config, "RUN": run, "AUDIT": audit, "source_digest": "a" * 64,
        "archive_stage": stages.append}
    code = (ROOT / "notebooks/kaggle_resume_pinned_smoke.py").read_text()
    exec(compile(code, "pinned-smoke-recovery", "exec"), namespace)
    assert stages == ["features-pinned"]
    assert namespace["CACHE"] == run / "feature_cache_pinned_01"
    assert yaml.safe_load(namespace["CONFIG"].read_text())["feature"]["cache_dir"] == str(namespace["CACHE"])
    assert sha256_file(config) == config_hash
    assert all(sha256_file(p) == h for p, h in original_hashes.items())
    assert extraction.build_backbone is builder
    assert len(calls) == 4
    for name, kw in calls:
        assert kw["image_size"] == (224 if name == "siglip" else 256)
        if name == "siglip":
            assert kw["revision"] == PIN and kw["use_fast"] is False
    receipt = json.loads((run / "pinning_recovery_01.json").read_text())
    assert receipt["status"] == "validated" and receipt["rows"] == {"train": 10, "test": 6}
    with pytest.raises(FileExistsError):
        exec(compile(code, "pinned-smoke-recovery", "exec"), namespace)
