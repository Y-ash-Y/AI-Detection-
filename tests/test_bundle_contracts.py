"""Fail-closed artifact contracts. All fixtures are synthetic and offline."""
import json
from importlib.metadata import version

import numpy as np
import pytest

from pmsa.artifacts import save_bundle, verify_bundle, validate_stream_metadata
from pmsa.calibration import Calibrator
from pmsa.preprocessing import ImagePolicy


def placeholder_bundle(path):
    for name in ("fusion.pt", "calibrator.json", "splits.json"):
        (path / name).write_text("{}")
    return save_bundle(path, "fusion.pt", "fusion", {"test": metadata()})


def metadata():
    return {"schema_version": 1, "image_policy": ImagePolicy().to_dict(),
            "packages": {"numpy": version("numpy")}, "backbone_kwargs": {}}


@pytest.mark.parametrize("payload", [[], None, {}, {"schema_version": True, "model_type": "fusion"}])
def test_bundle_schema_errors_are_actionable(tmp_path, payload):
    (tmp_path / "bundle.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unsupported model bundle"):
        verify_bundle(tmp_path)


def test_bundle_rejects_malformed_json_and_missing_artifact(tmp_path):
    (tmp_path / "bundle.json").write_text("{")
    with pytest.raises(ValueError, match="Cannot read bundle.json"):
        verify_bundle(tmp_path)
    placeholder_bundle(tmp_path)
    (tmp_path / "fusion.pt").unlink()
    with pytest.raises(ValueError, match="Missing bundled artifact: fusion.pt"):
        verify_bundle(tmp_path)


@pytest.mark.parametrize("field,value,message", [
    ("files", [], "Incomplete artifact hashes"),
    ("streams", [], "nonempty stream"),
    ("streams", {}, "nonempty stream"),
    ("model_file", "../fusion.pt", "filename disagree"),
])
def test_bundle_rejects_wrong_structure(tmp_path, field, value, message):
    bundle = placeholder_bundle(tmp_path)
    bundle[field] = value
    (tmp_path / "bundle.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match=message):
        verify_bundle(tmp_path)


@pytest.mark.parametrize("digest", [None, 42, "", "z"*64, "A"*64])
def test_digest_is_a_lowercase_sha256(tmp_path, digest):
    bundle = placeholder_bundle(tmp_path)
    bundle["files"]["fusion.pt"] = digest
    (tmp_path / "bundle.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="Invalid SHA256"):
        verify_bundle(tmp_path)


def test_artifact_symlink_cannot_escape_bundle(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    placeholder_bundle(folder)
    target = tmp_path / "external.pt"
    target.write_text("{}")
    (folder / "fusion.pt").unlink()
    (folder / "fusion.pt").symlink_to(target)
    with pytest.raises(ValueError, match="escapes bundle"):
        verify_bundle(folder)


@pytest.mark.parametrize("field,value", [("image_policy", None), ("image_policy", {"unexpected": True}),
    ("packages", {"numpy": None}), ("backbone_kwargs", []), ("packages", [])])
def test_bad_extraction_metadata_fails_explicitly(field, value):
    current = metadata()
    current[field] = value
    with pytest.raises(ValueError):
        validate_stream_metadata({"test": current})


def test_neural_processor_mode_requires_boolean():
    current = metadata()
    current["backbone_kwargs"] = {"weights": "fixture", "revision": "fixed", "use_fast": "false"}
    with pytest.raises(ValueError, match="Unpinned"):
        validate_stream_metadata({"siglip": current})


def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from pmsa.models import FusionDetector, StreamSpec
    model = FusionDetector([StreamSpec("test", 3, 4)])
    model._fit_norm([np.arange(18, dtype=np.float32).reshape(6,3)])
    path = tmp_path / "fusion.pt"
    model.save(path)
    return torch.load(path, weights_only=True), path


@pytest.mark.parametrize("defect,message", [
    ("nonfinite", "nonfinite fusion weights"),
    ("version", "Unsupported fusion checkpoint format"),
    ("missing", "Incomplete fusion checkpoint"),
    ("spec", "Invalid fusion stream specifications"),
    ("shape", "weights do not match"),
])
def test_invalid_tensor_checkpoints_are_rejected(tmp_path, defect, message):
    torch = pytest.importorskip("torch")
    from pmsa.models import FusionDetector
    blob, path = checkpoint(tmp_path)
    if defect == "nonfinite":
        blob["state"]["combiner.bias"][0] = float("nan")
    elif defect == "version":
        blob["format_version"] = 999
    elif defect == "missing":
        del blob["mean"]
    elif defect == "spec":
        blob["specs"] = [("test", 0, 4)]
    else:
        blob["specs"] = [("test", 4, 4)]
    torch.save(blob, path)
    with pytest.raises(ValueError, match=message):
        FusionDetector.load(path)


def test_empty_or_duplicate_streams_rejected():
    pytest.importorskip("torch")
    from pmsa.models import FusionDetector, StreamSpec
    for specs in ([], [StreamSpec("x",3), StreamSpec("x",3)]):
        with pytest.raises(ValueError, match="unique feature streams"):
            FusionDetector(specs)


def test_backbone_and_head_dimension_mismatch_is_explicit(tmp_path, monkeypatch):
    from pmsa.inference import Detector
    checkpoint(tmp_path)
    Calibrator(0., .05, "empirical", 20, "synthetic", False).save(tmp_path / "calibrator.json")
    (tmp_path / "splits.json").write_text("{}")
    save_bundle(tmp_path, "fusion.pt", "fusion", {"test": metadata()})
    class WrongEncoder:
        dim = 4
        def encode(self, images):
            pytest.fail("Dimension mismatch should fail before encoding")
    monkeypatch.setattr("pmsa.inference.build_backbone", lambda *a, **k: WrongEncoder())
    with pytest.raises(ValueError, match="Backbone/head dimension mismatch"):
        Detector(tmp_path / "fusion.pt", tmp_path / "calibrator.json")
