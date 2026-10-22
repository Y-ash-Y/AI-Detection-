"""Interface smoke tests without a real detector or external service."""
import hashlib

import pytest
from PIL import Image

gradio = pytest.importorskip("gradio")


def test_demo_builds(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "false")
    from app import build_demo
    demo = build_demo()
    assert isinstance(demo, gradio.Blocks)


def test_original_upload_reaches_detector_unchanged(tmp_path, monkeypatch):
    import app
    from pmsa.inference import Verdict
    path = tmp_path / "original.png"
    Image.new("RGB", (5, 6), "blue").save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    class Stub:
        def predict(self, image, image_path):
            assert image_path == str(path)
            assert image.size == (5, 6)
            return Verdict("No AI signal detected", False, -2., 0., -2.,
                           target_fpr=.05, per_stream={"toy": -2.})
    monkeypatch.setattr(app, "_load", lambda: Stub())
    headline, diagnostics, detail = app.classify(str(path))
    assert "No AI signal detected" in headline and "AI probability" not in headline
    assert diagnostics["Binary threshold exceeded"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_invalid_upload_fails_before_detector_load(tmp_path, monkeypatch):
    import app
    path = tmp_path / "bad.png"
    path.write_text("not an image")
    monkeypatch.setattr(app, "_load", lambda: pytest.fail("Detector must not load on invalid upload"))
    with pytest.raises(gradio.Error):
        app.classify(str(path))


def test_oversized_upload_rejected(tmp_path, monkeypatch):
    import app
    path = tmp_path / "too-big.png"
    path.write_bytes(b"a" * 10)
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", 5)
    monkeypatch.setattr(app, "_load", lambda: pytest.fail("Detector must not load"))
    with pytest.raises(gradio.Error, match="20 MiB"):
        app.classify(str(path))


def test_excessive_pixels_rejected_before_model_load(tmp_path, monkeypatch):
    import app
    from pmsa.preprocessing import ImagePolicy
    path = tmp_path / "pixels.png"
    Image.new("RGB", (11, 10)).save(path)
    monkeypatch.setattr(app, "ImagePolicy", lambda: ImagePolicy(max_pixels=100))
    monkeypatch.setattr(app, "_load", lambda: pytest.fail("Model must not load on oversized pixels"))
    with pytest.raises(gradio.Error, match="pixel limit"):
        app.classify(str(path))


def test_multiframe_rejected_before_model_load(tmp_path, monkeypatch):
    import app
    path = tmp_path / "animation.gif"
    Image.new("RGB", (5,5), "red").save(path, save_all=True,
        append_images=[Image.new("RGB", (5,5), "blue")])
    monkeypatch.setattr(app, "_load", lambda: pytest.fail("Model must not load on animation"))
    with pytest.raises(gradio.Error, match="Multi-frame"):
        app.classify(str(path))


def test_truncated_pixels_rejected_before_model_load(tmp_path, monkeypatch):
    import app
    path = tmp_path / "truncated.bmp"
    Image.new("RGB", (20,20), "red").save(path)
    path.write_bytes(path.read_bytes()[:-300])
    # The header is valid, but loading pixels must fail before loading a model.
    with Image.open(path) as opened:
        assert opened.size == (20,20)
    monkeypatch.setattr(app, "_load", lambda: pytest.fail("Model must not load on truncated pixels"))
    with pytest.raises(gradio.Error):
        app.classify(str(path))


def test_missing_bundle_does_not_use_external_fallback(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "_detector", None)
    monkeypatch.setenv("PMSA_BUNDLE", str(tmp_path))
    for key in ("PMSA_CKPT", "PMSA_CAL", "PMSA_ALLOW_LEGACY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="Legacy artifacts"):
        app._load()
