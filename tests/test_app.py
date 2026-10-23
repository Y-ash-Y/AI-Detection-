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


@pytest.mark.parametrize("score,label", [(2., "AI-generated signal"),
    (-2., "No AI signal detected"), (.1, "Uncertain")])
def test_result_card_has_three_states_without_fake_probability(score, label):
    from app import render_verdict
    from pmsa.inference import Verdict
    result, metrics, detail = render_verdict(Verdict(label, score > 0, score, 0., score,
        per_stream={"siglip": -1., "forensic": score + .5}, combiner_bias=.5))
    assert 'aria-live="polite"' in result
    assert "not a confidence percentage" in result
    assert "probability calibration has not been validated" in result
    assert "AI probability" not in result
    assert metrics["Score (not a probability)"] == score
    assert "Contributions plus bias equal the score" in detail
    assert "not causal" in detail
    if label == "Uncertain":
        assert "Inconclusive" in result
        assert metrics["Binary threshold exceeded"] is True


def test_html_result_escapes_untrusted_text():
    from app import render_verdict
    from pmsa.inference import Verdict
    result, _, detail = render_verdict(Verdict("No AI signal detected", False, -1., 0., -1.,
        explanation='<img src=x onerror="alert(1)">',
        provenance={"note": "<script>bad()</script>"}))
    assert "<img src=x" not in result and "&lt;img" in result
    assert "<script>" not in detail


def test_preview_preserves_original_and_does_not_load_model(tmp_path, monkeypatch):
    import app
    path = tmp_path / "preview.png"
    Image.new("RGB", (1400, 1000), "blue").save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(app, "_load", lambda: pytest.fail("Preview must not load detector"))
    preview, result, metrics, detail = app.preview_upload(str(path))
    assert preview.width <= 900 and preview.height <= 600
    assert result == app.EMPTY_RESULT and metrics == {}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert app.preview_upload(None)[0] is None


def test_bad_preview_clears_previous_diagnostics(tmp_path):
    from app import preview_upload
    path = tmp_path / "invalid.png"
    path.write_text("not an image")
    preview, result, metrics, detail = preview_upload(str(path))
    assert preview is None and metrics == {}
    assert "Cannot preview" in result
    assert "No model analysis" in detail


def test_analysis_locks_upload_until_result_or_error():
    from app import begin_analysis, finish_analysis
    started = begin_analysis()
    assert len(started) == 6
    assert all(update["interactive"] is False for update in started[3:])
    assert all(update["interactive"] is True for update in finish_analysis())


def test_upload_only_enables_analysis_after_valid_preview(tmp_path):
    from app import prepare_upload_ui
    assert prepare_upload_ui(None)[-2]["interactive"] is False
    path = tmp_path / "valid.png"
    Image.new("RGB", (4, 5)).save(path)
    result = prepare_upload_ui(str(path))
    assert result[-2]["interactive"] is True
    assert result[0]["visible"] is True and result[-1]["visible"] is False


def test_single_upload_panel_restores_drop_target_on_clear():
    from app import clear_upload_ui, EMPTY_RESULT
    file_update, preview_update, result, diagnostics, detail, button_update = clear_upload_ui()
    assert file_update["visible"] is True and file_update["value"] is None
    assert preview_update["visible"] is False and preview_update["value"] is None
    assert result == EMPTY_RESULT and diagnostics == {}
    assert button_update["interactive"] is False


def test_demo_exposes_plain_result_and_developer_diagnostics(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "false")
    from app import build_demo, launch_options
    demo = build_demo()
    config = demo.get_config_file()
    serialized = str(config)
    assert "How reliable is it?" not in serialized
    assert "Under the hood" not in serialized
    assert "0.9105" not in serialized and "64.66%" not in serialized
    developer = next(c for c in config["components"]
                     if c["props"].get("label") == "Developer analysis")
    assert developer["type"] == "accordion" and developer["props"]["open"] is False
    assert "pmsa-preview" in serialized
    assert "pmsa-upload" in serialized
    preview = next(c for c in config["components"] if c["props"].get("elem_id") == "pmsa-preview")
    assert preview["props"]["visible"] is False
    assert any(d.get("api_name") == "classify" for d in config["dependencies"])
    assert "pmsa-card" in launch_options()["css"]
