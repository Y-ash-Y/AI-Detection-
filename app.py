"""Gradio interface. No silent third-party fallback or calibrated probabilities."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from pmsa.inference import Detector
from pmsa.preprocessing import ImagePolicy, validate_image
from pmsa.utils import get_device

_detector = None
_load_lock = Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _load():
    global _detector
    with _load_lock:
        if _detector is None:
            directory = Path(os.getenv("PMSA_BUNDLE", "outputs/deploy_v3"))
            legacy = os.getenv("PMSA_ALLOW_LEGACY") == "1"
            checkpoint = Path(os.getenv("PMSA_CKPT", str(directory / "fusion.pt")))
            calibrator = Path(os.getenv("PMSA_CAL", str(directory / "calibrator.json")))
            policy = None
            if legacy:
                value = os.getenv("PMSA_LEGACY_NORMALIZE_JPEG")
                if value not in ("0", "1"):
                    raise ValueError("Legacy mode requires PMSA_LEGACY_NORMALIZE_JPEG=0 or 1")
                policy = ImagePolicy(normalize_jpeg=value == "1")
            if "PMSA_CKPT" not in os.environ and (directory / "bundle.json").exists():
                from pmsa.artifacts import verify_bundle
                checkpoint = directory / verify_bundle(directory)["model_file"]
            _detector = Detector(checkpoint, calibrator, device=get_device(os.getenv("PMSA_DEVICE")),
                                 allow_legacy=legacy, legacy_policy=policy,
                                 trusted_legacy=os.getenv("PMSA_TRUSTED_LEGACY") == "1")
    return _detector


def render_verdict(verdict):
    """Formatting is testable without model downloads."""
    headline = f"## {verdict.label}\n\n{verdict.explanation}"
    metrics = {"Score (not a probability)": verdict.score, "Threshold": verdict.tau,
               "Score minus threshold": verdict.margin,
               "Binary threshold exceeded": verdict.calibrated_fake,
               "Artifact status": verdict.artifact_status}
    contributions = "\n".join(f"- `{k}`: {value:+.4f}" for k, value in verdict.per_stream.items())
    detail = (
        f"Metadata: {verdict.provenance.get('note', '')}\n\n"
        f"Signed score contributions (not causal explanations):\n\n{contributions}\n\n"
        f"Combiner bias: {verdict.combiner_bias:+.4f}. Contributions plus bias equal the score.\n\n"
        f"Calibration reference domain: `{verdict.real_domain}`; target false-positive rate: "
        f"{format(verdict.target_fpr, '.1%') if verdict.target_fpr is not None else 'not established'}. "
        "This is **not** the probability that this image is AI, "
        "and is not guaranteed on a shifted domain.\n\n"
        "Scope: still images; AI edits, logos, screenshots and unfamiliar generators may be misclassified."
    )
    return headline, metrics, detail


def classify(path):
    if not path:
        return "Upload an image.", {}, ""
    import gradio as gr
    from PIL import Image, UnidentifiedImageError
    try:
        if Path(path).stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError("Upload exceeds the 20 MiB file limit")
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(path) as image:
            # Reject invalid/oversized/multi-frame data before loading weights.
            # No EXIF/RGB/JPEG transform here: Detector owns the recorded policy.
            validate_image(image, ImagePolicy())
            image.load()
            return render_verdict(_load().predict(image, image_path=path))
    except (ValueError, FileNotFoundError, UnidentifiedImageError,
            Image.DecompressionBombError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logging.exception("Image analysis failed")
        raise gr.Error("Analysis failed. Check server logs and trusted model configuration.") from exc


def build_demo():
    import gradio as gr
    with gr.Blocks(title="PMSA — image forensic detector", delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# PMSA — image forensic detector\n"
                    "Custom trained detection head on frozen features. Results are signals, not proof. "
                    "No probability-calibration claim or automatic external detector fallback.")
        gr.Markdown("**Privacy:** the app does not train on or publish uploads. Gradio temporarily "
                    "caches uploaded files for processing; scheduled cleanup is approximately hourly. "
                    "Hosting infrastructure may retain logs. Avoid sensitive uploads to public demos.")
        with gr.Row():
            with gr.Column():
                inp = gr.File(type="filepath", label="Still image (JPEG, PNG, WEBP or HEIC; up to 20 MiB)")
                button = gr.Button("Analyze", variant="primary")
            with gr.Column():
                label = gr.Markdown()
                metrics = gr.JSON(label="Decision diagnostics — not confidence")
                detail = gr.Markdown()
        button.click(classify, inputs=inp, outputs=[label, metrics, detail])
    return demo


if __name__ == "__main__":
    build_demo().launch(max_file_size="20mb")
