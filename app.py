"""Gradio interface. No silent third-party fallback or calibrated probabilities."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from html import escape
from pathlib import Path
from threading import Lock
from pmsa.inference import Detector
from pmsa.preprocessing import ImagePolicy, validate_image
from pmsa.utils import get_device

_detector = None
_load_lock = Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

APP_CSS = """
.gradio-container { max-width: 1240px !important; }
.pmsa-hero { padding: 24px 0 20px; }
.pmsa-eyebrow { font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--body-text-color-subdued); font-weight: 700; }
.pmsa-hero h1 { font-size: clamp(28px,4vw,42px); letter-spacing: -.04em;
    margin: 7px 0 8px; line-height: 1.12; }
.pmsa-hero p { max-width: 720px; color: var(--body-text-color-subdued); }
.pmsa-card { border: 1px solid var(--border-color-primary); border-radius: 18px;
    padding: 24px; background: var(--block-background-fill); color: var(--body-text-color); }
.pmsa-card h2 { font-size: 26px; margin: 12px 0 10px; line-height: 1.2; }
.pmsa-card p { line-height: 1.6; margin: 8px 0; }
.pmsa-badge { display: inline-block; border-radius: 99px; padding: 6px 11px;
    font-size: 12px; font-weight: 700; background: var(--background-fill-secondary); }
.pmsa-ai { border-top: 5px solid #e07840; }
.pmsa-no { border-top: 5px solid #23a18b; }
.pmsa-uncertain { border-top: 5px solid #d4a52d; }
.pmsa-muted { color: var(--body-text-color-subdued); font-size: 13px; }
.pmsa-readout { display: flex; flex-wrap: wrap; gap: 12px 28px; margin: 18px 0; }
.pmsa-readout strong { display: block; font-size: 23px; font-variant-numeric: tabular-nums; }
.pmsa-readout span { font-size: 12px; color: var(--body-text-color-subdued); }
.pmsa-axis { position: relative; height: 12px; border-radius: 20px; margin: 30px 8px 12px;
    background: linear-gradient(90deg, #23a18b44, #d4a52d55 50%, #e0784044); }
.pmsa-threshold { position: absolute; left: 50%; height: 28px; top: -8px;
    width: 2px; background: var(--body-text-color); }
.pmsa-point { position: absolute; width: 18px; height: 18px; border-radius: 50%;
    top: -3px; transform: translateX(-50%); background: var(--body-text-color);
    border: 3px solid var(--block-background-fill); box-shadow: 0 0 0 1px var(--body-text-color); }
.pmsa-axis-labels { display: flex; justify-content: space-between; gap: 10px;
    font-size: 11px; color: var(--body-text-color-subdued); }
.pmsa-note { border-top: 1px solid var(--border-color-primary); padding-top: 12px;
    margin-top: 18px !important; font-size: 13px; }
#pmsa-preview { border-radius: 14px; }
@media(max-width:640px) { .pmsa-card {padding:18px;} .pmsa-axis-labels {font-size:10px;} }
"""

EMPTY_RESULT = """<section class="pmsa-card" aria-live="polite">
<span class="pmsa-badge">READY WHEN YOU ARE</span><h2>What does your image reveal?</h2>
<p>Upload a still image and choose <strong>Analyze image</strong>.</p>
<p class="pmsa-muted">Your result will appear here. Optional technical details are in Developer analysis below.
The first analysis may take longer while the model loads.</p></section>"""


def result_card(verdict):
    """Raw score visualization, never a fabricated AI probability or confidence."""
    kind = {"AI-generated signal": "ai", "No AI signal detected": "no"}.get(verdict.label, "uncertain")
    title = {"ai": "AI-generation signal detected", "no": "No AI signal detected",
             "uncertain": "Inconclusive — review recommended"}[kind]
    advice = {"ai": "Treat this as a reason to check the image’s source, not a final accusation.",
              "no": "This is not proof the image is camera-original. Some generated images evade detection.",
              "uncertain": "The score is close to the decision boundary. Check the original file and its source."}[kind]
    span = max(1., abs(verdict.margin) * 1.25)
    position = 50 + 50 * verdict.margin / span
    low, high = verdict.tau - span, verdict.tau + span
    return f"""<section class="pmsa-card pmsa-{kind}" aria-live="polite">
<span class="pmsa-badge">ANALYSIS COMPLETE</span>
<h2>{escape(title)}</h2><p>{escape(verdict.explanation)}</p>
<div class="pmsa-readout"><div><strong>{verdict.score:+.3f}</strong><span>Detection score</span></div>
<div><strong>{verdict.tau:+.3f}</strong><span>Decision threshold</span></div>
<div><strong>{verdict.margin:+.3f}</strong><span>Distance from threshold</span></div></div>
<div role="img" aria-label="Score {verdict.score:.4f}, threshold {verdict.tau:.4f}; raw score units, not probability">
<div class="pmsa-axis"><span class="pmsa-threshold"></span>
<span class="pmsa-point" style="left:{position:.3f}%"></span></div>
<div class="pmsa-axis-labels"><span>{low:+.2f} · Less AI-like</span>
<span>Threshold {verdict.tau:+.2f}</span><span>More AI-like · {high:+.2f}</span></div></div>
<p class="pmsa-muted">Raw model-score scale; axis adapts to this result. It is not a confidence percentage.</p>
<p class="pmsa-note"><strong>What to do next:</strong> {escape(advice)}</p>
<p class="pmsa-muted">Confidence percentage: unavailable — probability calibration has not been validated.</p>
</section>"""


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
    headline = result_card(verdict)
    metrics = {"Score (not a probability)": verdict.score, "Threshold": verdict.tau,
               "Score minus threshold": verdict.margin,
               "Binary threshold exceeded": verdict.calibrated_fake,
               "Artifact status": verdict.artifact_status}
    names = {"siglip": "SigLIP · learned visual features", "forensic": "Forensic · image statistics"}
    contributions = "\n".join(
        f"| {escape(names.get(k, k)).replace('|', '&#124;')} | {value:+.4f} |"
        for k, value in verdict.per_stream.items())
    detail = (
        "### How this decision was made\n\n"
        "**1 · Extract features.** A frozen visual encoder captures learned image patterns; "
        "the forensic stream measures spectral, color and residual statistics.\n\n"
        "**2 · Combine evidence.** The trained heads produce signed contributions. "
        "Positive values raise the AI score; negative values lower it. These are "
        "model terms, not causal explanations or detected flaws.\n\n"
        f"| Stream | Signed contribution |\n|---|---:|\n{contributions}\n"
        f"| Combiner bias | {verdict.combiner_bias:+.4f} |\n"
        f"| **Final score** | **{verdict.score:+.4f}** |\n\n"
        "Contributions plus bias equal the score. A stream’s sign alone does not determine the verdict.\n\n"
        f"**3 · Compare with threshold.** Score `{verdict.score:+.4f}` versus "
        f"threshold `{verdict.tau:+.4f}`; margin `{verdict.margin:+.4f}`. "
        "Near-threshold abstention uses a heuristic score band, not a calibrated uncertainty interval.\n\n"
        f"**Metadata hints:** {escape(verdict.provenance.get('note', ''))} "
        "These hints are unverified and do not override the score.\n\n"
        f"Calibration reference domain: `{verdict.real_domain}`; target false-positive rate: "
        f"{format(verdict.target_fpr, '.1%') if verdict.target_fpr is not None else 'not established'}. "
        "This is **not** the probability that this image is AI, "
        "and is not guaranteed on a shifted domain.\n\n"
        "Scope: still images; AI edits, logos, screenshots and unfamiliar generators may be misclassified."
    )
    return headline, metrics, detail


@contextmanager
def decoded_upload(path):
    """Validate/decode originals before preview or expensive model loading."""
    from PIL import Image
    if Path(path).stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the 20 MiB file limit")
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as image:
        validate_image(image, ImagePolicy())
        image.load()
        yield image


def preview_upload(path):
    """Preview a bounded copy; leave the original detector input unchanged."""
    empty = "Technical details will appear after analysis."
    if not path:
        return None, EMPTY_RESULT, {}, empty
    from PIL import Image, ImageOps
    try:
        with decoded_upload(path) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((900, 600))
        return preview, EMPTY_RESULT, {}, empty
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        return None, (f'<section class="pmsa-card"><h2>Cannot preview this file</h2>'
                      f'<p>{escape(str(exc))}</p></section>'), {}, "No model analysis performed."


def reset_result():
    return EMPTY_RESULT, {}, "Analyzing image… Technical details will appear here."


def prepare_upload_ui(path):
    import gradio as gr
    preview, *result = preview_upload(path)
    return preview, *result, gr.update(interactive=preview is not None)


def begin_analysis():
    import gradio as gr
    # Freeze this upload until its result/error is returned, preventing a result
    # for a previous file from being shown beside a newly selected image.
    return (*reset_result(), *(gr.update(interactive=False) for _ in range(3)))


def finish_analysis():
    import gradio as gr
    return tuple(gr.update(interactive=True) for _ in range(3))


def classify(path):
    if not path:
        return EMPTY_RESULT, {}, "Upload an image first."
    import gradio as gr
    from PIL import Image, UnidentifiedImageError
    try:
        with decoded_upload(path) as image:
            # Reject invalid/oversized/multi-frame data before loading weights.
            # No EXIF/RGB/JPEG transform here: Detector owns the recorded policy.
            return render_verdict(_load().predict(image, image_path=path))
    except (ValueError, FileNotFoundError, UnidentifiedImageError,
            Image.DecompressionBombError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logging.exception("Image analysis failed")
        raise gr.Error("Analysis failed. Check server logs and trusted model configuration.") from exc


def build_demo():
    import gradio as gr
    with gr.Blocks(title="PMSA | Image Forensics", delete_cache=(3600, 3600), analytics_enabled=False) as demo:
        gr.HTML('<header class="pmsa-hero"><div class="pmsa-eyebrow">PMSA / IMAGE FORENSICS</div>'
                '<h1>A closer look at what’s real.</h1><p>Check an image for AI-generation signals. '
                'Upload a photo, run an analysis, and see the result.</p></header>')
        gr.Markdown("**A signal, not proof.** Camera photos can be falsely flagged, and AI images can be missed. "
                    "Use this tool to support a review—not to establish someone’s honesty.")
        with gr.Row(equal_height=True):
            with gr.Column(scale=6, min_width=320):
                inp = gr.File(type="filepath", label="Upload JPEG, PNG, WEBP or HEIC · up to 20 MiB",
                              file_count="single", height=150)
            with gr.Column(scale=5, min_width=320):
                preview = gr.Image(label="Image preview", interactive=False, height=150,
                                   buttons=[], elem_id="pmsa-preview")
        with gr.Row():
            button = gr.Button("Analyze image", variant="primary", scale=3, interactive=False)
            clear = gr.Button("Clear", scale=1, elem_id="pmsa-clear")
        label = gr.HTML(EMPTY_RESULT)
        with gr.Accordion("Developer analysis", open=False):
            detail = gr.Markdown("Technical details will appear after analysis.")
            with gr.Accordion("Raw diagnostics", open=False):
                metrics = gr.JSON(label="Raw score and threshold · not confidence")
        with gr.Accordion("Privacy & model credits", open=False):
            gr.Markdown("The app does not train on or publish uploads. Gradio temporarily caches files; "
                        "scheduled cleanup is approximately hourly, not immediate erasure. Hosting infrastructure "
                        "may retain logs. Avoid sensitive uploads to public demos.\n\n"
                        "Google supplies the frozen SigLIP backbone. PMSA supplies the trained detection heads, "
                        "forensic features, calibration and evaluation pipeline. No automatic third-party detector fallback.")
        outputs = [label, metrics, detail]
        inp.change(prepare_upload_ui, inputs=inp, outputs=[preview, *outputs, button], api_name=False,
                   queue=False, trigger_mode="always_last")
        controls = [inp, button, clear]
        button.click(begin_analysis, outputs=[*outputs, *controls], api_name=False, queue=False).then(
            classify, inputs=inp, outputs=outputs, api_name="classify", concurrency_limit=1).then(
            finish_analysis, outputs=controls, api_name=False, queue=False)
        clear.click(lambda: (None, None, EMPTY_RESULT, {}, "Technical details will appear after analysis."),
                    outputs=[inp, preview, *outputs], api_name=False, queue=False)
    return demo


def launch_options():
    """Gradio 6 takes styling at launch, shared by local and hosted entrypoints."""
    import gradio as gr
    return {"css": APP_CSS, "theme": gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
            font=["system-ui", "sans-serif"]), "max_file_size": "20mb"}


if __name__ == "__main__":
    build_demo().launch(**launch_options())
