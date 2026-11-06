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
EMPTY_DETAIL = "Technical details will appear after analysis."

APP_CSS = """
.gradio-container {
    max-width: 1140px !important; margin-inline: auto !important;
    padding: 0 clamp(16px, 3vw, 36px) !important;
}
#pmsa-header, #pmsa-image-heading, #pmsa-result, #pmsa-input-hint {
    padding: 0 !important; min-height: 0; background: transparent; border: 0;
}
.pmsa-brand {
    display: flex; align-items: center; gap: 10px; padding: 20px 0;
    border-bottom: 1px solid var(--border-color-primary);
}
.pmsa-mark {
    display: grid; place-items: center; width: 34px; height: 34px;
    border-radius: 10px; background: var(--button-primary-background-fill);
    color: var(--button-primary-text-color);
}
.pmsa-brand strong { font-size: 17px; letter-spacing: -.025em; }
.pmsa-brand span:last-child { margin-left: auto; }
.pmsa-intro { padding: 26px 0 10px; }
.pmsa-intro h1 { font-size: clamp(28px, 4vw, 44px); line-height: 1.12; letter-spacing: -.045em; margin: 0 0 8px; }
.pmsa-intro p, .pmsa-card p, .pmsa-section h2 { margin: 0; }
.pmsa-intro p, .pmsa-muted, .pmsa-card p, .pmsa-readout dt { color: var(--body-text-color-subdued); }
.pmsa-muted { font-size: 13px; }
#pmsa-workspace { gap: 18px; align-items: stretch; }
#pmsa-workspace > .column { min-width: 0 !important; }
.pmsa-panel {
    border: 1px solid var(--border-color-primary); border-radius: 18px;
    background: var(--block-background-fill); padding: clamp(16px, 2.4vw, 24px);
}
.pmsa-section { display: flex; gap: 10px; align-items: center; }
.pmsa-step {
    display: grid; place-items: center; width: 27px; height: 27px; border-radius: 8px;
    background: var(--background-fill-secondary); color: var(--body-text-color);
    font-size: 13px; font-weight: 650; font-variant-numeric: tabular-nums;
}
.pmsa-section h2 { font-size: 16px; letter-spacing: -.015em; }
#pmsa-upload { border-radius: 12px; overflow: hidden; border: 0; }
#pmsa-upload .upload-container { border-color: var(--border-color-primary); }
#pmsa-preview { border: 0; border-radius: 12px; overflow: hidden; background: var(--background-fill-secondary); }
#pmsa-preview img { object-fit: contain; }
#pmsa-analyze, #pmsa-clear { min-height: 44px; }
#pmsa-input-hint { overflow-wrap: anywhere; }
.pmsa-card { display: flex; flex-direction: column; gap: 13px; min-height: 204px;
    padding: 18px; border-radius: 13px; background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary); color: var(--body-text-color); }
.pmsa-card h3 { font-size: clamp(20px, 2.3vw, 26px); line-height: 1.15;
    letter-spacing: -.03em; margin: 0; }
.pmsa-card p { font-size: 14px; line-height: 1.52; }
.pmsa-badge { display: inline-flex; align-items: center; align-self: flex-start; gap: 7px;
    border-radius: 99px; padding: 4px 9px; border: 1px solid var(--border-color-primary);
    font-size: 12px; font-weight: 650; background: var(--block-background-fill); }
.pmsa-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--body-text-color-subdued); }
.pmsa-ai .pmsa-dot, .pmsa-error .pmsa-dot { background: #e17752; }
.pmsa-no .pmsa-dot, .pmsa-ready .pmsa-dot { background: #24a88e; }
.pmsa-uncertain .pmsa-dot { background: #d69a2c; }
.pmsa-busy .pmsa-dot { width: 10px; height: 10px; background: transparent;
    border: 2px solid var(--border-color-primary); border-top-color: var(--body-text-color);
    animation: pmsa-spin .8s linear infinite; }
@keyframes pmsa-spin { to { transform: rotate(360deg); } }
@media(prefers-reduced-motion:reduce) { .pmsa-busy .pmsa-dot { animation: none; } }
.pmsa-readout { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px; margin: 0; }
.pmsa-readout div { background: var(--block-background-fill); border-radius: 9px; padding: 8px 10px; }
.pmsa-readout dt { font-size: 12px; }
.pmsa-readout dd { margin: 2px 0 0; font-size: 19px; font-weight: 650;
    font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.pmsa-card .pmsa-disclaimer { font-size: 12px; }
#pmsa-notice { padding: 4px 0; }
@media(max-width:760px) {
    #pmsa-workspace { flex-direction: column; gap: 12px; }
    .pmsa-intro { padding-top: 18px; }
    .pmsa-panel { width: 100%; }
}
"""

HEADER = """<header>
<div class="pmsa-brand"><span class="pmsa-mark" aria-hidden="true">
<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
<path d="M3 8V4h4m10 0h4v4m0 8v4h-4m-10 0H3v-4M5 12h4l2-4 3 8 2-4h3"/>
</svg></span><strong>PMSA</strong><span class="pmsa-muted">Image forensics</span></div>
<div class="pmsa-intro"><h1>AI image analysis</h1>
<p>Check a still image for signs of AI generation.</p></div></header>"""


def state_card(state, badge, title, message):
    return (f'<section class="pmsa-card pmsa-{state}" aria-live="polite">'
            f'<span class="pmsa-badge"><span class="pmsa-dot" aria-hidden="true"></span>{escape(badge)}</span>'
            f'<h3>{escape(title)}</h3><p>{escape(message)}</p></section>')


EMPTY_RESULT = state_card("waiting", "Awaiting image", "No analysis yet",
                          "Add an image on the left or above to get started.")
READY_RESULT = state_card("ready", "Image ready", "Ready to analyze",
                          "Select Analyze image to check the original file.")
ANALYZING_RESULT = state_card("busy", "Analyzing", "Checking the image",
                              "The first analysis may take longer while the model loads.")
FAILED_RESULT = state_card("error", "Analysis failed", "Couldn’t analyze this image",
                           "Try again, or choose a different image if the problem persists.")
INPUT_HINT = '<p class="pmsa-muted">JPEG, PNG, WebP or HEIC · Up to 20 MiB</p>'


def result_card(verdict):
    """Raw score visualization, never a fabricated AI probability or confidence."""
    kind = {"AI-generated signal": "ai", "No AI signal detected": "no"}.get(verdict.label, "uncertain")
    title = {"ai": "AI-generation signal detected", "no": "No AI signal detected",
             "uncertain": "Inconclusive"}[kind]
    description = {"ai": "The score is above the decision threshold. Check the source before drawing a conclusion.",
                   "no": "The score is below the threshold. This does not establish camera origin.",
                   "uncertain": "The score is close to the threshold. Check the original image and its source."}[kind]
    return f"""<section class="pmsa-card pmsa-{kind}" aria-live="polite">
<span class="pmsa-badge"><span class="pmsa-dot" aria-hidden="true"></span>Analysis complete</span>
<h3>{title}</h3><p>{description}</p>
<dl class="pmsa-readout"><div><dt>Detector score</dt><dd>{verdict.score:+.3f}</dd></div>
<div><dt>Decision threshold</dt><dd>{verdict.tau:+.3f}</dd></div></dl>
<p class="pmsa-disclaimer">This is a model score, not a confidence percentage.</p>
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
    valid = preview is not None
    return (gr.update(value=preview, visible=valid), *result,
            gr.update(interactive=valid), gr.update(visible=not valid))


def clear_upload_ui():
    """Restore the drop target in the same panel without retaining a verdict."""
    import gradio as gr
    return (gr.update(value=None, visible=True), gr.update(value=None, visible=False),
            EMPTY_RESULT, {}, "Technical details will appear after analysis.",
            gr.update(interactive=False))


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
        with gr.Group(elem_id="pmsa-upload"):
            inp = gr.File(type="filepath", label="Drop an image here or browse · JPEG, PNG, WEBP, HEIC · up to 20 MiB",
                          file_count="single", height=180)
            preview = gr.Image(label="Your image", interactive=False, height=260,
                               buttons=[], elem_id="pmsa-preview", visible=False)
        with gr.Row():
            button = gr.Button("Analyze image", variant="primary", scale=3, interactive=False)
            clear = gr.Button("Clear / choose another", scale=1, elem_id="pmsa-clear")
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
        inp.change(prepare_upload_ui, inputs=inp, outputs=[preview, *outputs, button, inp], api_name=False,
                   queue=False, trigger_mode="always_last")
        controls = [inp, button, clear]
        button.click(begin_analysis, outputs=[*outputs, *controls], api_name=False, queue=False).then(
            classify, inputs=inp, outputs=outputs, api_name="classify", concurrency_limit=1).then(
            finish_analysis, outputs=controls, api_name=False, queue=False)
        clear.click(clear_upload_ui, outputs=[inp, preview, *outputs, button],
                    api_name=False, queue=False)
    return demo


def launch_options():
    """Gradio 6 takes styling at launch, shared by local and hosted entrypoints."""
    import gradio as gr
    return {"css": APP_CSS, "theme": gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
            font=["system-ui", "sans-serif"]), "max_file_size": "20mb"}


if __name__ == "__main__":
    build_demo().launch(**launch_options())
