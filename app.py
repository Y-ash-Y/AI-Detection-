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
    --pmsa-blue: #0875e1;
    --pmsa-label: #805300;
    width: 100% !important; max-width: 1140px !important; margin-inline: auto !important;
    padding: 0 clamp(16px, 3vw, 36px) !important;
}
.dark .gradio-container, .gradio-container.dark {
    --pmsa-label: #ffdc24;
}
#pmsa-header, #pmsa-image-heading, #pmsa-result, #pmsa-input-hint {
    padding: 0 !important; min-height: 0; background: transparent; border: 0;
}
/* Gradio adds inner HTML padding even when the outer block has no padding. */
#pmsa-header .html-container { padding-inline: 0 !important; }
.pmsa-brand {
    display: flex; align-items: center; gap: 10px; padding: 20px 0;
    border-bottom: 1px solid var(--border-color-primary);
}
.pmsa-mark {
    display: grid; place-items: center; width: 34px; height: 34px;
    border-radius: 10px; background: var(--pmsa-blue);
    color: var(--button-primary-text-color);
}
#pmsa-header .pmsa-mark svg path { stroke: #fff !important; }
#pmsa-header .pmsa-brand-name {
    min-width: 0; font-size: clamp(16px, 2.2vw, 25px);
    line-height: 1.25; letter-spacing: -.02em;
}
#pmsa-header .pmsa-brand-name strong { font-size: inherit; }
.pmsa-intro {
    margin-top: 16px; padding: clamp(20px, 3vw, 32px);
    border-radius: 18px; color: #f4fbff;
    background: var(--pmsa-blue);
}
#pmsa-header .pmsa-intro h1 {
    color: #f4fbff !important; font-size: clamp(30px, 5vw, 54px);
    line-height: 1.12; letter-spacing: -.045em; margin: 0 0 8px;
}
.pmsa-intro p, .pmsa-card p { margin: 0; }
#pmsa-header .pmsa-intro p { color: #fff !important; }
.pmsa-muted, .pmsa-card p, .pmsa-readout dt { color: var(--body-text-color-subdued); }
.pmsa-muted { font-size: 13px; }
#pmsa-workspace { gap: 18px; align-items: flex-start; }
#pmsa-workspace > .column { min-width: 0 !important; }
.pmsa-panel {
    border: 1px solid var(--border-color-primary); border-radius: 18px;
    background: var(--block-background-fill); padding: clamp(16px, 2.4vw, 24px);
    align-self: flex-start; justify-content: flex-start !important;
}
#pmsa-workspace h2.pmsa-section {
    display: flex; gap: 10px; align-items: center;
    margin: 0 !important; line-height: 1.25 !important;
    color: var(--body-text-color) !important;
    font-size: 16px; letter-spacing: -.015em;
}
.pmsa-step {
    display: grid; place-items: center; width: 27px; height: 27px; border-radius: 8px;
    flex: 0 0 27px;
    background: var(--background-fill-secondary); color: var(--body-text-color);
    font-size: 13px; font-weight: 650; font-variant-numeric: tabular-nums;
}
#pmsa-upload { border-radius: 12px; overflow: hidden; border: 0; }
#pmsa-file { border: 1.5px dashed var(--body-text-color-subdued); border-radius: 12px; }
#pmsa-file:hover, #pmsa-file:focus-within { border-color: var(--button-primary-background-fill); }
#pmsa-preview { border: 0; border-radius: 12px; overflow: hidden; background: var(--background-fill-secondary); }
#pmsa-preview img { object-fit: contain; }
#pmsa-analyze, #pmsa-clear { min-height: 44px; }
#pmsa-analyze:not(:disabled) {
    background: var(--pmsa-blue) !important; color: #fff !important;
}
#pmsa-input-hint { overflow-wrap: anywhere; }
.pmsa-card { display: flex; flex-direction: column; gap: 13px; min-height: 204px;
    padding: 18px; border-radius: 13px; background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary); color: var(--body-text-color); }
.pmsa-card h3 { font-size: clamp(20px, 2.3vw, 26px); line-height: 1.15;
    letter-spacing: -.03em; margin: 0; }
.pmsa-card p { font-size: 14px; line-height: 1.52; }
.pmsa-reason { display: grid; gap: 4px; border-top: 1px solid var(--border-color-primary); padding-top: 11px; }
.pmsa-reason strong { font-size: 13px; font-weight: 700; }
.pmsa-card .pmsa-reason p { font-size: 13px; }
.pmsa-badge { display: inline-flex; align-items: center; align-self: flex-start; gap: 7px;
    border-radius: 99px; padding: 4px 9px; border: 1px solid var(--border-color-primary);
    font-size: 12px; font-weight: 650; background: var(--block-background-fill); }
.pmsa-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--body-text-color-subdued); }
.pmsa-waiting .pmsa-dot, .pmsa-ready .pmsa-dot,
.pmsa-ai .pmsa-dot, .pmsa-no .pmsa-dot, .pmsa-uncertain .pmsa-dot {
    background: var(--pmsa-label);
}
.pmsa-error .pmsa-dot { background: #e17752; }
.pmsa-busy .pmsa-dot { width: 10px; height: 10px; background: transparent;
    border: 2px solid var(--border-color-primary); border-top-color: var(--pmsa-label);
    animation: pmsa-spin .8s linear infinite; }
@keyframes pmsa-spin { to { transform: rotate(360deg); } }
@media(prefers-reduced-motion:reduce) { .pmsa-busy .pmsa-dot { animation: none; } }
.pmsa-readout { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px; margin: 0; }
.pmsa-readout div { background: var(--block-background-fill); border-radius: 9px; padding: 8px 10px; }
#pmsa-result .pmsa-readout dt {
    color: var(--pmsa-label) !important; font-size: 12px;
}
.pmsa-readout dd { margin: 2px 0 0; font-size: 19px; font-weight: 650;
    font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.pmsa-card .pmsa-disclaimer { font-size: 12px; }
#pmsa-notice { padding: 4px 0; }
@media(max-width:760px) {
    .gradio-container { padding-inline: 0 !important; }
    #pmsa-workspace { flex-direction: column; gap: 12px; }
    .pmsa-panel { width: 100%; }
}
"""

HEADER = """<header>
<div class="pmsa-brand"><span class="pmsa-mark" aria-hidden="true">
<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
<path d="M3 8V4h4m10 0h4v4m0 8v4h-4m-10 0H3v-4M5 12h4l2-4 3 8 2-4h3"/>
</svg></span><div class="pmsa-brand-name"><strong>PMSA :</strong>
<span class="pmsa-brand-expansion">Probabilistic Media Synthesis &amp; Authentication</span>
</div></div>
<div class="pmsa-intro"><h1>AI Image Detector</h1>
<p>PMSA combines learned visual features with image forensics to flag signs of AI generation.</p>
</div></header>"""


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


def plain_explanation(verdict):
    """Describe only the actual direction of each known score contribution."""
    from math import isfinite

    names = {
        "siglip": "The check of how the image looks as a whole",
        "forensic": "The check of color, texture, and fine pixel patterns",
    }
    known = [(name, value) for name, value in verdict.per_stream.items()
             if name in names and isfinite(value) and value != 0]
    if not known:
        return "There’s no separate breakdown of what raised or lowered this score."
    known.sort(key=lambda entry: -abs(entry[1]))
    return " ".join(f"{names[name]} {'raised' if value > 0 else 'lowered'} the score."
                    for name, value in known)


def result_card(verdict):
    """Show the verdict and numeric decision boundary without a probability."""
    kind = {"AI-generated signal": "ai", "No AI signal detected": "no"}.get(verdict.label, "uncertain")
    title = {"ai": "AI-generation signal detected", "no": "No AI signal detected",
             "uncertain": "Inconclusive"}[kind]
    description = {"ai": "The score is above the decision threshold, so the image may be AI-generated.",
                   "no": "The score is below the decision threshold, so the image may be real.",
                   "uncertain": "The score is close to the threshold, so PMSA can’t tell whether the image is real or AI-generated."}[kind]
    legacy_note = ('<p class="pmsa-disclaimer">Legacy artifacts: model pairing and preprocessing are unverified.</p>'
                   if verdict.artifact_status == "legacy-unverified" else "")
    return f"""<section class="pmsa-card pmsa-{kind}" aria-live="polite">
<span class="pmsa-badge"><span class="pmsa-dot" aria-hidden="true"></span>Analysis complete</span>
<h3>{title}</h3><p>{description}</p>
<div class="pmsa-reason"><strong>Why this result?</strong>
<p>{escape(plain_explanation(verdict))}</p>
<p class="pmsa-disclaimer">These checks don’t identify specific flaws or show where they are.</p></div>
<dl class="pmsa-readout"><div><dt>Detector score</dt><dd>{verdict.score:+.3f}</dd></div>
<div><dt>Decision threshold</dt><dd>{verdict.tau:+.3f}</dd></div></dl>
<p class="pmsa-disclaimer">This is a model score, not a confidence percentage.</p>
{legacy_note}
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
    if not path:
        return None, EMPTY_RESULT, {}, EMPTY_DETAIL, INPUT_HINT
    from PIL import Image, ImageOps
    try:
        with decoded_upload(path) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            width, height = preview.size
            preview.thumbnail((900, 600))
        info = (f'<p class="pmsa-muted">{escape(Path(path).name)}'
                f' · {width:,} × {height:,} px</p>')
        return preview, READY_RESULT, {}, EMPTY_DETAIL, info
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "Choose a supported, readable still image."
        return (None, state_card("error", "Invalid image", "Cannot preview this file", reason),
                {}, "No model analysis performed.", INPUT_HINT)


def prepare_upload_ui(path):
    import gradio as gr
    preview, card, metrics, detail, hint = preview_upload(path)
    valid = preview is not None
    return (gr.update(value=preview, visible=valid), card, metrics, detail, hint,
            gr.update(visible=not valid), gr.update(interactive=valid),
            gr.update(interactive=bool(path)))


def clear_upload_ui():
    """Restore the drop target in the same panel without retaining a verdict."""
    import gradio as gr
    return (gr.update(value=None, visible=True), gr.update(value=None, visible=False),
            EMPTY_RESULT, {}, EMPTY_DETAIL, INPUT_HINT,
            gr.update(interactive=False), gr.update(interactive=False))


def begin_analysis():
    import gradio as gr
    # Freeze this upload until its result/error is returned, preventing a result
    # for a previous file from being shown beside a newly selected image.
    return (ANALYZING_RESULT, {}, "Analysis in progress.",
            *(gr.update(interactive=False) for _ in range(3)))


def finish_analysis():
    import gradio as gr
    return tuple(gr.update(interactive=True) for _ in range(3))


def fail_analysis():
    return FAILED_RESULT, {}, "No result available.", *finish_analysis()


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
    except (UnidentifiedImageError, FileNotFoundError, Image.DecompressionBombError, OSError) as exc:
        raise gr.Error("Could not read this image. Choose a supported, readable still image.") from exc
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logging.exception("Image analysis failed")
        raise gr.Error("Analysis failed. Check server logs and trusted model configuration.") from exc


def build_demo():
    import gradio as gr
    with gr.Blocks(title="PMSA | Image Forensics", fill_width=True,
                   delete_cache=(3600, 3600), analytics_enabled=False) as demo:
        gr.HTML(HEADER, elem_id="pmsa-header", padding=False)
        with gr.Row(equal_height=False, elem_id="pmsa-workspace"):
            with gr.Column(scale=1, min_width=0, elem_classes="pmsa-panel", elem_id="pmsa-input-panel"):
                gr.HTML('<h2 class="pmsa-section"><span class="pmsa-step" aria-hidden="true">1</span>'
                        '<span>Input image</span></h2>', elem_id="pmsa-image-heading", padding=False)
                with gr.Group(elem_id="pmsa-upload"):
                    inp = gr.File(type="filepath", label="Upload image", show_label=False,
                                  file_count="single", height=204, elem_id="pmsa-file")
                    preview = gr.Image(label="Uploaded image preview", show_label=False, interactive=False,
                                       height=204, buttons=[], elem_id="pmsa-preview", visible=False)
                hint = gr.HTML(INPUT_HINT, elem_id="pmsa-input-hint", padding=False)
                with gr.Row():
                    button = gr.Button("Analyze image", variant="primary", scale=2, min_width=135,
                                       interactive=False, elem_id="pmsa-analyze")
                    clear = gr.Button("Clear", scale=1, min_width=80, interactive=False, elem_id="pmsa-clear")
            with gr.Column(scale=1, min_width=0, elem_classes="pmsa-panel", elem_id="pmsa-result-panel"):
                gr.HTML('<h2 class="pmsa-section"><span class="pmsa-step" aria-hidden="true">2</span>'
                        '<span>Analysis result</span></h2>', padding=False)
                label = gr.HTML(EMPTY_RESULT, elem_id="pmsa-result", padding=False)
        gr.Markdown("**A signal, not proof.** Real photos can be flagged, and generated images can be missed.",
                    elem_id="pmsa-notice")
        with gr.Accordion("Developer analysis", open=False, elem_id="pmsa-developer"):
            detail = gr.Markdown(EMPTY_DETAIL)
            with gr.Accordion("Raw diagnostics", open=False):
                metrics = gr.JSON(label="Raw score and threshold · not confidence")
        with gr.Accordion("Privacy & model credits", open=False, elem_id="pmsa-privacy"):
            gr.Markdown("The app does not train on or publish uploads. Gradio temporarily caches files; "
                        "scheduled cleanup is approximately hourly, not immediate erasure. Hosting infrastructure "
                        "may retain logs. Avoid sensitive uploads to public demos.\n\n"
                        "Google supplies the frozen SigLIP backbone. PMSA supplies the trained detection heads, "
                        "forensic features, calibration and evaluation pipeline. No automatic third-party detector fallback.")
        outputs = [label, metrics, detail]
        inp.upload(prepare_upload_ui, inputs=inp, outputs=[preview, *outputs, hint, inp, button, clear], api_name=False,
                   queue=False, trigger_mode="always_last")
        controls = [inp, button, clear]
        analysis = button.click(begin_analysis, outputs=[*outputs, *controls], api_name=False,
                                queue=False, show_progress="hidden").success(
            classify, inputs=inp, outputs=outputs, api_name="classify", concurrency_limit=1, show_progress="hidden")
        analysis.success(finish_analysis, outputs=controls, api_name=False, queue=False, show_progress="hidden")
        analysis.failure(fail_analysis, outputs=[*outputs, *controls], api_name=False, queue=False, show_progress="hidden")
        clear_outputs = [inp, preview, *outputs, hint, button, clear]
        clear.click(clear_upload_ui, outputs=clear_outputs, api_name=False, queue=False, show_progress="hidden")
        inp.clear(clear_upload_ui, outputs=clear_outputs, api_name=False, queue=False, show_progress="hidden")
    return demo


def launch_options():
    """Gradio 6 takes styling at launch, shared by local and hosted entrypoints."""
    import gradio as gr
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="neutral", font=["system-ui", "sans-serif"]).set(
        background_fill_primary_dark="#000000", background_fill_secondary_dark="#101010",
        block_background_fill_dark="#181818", border_color_primary_dark="#303030",
        body_text_color_dark="#f4f4f5",
        body_text_color_subdued="*neutral_600", body_text_color_subdued_dark="*neutral_300",
        button_primary_background_fill="*primary_700", button_primary_background_fill_dark="*primary_700",
        button_primary_background_fill_hover="*primary_800", button_primary_background_fill_hover_dark="*primary_800",
        button_primary_text_color="white", button_primary_text_color_dark="white",
        button_primary_text_color_hover="white", button_primary_text_color_hover_dark="white",
    )
    return {"css": APP_CSS, "theme": theme, "max_file_size": "20mb",
            "footer_links": ["api", "settings"]}


if __name__ == "__main__":
    build_demo().launch(**launch_options())
