"""Gradio demo for the PMSA real-vs-AI image detector.

Run locally:   python app.py        (after scripts/train_deploy.py)
Deploy:        push to a Hugging Face Space (see README "Demo / deployment").

Layered detector: provenance metadata (Layer 1) -> forensic fusion (Layer 2)
-> calibrated verdict + per-stream explanation (Layer 3).
"""
from __future__ import annotations

import os
import tempfile
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

import gradio as gr

from pmsa.utils import get_device

# Default engine: a strong pretrained detector (trained on 100k+ diverse images).
# Set PMSA_HF_MODEL=local to use your own trained model in outputs/deploy/ instead.
HF_MODEL = os.environ.get("PMSA_HF_MODEL", "Ateeqq/ai-vs-human-image-detector")


def _default_ckpt() -> str:
    for p in ("outputs/deploy/probe.pkl", "outputs/deploy/fusion.pt"):
        if os.path.exists(p):
            return p
    return "outputs/deploy/probe.pkl"


CKPT = os.environ.get("PMSA_CKPT", _default_ckpt())
CAL = os.environ.get("PMSA_CAL", "outputs/deploy/calibrator.json")

_detector = None


def _load():
    global _detector
    if _detector is None:
        if HF_MODEL and HF_MODEL.lower() != "local":
            from pmsa.pretrained import PretrainedDetector
            _detector = PretrainedDetector(HF_MODEL, device=get_device())
        else:
            from pmsa.inference import Detector
            _detector = Detector(CKPT, CAL, device=get_device())
    return _detector


def classify(image):
    if image is None:
        return "Upload an image.", {}, ""
    det = _load()
    # write to a temp file so the provenance scanner can read embedded metadata
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name)
        path = f.name
    try:
        v = det.predict(image, image_path=path)
    finally:
        os.unlink(path)

    label = {
        "AI-generated": "🔴 AI-generated",
        "Real": "🟢 Real",
        "Uncertain": "🟡 Uncertain",
    }.get(v.label, v.label)
    headline = f"## {label}\n**AI probability: {v.ai_probability:.0%}**\n\n{v.explanation}"

    confidences = {"AI-generated": v.ai_probability, "Real": 1 - v.ai_probability}
    detail = (
        f"**Provenance (Layer 1):** {v.provenance.get('note','')}"
        + (f"\n- signals: {', '.join(v.provenance['signals'])}" if v.provenance.get("signals") else "")
        + f"\n\n**Forensic streams (Layer 2):** score={v.score} (threshold τ={v.tau})\n"
        + "\n".join(f"- {k}: {val:+.3f}" for k, val in v.per_stream.items())
        + "\n\n*Honest note: strongest on common generators; provenance metadata is the "
        "most reliable signal for frontier models (OpenAI/Gemini), but can be stripped.*"
    )
    return headline, confidences, detail


with gr.Blocks(title="PMSA — AI image detector") as demo:
    gr.Markdown(
        "# PMSA — real vs AI-generated image detector\n"
        "Layered: provenance/watermark check → frozen CLIP+DINOv2+NPR forensic "
        "fusion → calibrated verdict. Upload an image to test."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Image")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_label = gr.Markdown()
            out_conf = gr.Label(label="Confidence", num_top_classes=2)
            out_detail = gr.Markdown()
    btn.click(classify, inputs=inp, outputs=[out_label, out_conf, out_detail])
    inp.upload(classify, inputs=inp, outputs=[out_label, out_conf, out_detail])


if __name__ == "__main__":
    demo.launch()
