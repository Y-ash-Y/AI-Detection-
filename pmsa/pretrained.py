"""Pretrained forensic engine — wrap a strong off-the-shelf detector.

Our own tiny-genimage model can't represent the diversity of real-world photos, so
for the deployable app we stand on a detector trained on 100k+ diverse images (e.g.
Ateeqq/ai-vs-human-image-detector, SigLIP). Our value-add is the LAYERED SYSTEM
around it: provenance/watermark (Layer 1) + calibrated/explained verdict (Layer 3),
and few-shot adaptation when you have samples of a new generator.

Same .predict(image) -> Verdict interface as pmsa.inference.Detector, so the app is
engine-agnostic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .inference import Verdict
from .provenance import check_provenance, Provenance

_AI_KEYWORDS = ("ai", "fake", "artificial", "generat", "synth", "machine")


class PretrainedDetector:
    def __init__(self, model_id: str = "Ateeqq/ai-vs-human-image-detector",
                 device: str = "cpu", threshold: float = 0.5,
                 uncertain_band: float = 0.10):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        self.device = device
        self.threshold = threshold
        self.uncertain_band = uncertain_band
        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
        self.ai_idx = self._find_ai_index(self.model.config.id2label)
        self._short = model_id.split("/")[-1]

    @staticmethod
    def _find_ai_index(id2label: dict) -> int:
        for i, name in id2label.items():
            if any(k in str(name).lower() for k in _AI_KEYWORDS):
                return int(i)
        # fallback: if one label looks "real/human", AI is the other
        for i, name in id2label.items():
            if any(k in str(name).lower() for k in ("real", "hum", "nature", "photo")):
                return int(1 - int(i)) if len(id2label) == 2 else int(i)
        return 0

    def _p_ai(self, pil_image) -> float:
        torch = self.torch
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[self.ai_idx])

    def predict(self, pil_image, image_path: str | Path | None = None) -> Verdict:
        prov: Provenance = (check_provenance(image_path) if image_path
                            else Provenance(found=False, note="No file path for provenance scan."))
        p_ai = self._p_ai(pil_image)
        calibrated_fake = p_ai > self.threshold

        if prov.found:
            label = "AI-generated"
            explanation = f"Provenance metadata present ({', '.join(prov.signals)})."
        elif abs(p_ai - 0.5) < self.uncertain_band:
            label = "Uncertain"
            explanation = f"Model unsure (P(AI)={p_ai:.0%}); no provenance markers."
        elif calibrated_fake:
            label = "AI-generated"
            explanation = f"Pretrained detector ({self._short}) flags AI at {p_ai:.0%}."
        else:
            label = "Real"
            explanation = f"Pretrained detector ({self._short}) reads real (P(AI)={p_ai:.0%}); no provenance markers."

        return Verdict(
            label=label, ai_probability=round(p_ai, 4),
            calibrated_fake=calibrated_fake, score=round(p_ai, 4),
            tau=self.threshold, per_stream={self._short: round(p_ai, 4)},
            provenance=prov.as_dict(), explanation=explanation,
        )


if __name__ == "__main__":  # quick CLI: python -m pmsa.pretrained <image>
    import sys
    from PIL import Image

    path = sys.argv[1]
    det = PretrainedDetector()
    v = det.predict(Image.open(path).convert("RGB"), image_path=path)
    print(v.label, f"P(AI)={v.ai_probability:.2%}", "|", v.explanation)
