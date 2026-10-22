"""Optional third-party baseline, never an automatic app fallback.

Its weights belong to the credited upstream author. Softmax scores are not
probability-calibrated for our deployment data. This adapter is for explicit
comparison only; the main app serves the project's locally trained detector.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .inference import Verdict, decision_label
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
        raise ValueError("Cannot identify the AI class; inspect upstream id2label explicitly")

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

        label = decision_label(p_ai, self.threshold, self.uncertain_band)
        explanation = (f"External baseline {self.model_id}; softmax score={p_ai:.4f}, "
                       "not a calibrated AI probability. Metadata does not override this decision.")

        return Verdict(
            label=label, ai_probability=None, margin=p_ai-self.threshold,
            calibrated_fake=calibrated_fake, score=round(p_ai, 4),
            tau=self.threshold, per_stream={self._short: round(p_ai, 4)},
            provenance=prov.as_dict(), explanation=explanation, artifact_status="external-baseline",
        )


if __name__ == "__main__":  # quick CLI: python -m pmsa.pretrained <image>
    import sys
    from PIL import Image

    path = sys.argv[1]
    det = PretrainedDetector()
    v = det.predict(Image.open(path).convert("RGB"), image_path=path)
    print(v.label, f"softmax score={v.score:.4f}", "|", v.explanation)
