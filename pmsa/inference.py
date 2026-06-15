"""End-to-end detector for a single image: provenance + forensic fusion + calibration.

This is the engine behind the demo app. It combines:
  Layer 1  provenance/watermark check (pmsa.provenance)
  Layer 2  forensic fusion score (CLIP-L/14 + DINOv2 + NPR, frozen)
  Layer 3  calibrated verdict (NP threshold) + per-stream explanation

Load once, call .predict(image) per image.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .backbones import build_backbone
from .calibration import Calibrator
from .models import FusionDetector
from .provenance import check_provenance, Provenance


@dataclass
class Verdict:
    label: str                 # "AI-generated" | "Real" | "Uncertain"
    ai_probability: float      # model P(fake), 0..1
    calibrated_fake: bool      # score > tau at the calibrated FPR
    score: float               # raw fusion log-likelihood-ratio surrogate
    tau: float
    per_stream: dict[str, float] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    explanation: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class Detector:
    """Loads frozen backbones + a trained fusion detector + a calibrator."""

    def __init__(self, checkpoint: str | Path, calibrator: str | Path,
                 device: str = "cpu", backbones=("clip_l14", "dino_b14", "npr"),
                 uncertain_band: float = 0.10):
        self.device = device
        self.fusion = FusionDetector.load(checkpoint, device=device)
        self.calibrator = Calibrator.load(calibrator)
        self.uncertain_band = uncertain_band
        # backbone order MUST match the order the fusion streams were trained in
        self.backbones = [build_backbone(n, device=device) for n in backbones]
        self._stream_names = [s.name for s in self.fusion.specs]

    def _features(self, pil_image) -> list[np.ndarray]:
        return [bb.encode([pil_image]) for bb in self.backbones]  # each [1, dim]

    def predict(self, pil_image, image_path: str | Path | None = None) -> Verdict:
        prov: Provenance = (check_provenance(image_path) if image_path
                            else Provenance(found=False, note="No file path for provenance scan."))

        streams = self._features(pil_image)
        score = float(self.fusion.score(streams)[0])
        decomp = self.fusion.decompose(streams)
        per_stream = {k: float(v[0]) for k, v in decomp.items()}

        ai_prob = _sigmoid(score)
        calibrated_fake = score > self.calibrator.tau

        # provenance overrides: a positive provenance hit is decisive
        if prov.found:
            label = "AI-generated"
            explanation = f"Provenance metadata present ({', '.join(prov.signals)})."
        elif abs(ai_prob - 0.5) < self.uncertain_band:
            label = "Uncertain"
            explanation = self._stream_story(per_stream, hedge=True)
        elif calibrated_fake:
            label = "AI-generated"
            explanation = self._stream_story(per_stream, hedge=False)
        else:
            label = "Real"
            explanation = "No provenance markers and forensic streams below the AI threshold."

        return Verdict(
            label=label, ai_probability=round(ai_prob, 4),
            calibrated_fake=calibrated_fake, score=round(score, 4),
            tau=round(self.calibrator.tau, 4), per_stream=per_stream,
            provenance=prov.as_dict(), explanation=explanation,
        )

    def _stream_story(self, per_stream: dict[str, float], hedge: bool) -> str:
        ranked = sorted(per_stream.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{k} ({v:+.2f})" for k, v in ranked[:2])
        lead = "Leaning AI; " if hedge else "Flagged AI; "
        return lead + f"strongest streams: {top}."
