"""End-to-end detector for a single image: provenance + forensic model + calibration.

Engine behind the demo app. Two forensic backends, auto-detected by file extension:
  - CLIP-only linear probe (.pkl)  -> robust default (UnivFD-style). Most resilient
    to real-world photos (resolution / JPEG), so far fewer false alarms.
  - Fusion (.pt)                   -> CLIP + DINOv2 + NPR. Stronger in-distribution,
    but the NPR stream is fragile to resolution/compression and can false-alarm on
    out-of-distribution real photos.

Layered: provenance/watermark (Layer 1) -> forensic score (Layer 2) -> calibrated
verdict + explanation (Layer 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .backbones import build_backbone
from .calibration import Calibrator
from .models import FusionDetector, LinearProbe
from .provenance import check_provenance, Provenance


@dataclass
class Verdict:
    label: str                 # "AI-generated" | "Real" | "Uncertain"
    ai_probability: float
    calibrated_fake: bool
    score: float
    tau: float
    per_stream: dict[str, float] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    explanation: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class Detector:
    def __init__(self, checkpoint: str | Path, calibrator: str | Path,
                 device: str = "cpu", uncertain_band: float = 0.10):
        self.device = device
        self.calibrator = Calibrator.load(calibrator)
        self.uncertain_band = uncertain_band
        ckpt = str(checkpoint)
        if ckpt.endswith(".pkl"):                       # single-backbone linear probe
            self.mode = "probe"
            self.probe = LinearProbe.load(checkpoint)
            bbfile = Path(checkpoint).with_name("probe_backbone.txt")
            bb = bbfile.read_text().strip() if bbfile.exists() else "clip_l14"
            self.backbones = [build_backbone(bb, device=device)]
            self._stream_names = [bb]
        else:                                           # fusion
            self.mode = "fusion"
            self.fusion = FusionDetector.load(checkpoint, device=device)
            self._stream_names = [s.name for s in self.fusion.specs]
            self.backbones = [build_backbone(n, device=device)
                              for n in self._stream_names]

    def _features(self, pil_image) -> list[np.ndarray]:
        return [bb.encode([pil_image]) for bb in self.backbones]

    def _score_and_streams(self, streams):
        if self.mode == "probe":
            s = float(self.probe.score(streams[0])[0])
            return s, {self._stream_names[0]: s}
        s = float(self.fusion.score(streams)[0])
        decomp = self.fusion.decompose(streams)
        return s, {k: float(v[0]) for k, v in decomp.items()}

    def predict(self, pil_image, image_path: str | Path | None = None) -> Verdict:
        prov: Provenance = (check_provenance(image_path) if image_path
                            else Provenance(found=False, note="No file path for provenance scan."))
        streams = self._features(pil_image)
        score, per_stream = self._score_and_streams(streams)
        ai_prob = _sigmoid(score)
        calibrated_fake = score > self.calibrator.tau

        if prov.found:
            label = "AI-generated"
            explanation = f"Provenance metadata present ({', '.join(prov.signals)})."
        elif abs(ai_prob - 0.5) < self.uncertain_band:
            label = "Uncertain"
            explanation = self._story(per_stream, hedge=True)
        elif calibrated_fake:
            label = "AI-generated"
            explanation = self._story(per_stream, hedge=False)
        else:
            label = "Real"
            explanation = "No provenance markers and forensic score below the AI threshold."

        return Verdict(
            label=label, ai_probability=round(ai_prob, 4),
            calibrated_fake=calibrated_fake, score=round(score, 4),
            tau=round(self.calibrator.tau, 4), per_stream=per_stream,
            provenance=prov.as_dict(), explanation=explanation,
        )

    def _story(self, per_stream: dict[str, float], hedge: bool) -> str:
        ranked = sorted(per_stream.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{k} ({v:+.2f})" for k, v in ranked[:2])
        lead = "Leaning AI; " if hedge else "Flagged AI; "
        return lead + f"strongest streams: {top}."
