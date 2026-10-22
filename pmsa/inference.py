"""Trusted-artifact serving with extraction-consistent preprocessing.

A threshold calibrates a decision rule, NOT the probability an image is AI.
Metadata hints never override the forensic decision. Only load trusted artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

from .artifacts import verify_bundle, validate_stream_metadata, check_runtime
from .backbones import build_backbone
from .calibration import Calibrator
from .models import FusionDetector, LinearProbe
from .preprocessing import ImagePolicy, prepare_image
from .provenance import check_provenance, Provenance


@dataclass
class Verdict:
    label: str
    calibrated_fake: bool
    score: float
    tau: float
    margin: float
    per_stream: dict[str, float] = field(default_factory=dict)
    combiner_bias: float = 0.0
    provenance: dict = field(default_factory=dict)
    explanation: str = ""
    ai_probability: float | None = None  # No probability calibrator has been fitted.
    target_fpr: float | None = None
    real_domain: str = "unknown"
    artifact_status: str = "verified-bundle"

    def as_dict(self) -> dict:
        return asdict(self)


def decision_label(score, tau, uncertain_margin=.25):
    """Abstain near tau in SCORE units; this band is heuristic, not coverage."""
    if (not np.isfinite([score, tau, uncertain_margin]).all()
            or uncertain_margin < 0):
        raise ValueError("Finite score, threshold and nonnegative margin required")
    if abs(score - tau) <= uncertain_margin:
        return "Uncertain"
    return "AI-generated signal" if score > tau else "No AI signal detected"


class Detector:
    def __init__(self, checkpoint: str | Path, calibrator: str | Path,
                 device="cpu", uncertain_margin=.25, *, allow_legacy=False,
                 legacy_policy: ImagePolicy | None = None, trusted_legacy=False):
        self.device = device
        self.uncertain_margin = uncertain_margin
        self.bundle = None
        directory = Path(checkpoint).parent
        if (directory / "bundle.json").exists():
            self.bundle = verify_bundle(directory, checkpoint, calibrator)
            self.image_policy = validate_stream_metadata(self.bundle["streams"])
            check_runtime(self.bundle["streams"])
            self.mode = self.bundle["model_type"]
            self.artifact_status = "verified-bundle"
        else:
            if not allow_legacy or legacy_policy is None:
                raise ValueError("Legacy artifacts lack preprocessing/pairing records. "
                                 "Use a v3 bundle, or explicitly opt into legacy mode with an image policy.")
            self.image_policy = legacy_policy
            self.mode = "probe" if str(checkpoint).endswith(".pkl") else "fusion"
            self.artifact_status = "legacy-unverified"
        self.calibrator = Calibrator.load(calibrator)
        decision_label(0., self.calibrator.tau, uncertain_margin)
        if self.mode == "probe":
            if self.bundle:
                self._stream_names = list(self.bundle["streams"])
                if len(self._stream_names) != 1:
                    raise ValueError("Probe bundle must contain exactly one stream")
            else:
                bbfile = directory / "probe_backbone.txt"
                if not bbfile.exists():
                    raise ValueError("Legacy probe needs probe_backbone.txt")
                self._stream_names = [bbfile.read_text().strip()]
            self.probe = LinearProbe.load(checkpoint)  # Trusted joblib artifacts only.
        else:
            self.fusion = FusionDetector.load(checkpoint, device=device,
                                              trusted_legacy=trusted_legacy and allow_legacy)
            self._stream_names = [s.name for s in self.fusion.specs]
            if self.bundle and self._stream_names != list(self.bundle["streams"]):
                raise ValueError("Fusion stream order does not match bundle")
        self.backbones = [build_backbone(name, device=device, **(
            self.bundle["streams"][name]["backbone_kwargs"] if self.bundle else {}))
            for name in self._stream_names]
        expected_dims = ([self.probe.scaler.n_features_in_] if self.mode == "probe"
                         else [spec.dim for spec in self.fusion.specs])
        for name, backbone, expected in zip(self._stream_names, self.backbones, expected_dims):
            actual = getattr(backbone, "dim", None)
            if actual is not None and actual != expected:
                raise ValueError(f"Backbone/head dimension mismatch for {name}: "
                                 f"encoder={actual}, head={expected}")

    def _features(self, pil_image):
        image = prepare_image(pil_image, self.image_policy)
        return [bb.encode([image]) for bb in self.backbones]

    def _score_and_streams(self, streams):
        if self.mode == "probe":
            score = float(self.probe.score(streams[0])[0])
            return score, {self._stream_names[0]: score}, 0.
        score = float(self.fusion.score(streams)[0])
        contributions = {k: float(v[0]) for k, v in self.fusion.contributions(streams).items()}
        bias = float(self.fusion.net.combiner.bias.detach().cpu()[0])
        return score, contributions, bias

    def predict(self, pil_image, image_path=None):
        prov = (check_provenance(image_path) if image_path else
                Provenance(found=False, note="Original file unavailable; metadata not inspected."))
        score, per_stream, bias = self._score_and_streams(self._features(pil_image))
        label = decision_label(score, self.calibrator.tau, self.uncertain_margin)
        explanation = {
            "Uncertain": "Score is near the decision threshold. This is a heuristic abstention, not calibrated uncertainty.",
            "AI-generated signal": "Score exceeds the reference threshold; this is evidence, not proof of AI generation.",
            "No AI signal detected": "Score is below the reference threshold. This does not establish camera origin.",
        }[label]
        if self.artifact_status == "legacy-unverified":
            explanation += " Legacy artifact pairing and preprocessing have not been independently verified."
        return Verdict(label=label, calibrated_fake=bool(score > self.calibrator.tau),
                       score=score, tau=self.calibrator.tau, margin=score-self.calibrator.tau,
                       per_stream=per_stream, combiner_bias=bias, provenance=prov.as_dict(),
                       explanation=explanation, target_fpr=self.calibrator.alpha,
                       real_domain=self.calibrator.real_domain, artifact_status=self.artifact_status)
