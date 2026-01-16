# pmsa/models/explain.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

from pmsa.models.artifact_detector import ArtifactDetector


def grad_cam(model, inputs, target_layer):
    """
    Minimal Grad-CAM implementation.
    """
    model.eval()
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    layer = dict(model.named_modules())[target_layer]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    inputs = inputs.requires_grad_(True)
    score = model(inputs).sum()
    model.zero_grad()
    score.backward()

    A = activations["value"]
    G = gradients["value"]
    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    h1.remove()
    h2.remove()

    return cam.squeeze().cpu().numpy()


def artifact_to_text(scores: Dict[str, float]) -> str:
    """
    Converts artifact scores into grounded explanations.
    """
    explanations = []

    if scores["block_dct_energy"] > 0.06:
        explanations.append(
            "Elevated mid-frequency DCT energy indicates block quantization or resampling artifacts."
        )

    if scores["residual_energy"] > 0.02:
        explanations.append(
            "High residual energy suggests unnatural fine-detail textures."
        )

    if scores["checkerboard_score"] > 0.02:
        explanations.append(
            "Checkerboard-like patterns detected, often caused by upsampling or diffusion artifacts."
        )

    if not explanations:
        return "No dominant forensic artifact detected; prediction near decision threshold."

    return " ".join(explanations)


def explain_image(
    model,
    frame: torch.Tensor,
    target_layer: str = None
) -> Tuple[np.ndarray, str]:
    """
    Produces (heatmap, explanation).
    """
    artifact_model = ArtifactDetector()
    artifact_scores = artifact_model(frame)

    heatmap = None
    if target_layer is not None:
        heatmap = grad_cam(model, frame, target_layer)

    explanation = artifact_to_text(artifact_scores)
    return heatmap, explanation
