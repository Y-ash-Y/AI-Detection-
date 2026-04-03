import torch
from pmsa.image.models.fusion_model import FusionDetector


def run_inference(clip_feat, dino_feat, forensic_feat, checkpoint_path="fusion_detector.pt", device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = FusionDetector(
        clip_dim=clip_feat.shape[1],
        dino_dim=dino_feat.shape[1],
        forensic_dim=forensic_feat.shape[1],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        score = model(clip_feat.to(device), dino_feat.to(device), forensic_feat.to(device)).item()
    return score, ckpt.get("tau", None)
