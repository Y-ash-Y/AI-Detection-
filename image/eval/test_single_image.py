import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from pmsa.image.models.clip_model import CLIPFeatureExtractor
from pmsa.image.models.dino_model import DINOFeatureExtractor
from pmsa.image.models.forensic_features import ResidualHighPass, block_dct2, DCTBandPool
from pmsa.image.models.fusion_model import FusionDetector
from pmsa.image.pipeline.explainability import explain_image


def load_image(path, size=224):
    transform = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),  # normalize to [-1,1]
        ]
    )
    img = Image.open(path).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0) if isinstance(img, torch.Tensor) else torch.from_numpy(img).unsqueeze(0)


def extract_features(img):
    clip_model = CLIPFeatureExtractor()
    dino_model = DINOFeatureExtractor()
    hp = ResidualHighPass()
    pool = DCTBandPool(block=8, keep=16)

    with torch.no_grad():
        clip_feat = clip_model(img)
        dino_feat = dino_model(img)

    r = hp(img)
    H, W = r.shape[-2:]
    r = r[:, :, : H - (H % 8), : W - (W % 8)]
    dct_map = block_dct2(r, block=8)
    forensic_feat = pool(dct_map)

    return clip_feat, dino_feat, forensic_feat


def main(image_path):
    print("\nLoading image...")
    img = load_image(image_path)

    print("Extracting features...")
    clip_feat, dino_feat, forensic_feat = extract_features(img)

    print("Loading fusion model...")
    ckpt = torch.load("fusion_detector.pt", map_location="cpu")
    model = FusionDetector(
        clip_dim=clip_feat.shape[1],
        dino_dim=dino_feat.shape[1],
        forensic_dim=forensic_feat.shape[1],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tau = ckpt["tau"]

    print("Running inference...")
    with torch.no_grad():
        score = model(clip_feat, dino_feat, forensic_feat).item()

    print(f"\nScore: {score:.6f}")
    print(f"Threshold τ: {tau:.6f}")

    if score > tau:
        print("Prediction: AI-GENERATED")
    else:
        print("Prediction: REAL")

    print("\nExplanation:")
    _, explanation = explain_image(model, img)
    print(explanation)


if __name__ == "__main__":
    main("pmsa/eval/test_image.jpg")
