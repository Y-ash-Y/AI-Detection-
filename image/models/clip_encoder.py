import torch
import clip
from PIL import Image

from image.models.device import get_device


class CLIPEncoder:
    """
    Extracts semantic embeddings using CLIP ViT-B/32.
    Output dim: 512
    """

    def __init__(self, device=None):
        self.device = device or get_device()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.dim = 512

    def extract(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as img:
            image = self.preprocess(img.convert("RGB"))
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(image)

        # L2 normalize — important for cosine-style LRT scoring
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().float()

    def extract_batch(self, image_paths: list) -> torch.Tensor:
        processed = []
        for path in image_paths:
            with Image.open(path) as img:
                processed.append(self.preprocess(img.convert("RGB")))

        images = torch.stack(processed).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_image(images)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float()
