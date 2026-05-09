import torch
import timm
from torchvision import transforms
from PIL import Image

from image.models.device import get_device


class DINOEncoder:
    """
    Extracts structural embeddings using DINOv2 ViT-S/14.
    Output dim: 384
    Trained with self-supervised DINO objective — captures
    structural/geometric features CLIP misses.
    """

    def __init__(self, device=None):
        self.device = device or get_device()

        # vit_small_patch14_dinov2 = ~22M params, fast on M4
        self.model = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=True,
            num_classes=0      # remove classifier head → raw embeddings
        )
        self.model.to(self.device).eval()
        self.dim = 384

        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as image:
            img = self.transform(image.convert("RGB"))
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(img)

        return feat.squeeze(0).cpu().float()

    def extract_batch(self, image_paths: list) -> torch.Tensor:
        processed = []
        for path in image_paths:
            with Image.open(path) as image:
                processed.append(self.transform(image.convert("RGB")))

        imgs = torch.stack(processed).to(self.device)

        with torch.no_grad():
            feats = self.model(imgs)

        return feats.cpu().float()
    
    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept a pre-loaded tensor (B, 3, H, W) in [0,1] range.
        Applies DINOv2 normalization and resize internally.
        Used by adversarial attack pipeline.
        """
        import torchvision.transforms.functional as TF
        # DINOv2 expects 518×518
        if x.shape[-1] != 518:
            x = TF.resize(x, [518, 518])
        # DINOv2 normalization (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                            device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        x = x.to(self.device)
        feat = self.model(x)
        return feat.cpu().float()
