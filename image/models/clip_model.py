import torch
import open_clip


class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name="ViT-B-32", device="cpu"):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self.model = self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images):
        """
        images: (B,3,H,W) in [-1,1]
        returns: (B, D)
        """
        imgs = (images + 1.0) / 2.0
        imgs = torch.nn.functional.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
        return self.model.encode_image(imgs).float()
