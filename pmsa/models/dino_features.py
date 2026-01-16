import torch

class DINOFeatureExtractor(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", pretrained=True
        ).to(device)
        self.model.eval()
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
        feats = self.model(imgs)
        return feats.float()
