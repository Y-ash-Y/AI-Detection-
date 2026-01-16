import torch
import clip

class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name="ViT-L/14", device="cpu"):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
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
        return self.model.encode_image(imgs).float()
