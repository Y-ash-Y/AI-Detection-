import torch
import torch.nn.functional as F

def jpeg_compress(x: torch.Tensor, quality: int = 30) -> torch.Tensor:
    """
    Approx JPEG artifact: down/up quantization in DCT domain.
    Simpler proxy since we avoid full jpeg library.
    x: (B,C,H,W) float [-1,1]
    """
    # rescale to 0-255
    x = ((x + 1) * 127.5).clamp(0,255).byte()
    # crude approximation: downsample + upsample
    scale = max(1, 100 // quality)
    h, w = x.shape[-2:]
    h2, w2 = h//scale, w//scale
    y = F.interpolate(x.float(), size=(h2,w2), mode="area")
    y = F.interpolate(y, size=(h,w), mode="nearest")
    return (y/127.5 - 1.0).clamp(-1,1)

def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return (x + sigma*torch.randn_like(x)).clamp(-1,1)

def random_crop(x: torch.Tensor, crop_frac: float = 0.9) -> torch.Tensor:
    """
    Center crop and resize back to original.
    crop_frac=0.9 keeps 90% of size.
    """
    B,C,H,W = x.shape
    ch, cw = int(H*crop_frac), int(W*crop_frac)
    top = (H - ch)//2
    left = (W - cw)//2
    cropped = x[:,:,top:top+ch,left:left+cw]
    return F.interpolate(cropped, size=(H,W), mode="bilinear", align_corners=False)
