import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random


# -----------------------------------------------------------
# JPEG Compression (approximation using PIL via torchvision)
# -----------------------------------------------------------
def jpeg_compress(frames: torch.Tensor, quality: int = 30) -> torch.Tensor:
    """
    frames: (B, C, H, W) in [-1, 1]
    """
    from PIL import Image
    import io

    frames = (frames + 1.0) * 0.5  # to [0,1]
    frames = frames.clamp(0, 1)

    out = []
    for img in frames:
        pil = TF.to_pil_image(img.cpu())
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        tensor = TF.to_tensor(compressed)
        out.append(tensor)

    out = torch.stack(out).to(frames.device)
    out = out * 2.0 - 1.0  # back to [-1,1]
    return out


# -----------------------------------------------------------
# Gaussian Noise
# -----------------------------------------------------------
def add_gaussian_noise(frames: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    frames: (B, C, H, W) in [-1, 1]
    """
    noise = torch.randn_like(frames) * sigma
    noisy = frames + noise
    return noisy.clamp(-1.0, 1.0)


# -----------------------------------------------------------
# Random Crop + Resize Back
# -----------------------------------------------------------
def random_crop(frames: torch.Tensor, crop_frac: float = 0.85) -> torch.Tensor:
    """
    frames: (B, C, H, W)
    crop_frac: fraction of area retained
    """
    B, C, H, W = frames.shape
    new_h = int(H * crop_frac)
    new_w = int(W * crop_frac)

    top = random.randint(0, H - new_h)
    left = random.randint(0, W - new_w)

    cropped = frames[:, :, top : top + new_h, left : left + new_w]
    resized = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)

    return resized
