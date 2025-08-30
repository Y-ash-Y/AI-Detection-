import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

def _gaussian_blur(img, sigma=1.0):
    # img: (C,H,W)
    k = int(2 * round(3 * sigma) + 1)
    x = torch.arange(k) - k//2
    g = torch.exp(-(x**2) / (2*sigma**2))
    g = g / g.sum()
    g = g.to(img.dtype)
    g1 = g.view(1,1,1,k)
    g2 = g.view(1,1,k,1)
    img = F.pad(img[None], (k//2,k//2,k//2,k//2), mode="reflect")
    img = F.conv2d(img, g1.expand(img.shape[1],1,1,k), groups=img.shape[1])
    img = F.conv2d(img, g2.expand(img.shape[1],1,k,1), groups=img.shape[1])
    return img[0]

def _down_up(img, factor=2, mode_down="area", mode_up="nearest"):
    C,H,W = img.shape
    h2,w2 = H//factor, W//factor
    ds = F.interpolate(img[None], size=(h2,w2), mode=mode_down, align_corners=None if mode_down!="bilinear" else False)
    us = F.interpolate(ds, size=(H,W), mode=mode_up, align_corners=None if mode_up!="bilinear" else False)
    return us[0]

def _checkerboard(H, W, strength=0.02, freq=0.5):
    # cosine grid pattern
    y = torch.arange(H).float().view(H,1)
    x = torch.arange(W).float().view(1,W)
    pat = torch.cos(2*np.pi*freq*x/W) + torch.cos(2*np.pi*freq*y/H)
    pat = pat / pat.abs().max()
    return (strength * pat).float()

def _block_quantize(img, q=0.05, block=8):
    # rough "JPEG-like" block effect: round averages within blocks
    C,H,W = img.shape
    Hc, Wc = H - (H % block), W - (W % block)
    img = img[:, :Hc, :Wc]
    img = img.view(C, Hc//block, block, Wc//block, block).permute(0,1,3,2,4)  # C,HB,WB,b,b
    blk = img.mean(dim=(-1,-2), keepdim=True)
    blk = (blk / q).round() * q
    img_q = blk.expand_as(img).permute(0,1,3,2,4).reshape(C, Hc, Wc)
    return F.pad(img_q, (0, W-Wc, 0, H-Hc), mode="reflect")

class StructuredAVDataset(Dataset):
    """
    Synthetic 'real' vs 'fake' frames with realistic artifacts.

    Real: smooth natural-ish textures (blurred Gaussian field).
    Fake: apply one or more of: down/up sampling (alias), block quantization, checkerboard pattern.

    Returns (audio_feat, frame, label)
    """
    def __init__(self, n=6000, feat_dim=128, frame_dim=(3,64,64), seed=42,
                 p_downup=0.7, p_block=0.7, p_checker=0.5):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n = n
        self.C, self.H, self.W = frame_dim
        self.feat_dim = feat_dim
        # labels half/half
        labels = np.concatenate([np.zeros(n//2, np.int64), np.ones(n - n//2, np.int64)])
        rng.shuffle(labels)
        self.labels = labels
        # store Bernoulli probabilities for fake transforms
        self.p_downup = p_downup
        self.p_block = p_block
        self.p_checker = p_checker

    def __len__(self): return self.n

    def _make_real(self):
        # smooth Gaussian texture in [-0.5, 0.5]
        x = torch.randn(self.C, self.H, self.W) * 0.5
        x = _gaussian_blur(x, sigma=1.0)
        x = torch.tanh(x * 0.8) * 0.5
        return x

    def _make_fake_from(self, base: torch.Tensor):
        x = base.clone()
        # random subset of artifacts
        if self.rng.random() < self.p_downup:
            x = _down_up(x, factor=2, mode_down="area", mode_up="nearest")
        if self.rng.random() < self.p_block:
            x = _block_quantize(x, q=0.08, block=8)
        if self.rng.random() < self.p_checker:
            patt = _checkerboard(self.H, self.W, strength=0.03, freq=self.rng.choice([0.5,1.0,1.5]))
            x = x + patt.unsqueeze(0)  # add to each channel
        # light clipping
        x = x.clamp(-1.0, 1.0)
        return x

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        audio_feat = torch.from_numpy(self.rng.normal(size=(self.feat_dim,)).astype("float32"))
        base = self._make_real()
        frame = base if label == 0 else self._make_fake_from(base)
        return (audio_feat, frame, torch.tensor(label, dtype=torch.long))
