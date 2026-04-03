import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHighPass(nn.Module):
    """
    High-pass residual using a fixed 3x3 kernel (like a Laplacian-ish filter).
    Input: (B,3,H,W) float32
    Output: (B,1,H,W) residual magnitude (grayscale)
    """
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        weight = k.view(1, 1, 3, 3)
        self.register_buffer("weight", weight)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            x = 0.2989 * r + 0.5870 * g + 0.1140 * b
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + self.eps)
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        res = F.conv2d(x, self.weight)
        return res.abs()


def _dct_1d_mat(N: int, device: torch.device):
    n = torch.arange(N, device=device).float()
    k = n.view(-1, 1)
    W = torch.cos((torch.pi / N) * (n + 0.5) * k)
    W[0, :] /= torch.sqrt(torch.tensor(2.0, device=W.device))
    return W * torch.sqrt(torch.tensor(2.0 / N, device=W.device))


def block_dct2(x: torch.Tensor, block: int = 8) -> torch.Tensor:
    B, C, H, W = x.shape
    assert C == 1, "expect 1-channel"
    device = x.device
    assert H % block == 0 and W % block == 0, "H,W must be divisible by block"
    Wm = _dct_1d_mat(block, device)
    WT = Wm.t()
    x = x.view(B, 1, H // block, block, W // block, block)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(-1, block, block)
    X = Wm @ x @ WT
    X = X.view(B, H // block, W // block, 1, block, block).permute(0, 3, 1, 4, 2, 5)
    X = X.reshape(B, 1, H, W)
    return X


class DCTBandPool(nn.Module):
    def __init__(self, block: int = 8, keep: int = 16):
        super().__init__()
        self.block = block
        self.keep = keep
        order = []
        for s in range(2 * block - 1):
            for i in range(max(0, s - (block - 1)), min(block - 1, s) + 1):
                j = s - i
                if s % 2 == 0:
                    order.append((j, i))
                else:
                    order.append((i, j))
        self.register_buffer("zigzag_idx", torch.tensor(order[:keep], dtype=torch.long))

    def forward(self, dct_map: torch.Tensor) -> torch.Tensor:
        B, _, H, W = dct_map.shape
        b = self.block
        x = dct_map.view(B, 1, H // b, b, W // b, b)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, (H // b) * (W // b), 1, b, b)
        ii = self.zigzag_idx[:, 0]
        jj = self.zigzag_idx[:, 1]
        coeffs = x[:, :, 0][:, :, ii, jj]
        feat = coeffs.mean(dim=1)
        return feat


class ArtifactDetector(nn.Module):
    def __init__(self, dct_keep: int = 16):
        super().__init__()
        self.hp = ResidualHighPass()
        self.dct_pool = DCTBandPool(block=8, keep=dct_keep)

    @staticmethod
    def _checkerboard_score(frame: torch.Tensor) -> float:
        lap = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=frame.device).view(1, 1, 3, 3)
        gray = 0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3]
        pad = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        filt = F.conv2d(pad, lap)
        return float(filt.abs().mean().item())

    def forward(self, frames: torch.Tensor):
        with torch.no_grad():
            residual = self.hp(frames)
            H, W = residual.shape[-2:]
            residual = residual[:, :, : H - (H % 8), : W - (W % 8)]

            dct_map = block_dct2(residual, block=8)
            dct_feats = self.dct_pool(dct_map)

            blockiness = float(dct_feats.abs().mean().item())
            residual_energy = float(residual.abs().mean().item())
            checkerboard = self._checkerboard_score(frames)

        return {
            "block_dct_energy": blockiness,
            "residual_energy": residual_energy,
            "checkerboard_score": checkerboard,
        }
