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
        k = torch.tensor([[0., -1., 0.],
                          [-1., 4., -1.],
                          [0., -1., 0.]], dtype=torch.float32)
        weight = k.view(1, 1, 3, 3)
        self.register_buffer("weight", weight)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # convert to gray
        if x.shape[1] == 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            x = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # normalize per-sample
        x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + self.eps)
        # pad and high-pass
        x = F.pad(x, (1,1,1,1), mode="reflect")
        res = F.conv2d(x, self.weight)
        return res.abs()  # magnitude

def _dct_1d_mat(N: int, device: torch.device):
    # Orthonormal DCT-II matrix
    n = torch.arange(N, device=device).float()
    k = n.view(-1, 1)
    W = torch.cos((torch.pi / N) * (n + 0.5) * k)
    W[0, :] /= torch.sqrt(torch.tensor(2.0, device= W.device))
    return W * torch.sqrt(torch.tensor(2.0 / N, device = W.device))

def block_dct2(x: torch.Tensor, block: int = 8) -> torch.Tensor:
    """
    2D DCT applied per non-overlapping block.
    x: (B,1,H,W) -> (B,1,H,W) DCT coeffs
    """
    B, C, H, W = x.shape
    assert C == 1, "expect 1-channel"
    device = x.device
    assert H % block == 0 and W % block == 0, "H,W must be divisible by block"
    Wm = _dct_1d_mat(block, device)          # (b,b)
    WT = Wm.t()
    x = x.view(B, 1, H // block, block, W // block, block)  # (B,1,HB,b,WB,b)
    x = x.permute(0,2,4,1,3,5)                              # (B,HB,WB,1,b,b)
    x = x.reshape(-1, block, block)                         # (B*HB*WB, b, b)
    X = Wm @ x @ WT                                         # 2D DCT
    X = X.view(B, H // block, W // block, 1, block, block).permute(0,3,1,4,2,5)
    X = X.reshape(B, 1, H, W)
    return X

class DCTBandPool(nn.Module):
    """
    Keep only a small set of low-mid frequency bands from block-DCT.
    Returns a flattened vector.
    """
    def __init__(self, block: int = 8, keep: int = 16):
        super().__init__()
        self.block = block
        self.keep = keep  # number of lowest zig-zag coefficients to keep per block

        # Precompute zig-zag order for an 8x8
        order = []
        for s in range(2*block-1):
            for i in range(max(0, s - (block-1)), min(block-1, s)+1):
                j = s - i
                if s % 2 == 0:
                    order.append((j, i))
                else:
                    order.append((i, j))
        self.register_buffer("zigzag_idx", torch.tensor(order[:keep], dtype=torch.long))

    def forward(self, dct_map: torch.Tensor) -> torch.Tensor:
        # dct_map: (B,1,H,W) where H,W divisible by block
        B, _, H, W = dct_map.shape
        b = self.block
        x = dct_map.view(B, 1, H//b, b, W//b, b)      # (B,1,HB,b,WB,b)
        x = x.permute(0,2,4,1,3,5).contiguous()       # (B,HB,WB,1,b,b)
        x = x.view(B, (H//b)*(W//b), 1, b, b)         # (B,Nblk,1,b,b)
        # Gather zig-zag coefficients
        ii = self.zigzag_idx[:, 0]
        jj = self.zigzag_idx[:, 1]
        coeffs = x[:, :, 0][:, :, ii, jj]             # (B,Nblk,keep)
        # Mean-pool across blocks to get a compact representation
        feat = coeffs.mean(dim=1)                     # (B,keep)
        return feat
