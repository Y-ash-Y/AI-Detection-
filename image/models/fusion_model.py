import torch
import torch.nn as nn


class FusionDetector(nn.Module):
    """
    Learned surrogate for the log-likelihood ratio:

        s(x) ≈ log p(x | fake) - log p(x | real)

    Architecture: MLP with residual skip + batch norm.
    Output: scalar score (NOT sigmoid — raw logit)

    Decision rule (Neyman-Pearson):
        s(x) > τ  →  FAKE
        s(x) ≤ τ  →  REAL

    τ is calibrated post-training to fix FPR ≤ α on a held-out
    set of real images.

    Input dims:
        clip_dim     : 512   (CLIP ViT-B/32)
        dino_dim     : 384   (DINOv2 ViT-S/14)
        forensic_dim : 1024  (DCT + noise residual)
        total        : 1920
    """

    def __init__(self, clip_dim: int, dino_dim: int, forensic_dim: int,
                 dropout: float = 0.3):
        super().__init__()

        input_dim = clip_dim + dino_dim + forensic_dim

        # Per-modality projection — lets each stream learn its own scale
        self.clip_proj     = nn.Linear(clip_dim, 256)
        self.dino_proj     = nn.Linear(dino_dim, 256)
        self.forensic_proj = nn.Linear(forensic_dim, 256)

        fused_dim = 256 * 3  # 768

        self.net = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),

            nn.Linear(256, 64),
            nn.GELU(),

            nn.Linear(64, 1)   # raw logit = LRT score
        )

        # Residual path: input_dim → 1 (skip connection)
        self.skip = nn.Linear(input_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                clip_feat: torch.Tensor,
                dino_feat: torch.Tensor,
                forensic_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_feat     : (B, 512)
            dino_feat     : (B, 384)
            forensic_feat : (B, 1024)

        Returns:
            score : (B,)  — log-likelihood ratio approximation
        """
        c = torch.relu(self.clip_proj(clip_feat))
        d = torch.relu(self.dino_proj(dino_feat))
        f = torch.relu(self.forensic_proj(forensic_feat))

        fused = torch.cat([c, d, f], dim=-1)

        raw_input = torch.cat([clip_feat, dino_feat, forensic_feat], dim=-1)

        score = self.net(fused).squeeze(-1) + self.skip(raw_input).squeeze(-1)
        return score
