import torch
import torch.nn as nn

class FusionDetector(nn.Module):
    """
    CLIP + DINO + Forensic features → NP-test score
    """
    def __init__(self, clip_dim, dino_dim, forensic_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim + dino_dim + forensic_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, clip_f, dino_f, forensic_f):
        x = torch.cat([clip_f, dino_f, forensic_f], dim=1)
        return self.net(x).squeeze(-1)
