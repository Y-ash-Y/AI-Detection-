"""PMSA v2 fusion detector.

Per-stream heads (CLIP semantic, DINOv2 structural, NPR artifact) each emit a
scalar logit; a learned combiner produces the final score. Keeping per-stream
logits gives the score DECOMPOSITION carried over from v1 — the one genuinely
good v1 idea — so every decision is attributable to streams ("flagged on artifact
+ structure, semantic abstained").

Final score = NP log-likelihood-ratio surrogate; higher = more fake.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StreamSpec:
    name: str
    dim: int
    hidden: int = 256


def _make_module(specs: list[StreamSpec]):
    import torch.nn as nn

    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.specs = specs
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(s.dim, s.hidden), nn.GELU(),
                    nn.Dropout(0.3), nn.Linear(s.hidden, 1),
                ) for s in specs
            ])
            # learned, non-negative-ish combiner over per-stream logits
            self.combiner = nn.Linear(len(specs), 1)

        def stream_logits(self, xs):
            return [head(x).squeeze(-1) for head, x in zip(self.heads, xs)]

        def forward(self, xs):
            sl = self.stream_logits(xs)
            import torch
            stacked = torch.stack(sl, dim=1)  # [B, n_streams]
            return self.combiner(stacked).squeeze(-1), stacked

    return FusionNet()


class FusionDetector:
    def __init__(self, specs: list[StreamSpec], device: str = "cpu", seed: int = 0):
        from ..utils.seed import set_seed

        set_seed(seed)
        self.specs = specs
        self.device = device
        self.net = _make_module(specs).to(device)
        self._mean: list[np.ndarray] = []
        self._std: list[np.ndarray] = []

    # ---- standardization per stream ------------------------------------
    def _fit_norm(self, streams: list[np.ndarray]):
        self._mean = [s.mean(0) for s in streams]
        self._std = [s.std(0) + 1e-6 for s in streams]

    def _norm(self, streams: list[np.ndarray]):
        import torch

        return [torch.tensor((s - m) / sd, dtype=torch.float32, device=self.device)
                for s, m, sd in zip(streams, self._mean, self._std)]

    # ---- train ----------------------------------------------------------
    def fit(self, streams: list[np.ndarray], y: np.ndarray,
            val_streams=None, val_y=None, epochs=50, lr=1e-3,
            weight_decay=1e-4, batch_size=256, verbose=True) -> "FusionDetector":
        import torch

        self._fit_norm(streams)
        xs = self._norm(streams)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr,
                                weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        n = len(y)
        best_state, best_val = None, np.inf
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch = [x[idx] for x in xs]
                logit, _ = self.net(batch)
                loss = loss_fn(logit, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if val_streams is not None:
                vloss = self._val_loss(val_streams, val_y, loss_fn)
                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: v.detach().clone()
                                  for k, v in self.net.state_dict().items()}
                if verbose and ep % 10 == 0:
                    print(f"epoch {ep}: val_loss={vloss:.4f}", flush=True)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _val_loss(self, streams, y, loss_fn):
        import torch

        self.net.eval()
        with torch.no_grad():
            xs = self._norm(streams)
            logit, _ = self.net(xs)
            return float(loss_fn(logit, torch.tensor(
                y, dtype=torch.float32, device=self.device)))

    # ---- inference ------------------------------------------------------
    def score(self, streams: list[np.ndarray]) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            logit, _ = self.net(self._norm(streams))
        return logit.cpu().numpy().ravel()

    def decompose(self, streams: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Per-stream logit contributions for explainability."""
        import torch

        self.net.eval()
        with torch.no_grad():
            _, stacked = self.net(self._norm(streams))
        sl = stacked.cpu().numpy()
        return {spec.name: sl[:, i] for i, spec in enumerate(self.specs)}

    # ---- io -------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "specs": [(s.name, s.dim, s.hidden) for s in self.specs],
            "state": self.net.state_dict(),
            "mean": self._mean, "std": self._std,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "FusionDetector":
        import torch

        blob = torch.load(path, map_location=device, weights_only=False)
        specs = [StreamSpec(n, d, h) for n, d, h in blob["specs"]]
        obj = cls(specs, device=device)
        obj.net.load_state_dict(blob["state"])
        obj._mean, obj._std = blob["mean"], blob["std"]
        return obj
