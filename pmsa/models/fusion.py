"""Configurable fusion heads; current streams are SigLIP and forensic statistics.

Each stream emits a scalar logit. A signed linear combiner produces the score,
which decomposes into weighted stream contributions plus a separate bias.
Raw head logits alone are not contributions; learned combiner signs matter.

Higher scores indicate more AI evidence. This discriminative score is not a
verified likelihood ratio or a calibrated per-image probability.
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

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name
                or type(self.dim) is not int or self.dim < 1
                or type(self.hidden) is not int or self.hidden < 1):
            raise ValueError("Stream needs a name and positive integer dimensions")


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
            # Unconstrained signed combiner (weights may be negative).
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

        if not specs or len({s.name for s in specs}) != len(specs):
            raise ValueError("Fusion requires nonempty, unique feature streams")
        set_seed(seed)
        self.specs = specs
        self.device = device
        self.net = _make_module(specs).to(device)
        self._mean: list[np.ndarray] = []
        self._std: list[np.ndarray] = []

    # ---- standardization per stream ------------------------------------
    def _fit_norm(self, streams: list[np.ndarray]):
        self._validate_streams(streams)
        self._mean = [s.mean(0) for s in streams]
        self._std = [s.std(0) + 1e-6 for s in streams]

    def _norm(self, streams: list[np.ndarray]):
        import torch
        self._validate_streams(streams)
        if len(self._mean) != len(self.specs) or len(self._std) != len(self.specs):
            raise ValueError("Missing model normalization")
        return [torch.tensor((s - m) / sd, dtype=torch.float32, device=self.device)
                for s, m, sd in zip(streams, self._mean, self._std)]

    def _validate_streams(self, streams):
        if not self.specs or len(streams) != len(self.specs):
            raise ValueError("Wrong feature stream count")
        for spec, s in zip(self.specs, streams):
            if s.shape != (len(streams[0]), spec.dim) or not np.isfinite(s).all():
                raise ValueError(f"Invalid stream {spec.name}")

    # ---- train ----------------------------------------------------------
    def fit(self, streams: list[np.ndarray], y: np.ndarray,
            val_streams=None, val_y=None, epochs=50, lr=1e-3,
            weight_decay=1e-4, batch_size=256, verbose=True) -> "FusionDetector":
        import torch
        self._validate_streams(streams)
        if epochs < 1 or batch_size < 1 or len(y) != len(streams[0]) or set(np.unique(y)) != {0, 1}:
            raise ValueError("Invalid training dimensions, classes or epochs")
        self._fit_norm(streams)
        xs = self._norm(streams)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        if (val_streams is None) != (val_y is None):
            raise ValueError("Validation features and labels must be supplied together")
        vx, vy = None, None
        if val_streams is not None:
            self._validate_streams(val_streams)
            if len(val_y) != len(val_streams[0]) or not len(val_y) or not np.isin(val_y, [0, 1]).all():
                raise ValueError("Invalid validation dimensions or labels")
            # Validation features never change; normalize/copy once, not per epoch.
            vx = self._norm(val_streams)
            vy = torch.tensor(val_y, dtype=torch.float32, device=self.device)
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
                self.net.eval()
                with torch.no_grad():
                    logits, _ = self.net(vx)
                    vloss = float(loss_fn(logits, vy))
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

    def fit_replay(self, broad_streams, broad_y, modern_streams, modern_y,
                   broad_validation, broad_val_y, modern_validation, modern_val_y,
                   *, modern_weight=.25, epochs=15, lr=1e-4,
                   weight_decay=1e-4, batch_size=256, seed=0, verbose=True):
        """Adapt an already fitted head with explicit, disjoint role inputs.

        Keep the original normalization. Each broad minibatch is paired with a
        replacement sample of modern TRAIN rows; optimize .75 broad + .25 modern
        mean BCE by default. Repeated modern draws are not independent new data.
        Select checkpoints by the same weighted validation loss, including the
        unchanged starting head. Calibration/test data are never accepted here.
        The caller must audit image-level cross-role overlap before fitting.
        """
        import torch
        from ..utils.seed import set_seed

        if (not np.isfinite(modern_weight) or not 0 < modern_weight < 1
                or epochs < 1 or batch_size < 2 or not np.isfinite(lr) or lr <= 0
                or not np.isfinite(weight_decay) or weight_decay < 0):
            raise ValueError("Invalid replay settings")
        tensors = []
        for streams, labels in ((broad_streams, broad_y), (modern_streams, modern_y),
                                (broad_validation, broad_val_y), (modern_validation, modern_val_y)):
            self._validate_streams(streams)
            labels = np.asarray(labels)
            if labels.shape != (len(streams[0]),) or set(np.unique(labels)) != {0, 1}:
                raise ValueError("Replay train/validation roles require both classes")
            tensors.append((self._norm(streams), torch.as_tensor(labels, dtype=torch.float32, device=self.device)))
        (bx, by), (mx, my), (bvx, bvy), (mvx, mvy) = tensors
        set_seed(seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def validation():
            self.net.eval()
            with torch.no_grad():
                bl = float(loss_fn(self.net(bvx)[0], bvy))
                ml = float(loss_fn(self.net(mvx)[0], mvy))
            return bl, ml, (1-modern_weight)*bl + modern_weight*ml

        def snapshot():
            return {k: v.detach().clone() for k, v in self.net.state_dict().items()}

        bl, ml, best = validation()
        best_state, best_epoch = snapshot(), -1
        history = [{"epoch": -1, "broad_bce": bl, "modern_bce": ml, "weighted_bce": best}]
        broad_batch = max(1, int(round(batch_size*(1-modern_weight))))
        modern_batch = max(1, batch_size-broad_batch)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(by), device=self.device)
            for i in range(0, len(by), broad_batch):
                bi = perm[i:i+broad_batch]
                mi = torch.randint(len(my), (modern_batch,), device=self.device)
                loss = ((1-modern_weight)*loss_fn(self.net([x[bi] for x in bx])[0], by[bi])
                        + modern_weight*loss_fn(self.net([x[mi] for x in mx])[0], my[mi]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            bl, ml, current = validation()
            history.append({"epoch": ep, "broad_bce": bl, "modern_bce": ml, "weighted_bce": current})
            if current < best:
                best, best_state, best_epoch = current, snapshot(), ep
            if verbose:
                print(f"replay seed={seed} epoch={ep}: broad={bl:.4f} modern={ml:.4f} weighted={current:.4f}", flush=True)
        self.net.load_state_dict(best_state)
        self.fit_history = {"selected_epoch": best_epoch, "history": history,
                            "selection": "Minimum weighted validation BCE; starting checkpoint included",
                            "normalization": "Frozen from starting model; not refitted"}
        return self

    # ---- inference ------------------------------------------------------
    def score(self, streams: list[np.ndarray], batch_size=2048) -> np.ndarray:
        import torch
        self._validate_streams(streams)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        result = []
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(streams[0]), batch_size):
                logit, _ = self.net(self._norm([s[i:i+batch_size] for s in streams]))
                result.append(logit.cpu().numpy().ravel())
        return np.concatenate(result) if result else np.empty(0, dtype=np.float32)

    def decompose(self, streams: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Raw per-stream head logits (pre-combiner)."""
        import torch

        self.net.eval()
        with torch.no_grad():
            _, stacked = self.net(self._norm(streams))
        sl = stacked.cpu().numpy()
        return {spec.name: sl[:, i] for i, spec in enumerate(self.specs)}

    def contributions(self, streams: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Each stream's SIGNED contribution to the final score (combiner_weight x
        logit). These are sign-consistent with the verdict: positive pushes toward
        FAKE, negative toward REAL, and they sum (+ bias) to the score."""
        import torch

        self.net.eval()
        with torch.no_grad():
            _, stacked = self.net(self._norm(streams))
            w = self.net.combiner.weight.detach()[0]      # [n_streams]
            contrib = (stacked * w).cpu().numpy()
        return {spec.name: contrib[:, i] for i, spec in enumerate(self.specs)}

    # ---- io -------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 2,
            "specs": [(s.name, s.dim, s.hidden) for s in self.specs],
            "state": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "mean": [torch.as_tensor(m) for m in self._mean],
            "std": [torch.as_tensor(s) for s in self._std],
        }, path)

    @classmethod
    def load(cls, path: str | Path, device="cpu", trusted_legacy=False) -> "FusionDetector":
        import torch

        # Never silently retry unrestricted pickle after a safe-load failure.
        blob = torch.load(path, map_location="cpu", weights_only=not trusted_legacy)
        if not isinstance(blob, dict) or not {"specs", "state", "mean", "std"} <= blob.keys():
            raise ValueError("Incomplete fusion checkpoint")
        if "format_version" in blob and blob["format_version"] != 2:
            raise ValueError("Unsupported fusion checkpoint format")
        if (not isinstance(blob["state"], dict) or not blob["state"]
                or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                       for v in blob["state"].values())):
            raise ValueError("Invalid or nonfinite fusion weights")
        try:
            specs = [StreamSpec(n, d, h) for n, d, h in blob["specs"]]
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid fusion stream specifications") from exc
        obj = cls(specs, device=device)
        try:
            obj.net.load_state_dict(blob["state"])
        except RuntimeError as exc:
            raise ValueError("Fusion weights do not match stream specifications") from exc
        if not isinstance(blob["mean"], (list, tuple)) or not isinstance(blob["std"], (list, tuple)):
            raise ValueError("Invalid normalization checkpoint")
        obj._mean = [np.asarray(m) for m in blob["mean"]]
        obj._std = [np.asarray(s) for s in blob["std"]]
        if len(obj._mean) != len(specs) or len(obj._std) != len(specs):
            raise ValueError("Normalization stream count mismatch")
        for spec, mean, std in zip(specs, obj._mean, obj._std):
            if (mean.shape != (spec.dim,) or std.shape != (spec.dim,)
                    or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
                raise ValueError("Invalid normalization checkpoint")
        return obj
