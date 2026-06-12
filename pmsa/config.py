"""Typed configuration. YAML-backed, dataclass-validated.

Load with `Config.load("configs/default.yaml")`. Experiment configs in
configs/experiments/ override the defaults shallowly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class BackboneConfig:
    name: str                       # registry key: "clip_l14", "dino_b14", "npr"
    weights: str = ""               # hf id or checkpoint; "" = library default
    image_size: int = 224
    enabled: bool = True


@dataclass
class FeatureConfig:
    cache_dir: str = "feature_cache"
    backbones: list[BackboneConfig] = field(default_factory=lambda: [
        BackboneConfig("clip_l14", "openai/clip-vit-large-patch14", 224),
        BackboneConfig("dino_b14", "facebook/dinov2-base", 224),
        BackboneConfig("npr", "", 256),  # native-res pixel-relation stream
    ])
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class CalibrationConfig:
    alpha: float = 0.01             # target FPR (false-alarm rate on real)
    method: str = "conformal"       # "conformal" (finite-sample) | "empirical"
    cal_fraction: float = 0.5       # of the real-only calibration pool
    seed: int = 0


@dataclass
class TrainConfig:
    model: str = "fusion"           # "linear_probe" (UnivFD) | "fusion" (PMSA v2)
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    val_fraction: float = 0.15


@dataclass
class EvalConfig:
    bootstrap_n: int = 2000
    ci: float = 0.95
    fpr_targets: list[float] = field(default_factory=lambda: [0.01, 0.05])


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "outputs"
    data_root: str = "data"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- io -------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        for k, v in raw.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key: {k}")
            cur = getattr(cfg, k)
            if k == "feature" and isinstance(v, dict):
                bbs = v.pop("backbones", None)
                fc = FeatureConfig(**{**asdict(cur), **v, "backbones": cur.backbones})
                if bbs is not None:
                    fc.backbones = [BackboneConfig(**b) for b in bbs]
                setattr(cfg, k, fc)
            elif is_dataclass_field(cur) and isinstance(v, dict):
                setattr(cfg, k, type(cur)(**{**asdict(cur), **v}))
            else:
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_dataclass_field(obj: Any) -> bool:
    from dataclasses import is_dataclass

    return is_dataclass(obj) and not isinstance(obj, type)
