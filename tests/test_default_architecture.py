"""Defaults follow the current architecture without removing legacy imports."""
from pathlib import Path

from pmsa.config import Config


ROOT = Path(__file__).resolve().parents[1]


def enabled_names(cfg):
    return [bb.name for bb in cfg.feature.backbones if bb.enabled]


def test_yaml_and_python_defaults_use_current_streams():
    assert enabled_names(Config()) == ["siglip", "forensic"]
    assert Config.load(ROOT / "configs/default.yaml").feature == Config().feature


def test_legacy_streams_remain_explicit_opt_ins():
    cfg = Config.load(ROOT / "configs/legacy_v2.yaml")
    assert enabled_names(cfg) == ["clip_l14", "dino_b14", "npr"]
    from pmsa.backbones import clip_encoder, dino_encoder, npr
    assert clip_encoder.CLIPEncoder.name == "clip_l14"
    assert dino_encoder.DINOEncoder.name == "dino_b14"
    assert npr is not None
