"""Only current production streams ship; retired requests fail without downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from pmsa.config import Config


ROOT = Path(__file__).resolve().parents[1]


def enabled_names(cfg):
    return [bb.name for bb in cfg.feature.backbones if bb.enabled]


def test_yaml_and_python_defaults_use_current_streams():
    assert enabled_names(Config()) == ["siglip", "forensic"]
    assert Config.load(ROOT / "configs/default.yaml").feature == Config().feature


def test_default_configuration_does_not_retain_disabled_streams():
    assert [bb.name for bb in Config().feature.backbones] == ["siglip", "forensic"]
    assert all(bb.enabled for bb in Config().feature.backbones)


@pytest.mark.parametrize("name", ["clip_l14", "dino_b14", "npr"])
def test_retired_backbones_fail_before_external_loads(name, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A retired backbone attempted an external model load")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=forbidden))
    api = SimpleNamespace(from_pretrained=forbidden)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPModel=api, AutoModel=api, SiglipVisionModel=api,
        AutoImageProcessor=api, CLIPImageProcessor=api))
    from pmsa.backbones import build_backbone
    with pytest.raises(KeyError, match="retired"):
        build_backbone(name)


def test_retired_modules_are_not_distributed():
    for name in ("pmsa.backbones.clip_encoder", "pmsa.backbones.dino_encoder",
                 "pmsa.backbones.npr", "pmsa.pretrained"):
        assert importlib.util.find_spec(name) is None
    assert not (ROOT / "configs/legacy_v2.yaml").exists()


@pytest.mark.parametrize("name", ["clip_l14", "dino_b14", "npr"])
def test_retired_bundle_metadata_is_rejected(name):
    from pmsa.artifacts import validate_stream_metadata
    with pytest.raises(ValueError, match="Retired backbone"):
        validate_stream_metadata({name: {}})
