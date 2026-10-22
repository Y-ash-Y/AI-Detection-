"""Offline pinning regression tests; no Hub downloads or real model loads."""
import sys
from types import SimpleNamespace

import pytest

from pmsa.backbones.hub_revision import resolve_hub_revision

COMMIT = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


def fake_hub(monkeypatch, path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    return calls


def snapshot(tmp_path):
    config = tmp_path / "models--fixture--model" / "snapshots" / COMMIT / "config.json"
    config.parent.mkdir(parents=True)
    blob = tmp_path / "blob.json"
    blob.write_text("{}")
    config.symlink_to(blob)
    return config


def test_revision_resolves_branch_and_preserves_snapshot_symlink(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, snapshot(tmp_path))
    assert resolve_hub_revision("fixture/model", "release") == COMMIT
    assert calls == [{"repo_id": "fixture/model", "filename": "config.json", "revision": "release"}]


@pytest.mark.parametrize("invalid", ["main", "1234567", "not-a-hash"])
def test_revision_rejects_unresolved_snapshot(tmp_path, monkeypatch, invalid):
    config = tmp_path / "snapshots" / invalid / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    fake_hub(monkeypatch, config)
    with pytest.raises(ValueError, match="immutable"):
        resolve_hub_revision("fixture/model")


def test_revision_rejects_commit_mismatch(tmp_path, monkeypatch):
    fake_hub(monkeypatch, snapshot(tmp_path))
    with pytest.raises(ValueError, match="requested commit"):
        resolve_hub_revision("fixture/model", "a" * 40)


@pytest.mark.parametrize("encoder", ["siglip", "clip_l14", "dino_b14"])
def test_vision_config_without_private_commit_still_pins_both_loads(tmp_path, monkeypatch, encoder):
    pytest.importorskip("torch")
    hub_calls = fake_hub(monkeypatch, snapshot(tmp_path))
    calls = []

    class Model:
        config = SimpleNamespace(hidden_size=768)  # No _commit_hash.

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    model = Model()
    model.vision_model = Model()
    model.visual_projection = Model()

    def load_model(weights, **kwargs):
        calls.append(("model", weights, kwargs))
        return model

    def load_processor(weights, **kwargs):
        calls.append(("processor", weights, kwargs))
        return SimpleNamespace()

    model_api = SimpleNamespace(from_pretrained=load_model)
    processor_api = SimpleNamespace(from_pretrained=load_processor)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        SiglipVisionModel=model_api, CLIPModel=model_api, AutoModel=model_api,
        AutoImageProcessor=processor_api, CLIPImageProcessor=processor_api))
    from pmsa.backbones import build_backbone
    bb = build_backbone(encoder, weights="fixture/model")
    assert hub_calls[0]["revision"] == "main"
    assert calls == [("model", "fixture/model", {"revision": COMMIT}),
                     ("processor", "fixture/model", {"revision": COMMIT, "use_fast": False})]
    assert bb.load_config == {"weights": "fixture/model", "revision": COMMIT, "use_fast": False}
