"""Deployment staging never copies private training identities or arbitrary files."""
import importlib.util
import json
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def stager():
    spec=importlib.util.spec_from_file_location("prepare_space",ROOT/"scripts/19_prepare_space.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    pytest.importorskip("torch")
    from pmsa.artifacts import save_bundle
    from pmsa.calibration import Calibrator
    from pmsa.models import FusionDetector,StreamSpec
    from pmsa.preprocessing import ImagePolicy
    stage=stager()
    repo=tmp_path/"repo"
    for name in stage.SERVING_FILES:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("# serving fixture\n")
    for name in stage.TEMPLATES:
        path=repo/"deployment/space"/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("fixture\n")
    (repo/"LICENSE").write_text("license fixture")
    (repo/"private.pdf").write_bytes(b"private")
    source=repo/"outputs/source"
    source.mkdir(parents=True)
    model=FusionDetector([StreamSpec("siglip",768,4),StreamSpec("forensic",66,4)])
    rng=np.random.default_rng(2)
    model._fit_norm([rng.normal(size=(20,d)).astype(np.float32) for d in (768,66)])
    model.save(source/"fusion.pt")
    Calibrator(1.,.05,"empirical",20,"synthetic",False).save(source/"calibrator.json")
    (source/"splits.json").write_text(json.dumps({"seed":0,"development_ids":{"paths":["PRIVATE-IDENTITY"]},
        "partitions":{"train":{"n_real":10,"n_fake":10,"groups":["PRIVATE-GROUP"],"original_indices":[1,2]}}}))
    metadata={"schema_version":1,"image_policy":ImagePolicy().to_dict(),"packages":{"numpy":version("numpy")},"backbone_kwargs":{}}
    semantic=dict(metadata,backbone_kwargs={"weights":"fixture","revision":"fixed","use_fast":False})
    save_bundle(source,"fusion.pt","fusion",{"siglip":semantic,"forensic":metadata})
    return stage,repo,source


def test_staging_preserves_model_and_sanitizes_split_identities(tmp_path):
    from pmsa.artifacts import sha256_file,verify_bundle
    stage,repo,source=fixture(tmp_path)
    before={p.name:sha256_file(p) for p in source.iterdir()}
    out=repo/"outputs/release/payload"
    receipt=stage.prepare(repo,source,out)
    for name,expected in before.items():
        assert sha256_file(source/name)==expected
    assert sha256_file(out/"model/fusion.pt")==before["fusion.pt"]
    assert sha256_file(out/"model/calibrator.json")==before["calibrator.json"]
    assert sha256_file(out/"model/splits.json")!=before["splits.json"]
    split=json.loads((out/"model/splits.json").read_text())
    assert split["partitions"]=={"train":{"n_real":10,"n_fake":10}}
    for text in (out/"model/splits.json",out/"model/bundle.json",out/"release_manifest.json"):
        assert "PRIVATE-IDENTITY" not in text.read_text() and "PRIVATE-GROUP" not in text.read_text()
        assert str(tmp_path) not in text.read_text()
    verify_bundle(out/"model")
    assert not (out/"private.pdf").exists()
    assert (out/"LICENSE").read_text()=="license fixture"
    assert set(receipt["files"])=={str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}-{"release_manifest.json"}
    with pytest.raises(FileExistsError):stage.prepare(repo,source,out)


def test_staging_requires_private_output_directory(tmp_path):
    stage,repo,source=fixture(tmp_path)
    with pytest.raises(ValueError,match="ignored outputs"):
        stage.prepare(repo,source,repo/"public-release")


def test_space_card_configuration():
    import yaml
    text=(ROOT/"deployment/space/README.md").read_text()
    metadata=yaml.safe_load(text.split("---",2)[1])
    assert metadata["sdk"]=="gradio"
    assert metadata["sdk_version"]=="6.18.0"
    assert metadata["python_version"]=="3.12"
    assert len(metadata["short_description"])<=60
    assert (ROOT/"deployment/space"/metadata["app_file"]).is_file()
    requirements=(ROOT/"deployment/space/requirements.txt").read_text()
    assert "gradio[oauth,mcp]==6.18.0" in requirements
    assert "pydantic==2.12.5" in requirements  # HF SDK auto-adds the MCP extra.
