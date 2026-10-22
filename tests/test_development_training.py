"""Explicit development roles and replay must not silently resplit or renormalize."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pmsa.artifacts import sha256_file
from pmsa.data.manifest import ImageRecord, Manifest
from pmsa.development import compatible_metadata, prepared_roles, load_modern, ROLES, NAMES
from pmsa.features.cache import FeatureSet


def metadata():
    return {"schema_version":1, "image_policy":{}, "backbone_kwargs":{"revision":"fixed"},
            "packages":{"torch":"2.11.0+cu128","torchvision":"0.26.0+cu128","Pillow":"12.3.0"}}


def test_metadata_only_tolerates_torch_build_suffix():
    a, b = metadata(), metadata()
    b["packages"].update(torch="2.11.0", torchvision="0.26.0")
    assert compatible_metadata(a,b)
    for field, value in (("backbone_kwargs",{"revision":"other"}), ("image_policy",{"changed":True}),
                         ("packages",{**b["packages"],"Pillow":"different"})):
        wrong = copy.deepcopy(b)
        wrong[field] = value
        assert not compatible_metadata(a,wrong)
    assert a == metadata()  # comparison does not mutate source metadata


def fixture_roles(tmp_path):
    roles = tmp_path/"roles"
    roles.mkdir()
    for i,role in enumerate((*ROLES,"challenge")):
        Manifest([] if role == "challenge" else [
            ImageRecord(f"modern_{i}_{j}",j,"mixed","real" if not j else "generator",f"group_{i}_{j}")
            for j in (0,1)]).to_csv(roles/f"manifest_{role}.csv")
    plan = {"protocol":"reused-generator-development-v1","untouched_confirmation_test":False,
            "manifest_sha256":{f"manifest_{r}.csv":sha256_file(roles/f"manifest_{r}.csv")
                               for r in (*ROLES,"challenge")}}
    (roles/"split_plan.json").write_text(json.dumps(plan))
    return roles


def test_roles_are_bound_to_checksums_and_no_fresh_claim(tmp_path):
    roles = fixture_roles(tmp_path)
    _, manifests = prepared_roles(roles)
    assert len(manifests["train"]) == 2
    path = roles/"manifest_test.csv"
    path.write_text(path.read_text()+"changed")
    with pytest.raises(ValueError,match="Changed role manifest"):
        prepared_roles(roles)


def test_duplicate_groups_rejected_even_if_manifest_rehashed(tmp_path):
    roles = fixture_roles(tmp_path)
    path = roles/"manifest_test.csv"
    m = Manifest.from_csv(path)
    m.records[0].group_id = "group_0_0"
    m.to_csv(path)
    plan = json.loads((roles/"split_plan.json").read_text())
    plan["manifest_sha256"][path.name] = sha256_file(path)
    (roles/"split_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError,match="Duplicate image/group"):
        prepared_roles(roles)


def fixture_caches(tmp_path):
    roles = fixture_roles(tmp_path)
    _, manifests = prepared_roles(roles)
    recs = [r for m in manifests.values() for r in m.records]
    directory = tmp_path/"features"
    directory.mkdir()
    modern = [FeatureSet(np.arange(16,dtype=np.float32).reshape(8,2),
                        np.array([r.label for r in recs]),np.array([r.path for r in recs]),
                        np.array([r.domain for r in recs]),np.array([r.source for r in recs]),name,
                        metadata(),np.array([f"modernhash{i}" for i in range(8)]),
                        np.array([r.group_id for r in recs])) for name in NAMES]
    broad = [FeatureSet(np.zeros((2,2),np.float32),np.array([0,1]),np.array(["b0","b1"]),
                       np.array(["b","b"]),np.array(["real","fake"]),name,metadata(),
                       np.array(["bhash0","bhash1"]),np.array(["bg0","bg1"])) for name in NAMES]
    for fs in modern:
        fs.save(directory/f"{fs.backbone}_modern.npz")
    receipt = {"split_plan_sha256":sha256_file(roles/"split_plan.json"),
               "role_indices":{r:[2*i,2*i+1] for i,r in enumerate(ROLES)},
               "feature_sha256":{name:sha256_file(directory/f"{name}_modern.npz") for name in NAMES}}
    (directory/"extraction.json").write_text(json.dumps(receipt))
    return roles,directory,broad,modern,receipt


def test_features_preserve_exact_role_membership(tmp_path):
    roles,directory,broad,_,_ = fixture_caches(tmp_path)
    _, partitions,_ = load_modern(directory,roles,broad)
    assert partitions["train"].tolist() == [0,1]
    assert partitions["test"].tolist() == [6,7]


@pytest.mark.parametrize("defect",["integrity","resplit","overlap"])
def test_changed_features_or_leakage_fail(tmp_path,defect):
    roles,directory,broad,modern,receipt = fixture_caches(tmp_path)
    if defect == "integrity":
        with (directory/"siglip_modern.npz").open("ab") as handle:
            handle.write(b"changed")
    elif defect == "resplit":
        receipt["role_indices"]["train"],receipt["role_indices"]["test"] = receipt["role_indices"]["test"],receipt["role_indices"]["train"]
    else:
        for fs in modern:
            fs.content_hashes[0] = "bhash0"
            path = directory/f"{fs.backbone}_modern.npz"
            fs.save(path)
            receipt["feature_sha256"][fs.backbone] = sha256_file(path)
    (directory/"extraction.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        load_modern(directory,roles,broad)


def replay_fixture():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from pmsa.models.fusion import FusionDetector, StreamSpec
    rng = np.random.default_rng(7)
    features = [rng.normal(size=(16,3)).astype(np.float32)]
    y = np.arange(16)%2
    model = FusionDetector([StreamSpec("siglip",3,4)])
    model.fit(features,y,features,y,epochs=1,batch_size=8,verbose=False)
    return model,features,y


def test_replay_preserves_normalization_and_roundtrip(tmp_path):
    from pmsa.models.fusion import FusionDetector
    model,x,y = replay_fixture()
    before = [a.copy() for a in model._mean+model._std]
    model.fit_replay(x,y,x,y,x,y,x,y,epochs=2,batch_size=8,verbose=False)
    for a,b in zip(before,model._mean+model._std):
        np.testing.assert_array_equal(a,b)
    history = model.fit_history["history"]
    selected = model.fit_history["selected_epoch"]
    assert len(history) == 3 and history[0]["epoch"] == -1
    assert next(r["weighted_bce"] for r in history if r["epoch"] == selected) == min(r["weighted_bce"] for r in history)
    model.save(tmp_path/"fusion.pt")
    np.testing.assert_allclose(model.score(x),FusionDetector.load(tmp_path/"fusion.pt").score(x),atol=1e-6)


@pytest.mark.parametrize("weight",[0,1,-.1,float("nan")])
def test_replay_invalid_weights_fail(weight):
    model,x,y = replay_fixture()
    with pytest.raises(ValueError,match="Invalid replay settings"):
        model.fit_replay(x,y,x,y,x,y,x,y,modern_weight=weight)


def test_replay_requires_both_validation_classes():
    model,x,y = replay_fixture()
    with pytest.raises(ValueError,match="both classes"):
        model.fit_replay(x,y,x,y,x,y,x,np.zeros_like(y),epochs=1)
