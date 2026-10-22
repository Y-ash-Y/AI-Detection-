"""Offline raw-data and fresh-evaluation contracts; never download backbones."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from pmsa.data.genimage import scan_fake_dir, scan_real_dir
from pmsa.data.manifest import ImageRecord, Manifest
from pmsa.data.raw_audit import audit_manifests
from pmsa.features import FeatureSet
from pmsa.preprocessing import ImagePolicy

ROOT = Path(__file__).resolve().parents[1]


def image(path, color=10):
    Image.new("RGB", (12, 10), (color, 20, 30)).save(path)
    return ImageRecord(str(path), 0, "camera", "real")


def test_raw_audit_curates_decoded_identity_without_mutating_input(tmp_path):
    first = image(tmp_path / "a.png")
    extra = PngInfo()
    extra.add_text("different", "encoding metadata")
    Image.new("RGB", (12, 10), (10, 20, 30)).save(tmp_path / "b.png", pnginfo=extra)
    second = ImageRecord(str(tmp_path / "b.png"), 0, "camera", "real")
    report, curated = audit_manifests({"train": Manifest([first, second])})
    assert report["summary"]["train"]["independent_groups"] == 1
    assert report["images"][0]["sha256"] != report["images"][1]["sha256"]
    assert curated["train"].records[0].group_id == curated["train"].records[1].group_id
    assert first.group_id == ""


def test_raw_audit_rejects_cross_role_identity(tmp_path):
    record = image(tmp_path / "a.png")
    with pytest.raises(ValueError, match="overlaps across"):
        audit_manifests({"train": Manifest([record]), "test": Manifest([record])})


def test_raw_audit_rejects_conflicting_labels(tmp_path):
    record = image(tmp_path / "a.png")
    fake = copy.deepcopy(record)
    fake.label = 1
    with pytest.raises(ValueError, match="Mixed labels"):
        audit_manifests({"train": Manifest([record, fake])})


def test_raw_audit_rejects_curated_related_overlap(tmp_path):
    first, second = image(tmp_path / "a.png"), image(tmp_path / "b.png", 11)
    first.group_id = second.group_id = "same-original"
    with pytest.raises(ValueError, match="overlaps across"):
        audit_manifests({"train": Manifest([first]), "test": Manifest([second])})


def test_raw_audit_rejects_missing_and_invalid_images(tmp_path):
    missing = ImageRecord(str(tmp_path / "missing.png"), 0, "camera", "real")
    with pytest.raises(FileNotFoundError):
        audit_manifests({"train": Manifest([missing])})
    (tmp_path / "missing.png").write_text("not an image")
    with pytest.raises(ValueError, match="Cannot decode"):
        audit_manifests({"train": Manifest([missing])})


def test_sorted_scan_and_missing_directory(tmp_path):
    image(tmp_path / "z.png")
    image(tmp_path / "a.png", 11)
    assert Path(scan_real_dir(tmp_path, "photos", 1).records[0].path).name == "a.png"
    with pytest.raises(FileNotFoundError):
        scan_fake_dir(tmp_path / "missing", "fake")
    with pytest.raises(ValueError, match="positive"):
        scan_real_dir(tmp_path, "photos", 0)


def fresh_caches(tmp_path):
    rng = np.random.default_rng(123)
    for tag, n, offset in (("development", 800, 0), ("external", 200, 10000)):
        labels = (np.arange(n) % 2).astype(np.int8)
        paths = np.array([f"image_{i+offset}.png" for i in range(n)])
        hashes = np.array([f"{i+offset:064x}" for i in range(n)])
        for name in ("siglip", "forensic"):
            feats = rng.normal(size=(n, 4)).astype(np.float32)
            feats[:, 0] += labels
            metadata = {"schema_version": 1, "image_policy": ImagePolicy().to_dict(),
                "packages": {"Pillow": "synthetic-fixture"}, "backbone_kwargs": {}}
            if name == "siglip":
                metadata["backbone_kwargs"] = {"weights": "synthetic-fixture", "revision": "fixture", "use_fast": False}
            FeatureSet(feats, labels, paths, np.array(["test"]*n), np.where(labels, "fake", "real"),
                name, metadata, hashes, np.array([""]*n)).save(tmp_path / f"{name}_{tag}.npz")


def compare(tmp_path, out):
    return subprocess.run([sys.executable, str(ROOT / "scripts/10_evaluate_recovered.py"),
        "--cache-dir", str(tmp_path), "--train-tag", "development", "--test-tag", "external",
        "--require-metadata", "--models", "siglip_probe,forensic_probe,siglip_mlp,fusion",
        "--seeds", "0", "--epochs", "1", "--bootstrap", "3", "--cpu-threads", "1",
        "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=40)


@pytest.mark.parametrize("defect", ["legacy", "policy", "overlap"])
def test_fresh_comparison_fails_before_fitting(tmp_path, defect):
    pytest.importorskip("torch")
    fresh_caches(tmp_path)
    for name in ("siglip", "forensic"):
        path = tmp_path / f"{name}_external.npz"
        fs = FeatureSet.load(path)
        if defect == "legacy":
            fs.metadata, fs.content_hashes = {}, None
        elif defect == "policy":
            fs.metadata["image_policy"]["normalize_jpeg"] = True
        else:
            fs.content_hashes[0] = "0"*64  # development identity
        fs.save(path)
    out = tmp_path / "results"
    result = compare(tmp_path, out)
    assert result.returncode != 0
    assert not out.exists()


def test_fresh_four_head_smoke_has_complete_bundles_and_shared_splits(tmp_path):
    pytest.importorskip("torch")
    fresh_caches(tmp_path)
    out = tmp_path / "results"
    result = compare(tmp_path, out)
    assert result.returncode == 0, result.stdout+result.stderr
    entries = json.loads((out / "results.json").read_text())
    assert len(entries) == 4
    parts = []
    for entry in entries:
        assert not entry["training"]["exploratory_legacy"]
        assert not entry["external"]["exploratory_legacy"]
        directory = out / f'{entry["model"]}_seed0'
        assert (directory / "bundle.json").is_file()
        assert not (directory / "NOT_DEPLOYABLE.txt").exists()
        parts.append(json.loads((directory / "splits.json").read_text())["partitions"])
    assert all(part == parts[0] for part in parts)


def packager():
    spec = importlib.util.spec_from_file_location("kaggle_package", ROOT / "scripts/14_package_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.package


def test_package_excludes_private_materials_and_raw_images(tmp_path):
    repo = tmp_path / "repo"
    for filename in ("pmsa/a.py", "scripts/a.py", "examples/offline_smoke.py", "README.md", "docs/interview_book/main.tex",
                     "docs/ENGINEERING_STATUS.md", "docs/evidence/fresh_full_summary.json",
                     "feature_cache/old.npz", ".venv/a.py", "outputs/run/private.jpg", "outputs/run/summary.json"):
        path = repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    output = repo / "outputs" / "handoff.zip"
    packager()(repo, output, repo / "outputs/run")
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {"source/pmsa/a.py", "source/scripts/a.py", "source/README.md",
                                           "source/examples/offline_smoke.py", "source/docs/ENGINEERING_STATUS.md",
                                           "source/docs/evidence/fresh_full_summary.json",
                                           "source/source_manifest.json", "artifacts/summary.json"}
        manifest = json.loads(archive.read("source/source_manifest.json"))
        assert manifest["schema_version"] == 1
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read("source/"+name)).hexdigest() == digest
    assert output.with_suffix(".zip.sha256").is_file()
    with pytest.raises(FileExistsError):
        packager()(repo, output)


def test_notebook_is_clean_and_code_cells_compile():
    notebook = json.loads((ROOT / "notebooks/kaggle_fresh_evaluation.ipynb").read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), cell["id"], "exec")


def source_fixture(tmp_path):
    root = tmp_path / "input" / "source"
    root.mkdir(parents=True)
    files = {"pyproject.toml": b"[project]\nname='fixture'\n", "pmsa/fixture.py": b"value = 1\n"}
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {"schema_version": 1, "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    (root / "source_manifest.json").write_text(json.dumps(manifest))
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "x") as output:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                output.write(path, "source/"+str(path.relative_to(root)))
    return root, archive


def bootstrap():
    spec = importlib.util.spec_from_file_location("source_bootstrap", ROOT / "kaggle_bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", ["zip", "dataset", "project", "dataset-zip"])
def test_source_snapshot_accepts_kaggle_layouts(tmp_path, layout):
    root, archive = source_fixture(tmp_path)
    if layout == "dataset-zip":
        folder = tmp_path / "zip-dataset"
        folder.mkdir()
        (folder / "pmsa-kaggle-source.zip").write_bytes(archive.read_bytes())
        source = folder
    else:
        source = {"zip": archive, "dataset": root.parent, "project": root}[layout]
    module = bootstrap()
    expected, digest = module.load_source_snapshot(archive)
    output, actual = module.prepare_source_snapshot(source, tmp_path / "working")
    assert actual == digest
    for name, data in expected.items():
        assert (output / name).read_bytes() == data
    sentinel = output / "outputs/previous.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("keep")
    assert module.prepare_source_snapshot(source, tmp_path / "working") == (output, digest)
    assert sentinel.read_text() == "keep"


def test_source_tampering_fails_before_workspace_creation(tmp_path):
    root, archive = source_fixture(tmp_path)
    (root / "pmsa/fixture.py").write_text("value = 2")
    with pytest.raises(ValueError, match="integrity mismatch"):
        bootstrap().prepare_source_snapshot(root, tmp_path / "working")
    assert not (tmp_path / "working").exists()


def test_source_reuse_rejects_changed_code_without_overwrite(tmp_path):
    root, archive = source_fixture(tmp_path)
    module = bootstrap()
    output, _ = module.prepare_source_snapshot(archive, tmp_path / "working")
    target = output / "pmsa/fixture.py"
    target.write_text("user edit")
    with pytest.raises(ValueError, match="workspace changed"):
        module.prepare_source_snapshot(archive, tmp_path / "working")
    assert target.read_text() == "user edit"


def test_source_refuses_unmarked_existing_workspace(tmp_path):
    root, archive = source_fixture(tmp_path)
    workspace = tmp_path / "working"
    workspace.mkdir()
    sentinel = workspace / "saved-model.pt"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        bootstrap().prepare_source_snapshot(root, workspace)
    assert sentinel.read_bytes() == b"preserve"


@pytest.mark.parametrize("member", ["../escape.py", "/absolute.py", "source/../../escape.py"])
def test_archive_traversal_fails_before_workspace_creation(tmp_path, member):
    root, archive = source_fixture(tmp_path)
    with ZipFile(archive, "a") as output:
        output.writestr(member, "bad")
    with pytest.raises(ValueError, match="Unsafe source"):
        bootstrap().prepare_source_snapshot(archive, tmp_path / "working")
    assert not (tmp_path / "working").exists()


def test_unpacked_source_symlink_rejected(tmp_path):
    root, archive = source_fixture(tmp_path)
    target = root / "pmsa/fixture.py"
    outside = tmp_path / "outside.py"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        bootstrap().prepare_source_snapshot(root, tmp_path / "working")


def test_notebook_actual_setup_cell_runs_from_unpacked_source(tmp_path, monkeypatch):
    root, archive = source_fixture(tmp_path)
    notebook = json.loads((ROOT / "notebooks/kaggle_fresh_evaluation.ipynb").read_text())
    code = "".join(next(c["source"] for c in notebook["cells"] if c["id"] == "fresh-005"))
    assert code.startswith((ROOT / "kaggle_bootstrap.py").read_text().rstrip())
    monkeypatch.chdir(tmp_path)
    namespace = {"SOURCE_INPUT": root.parent, "SOURCE_ROOT": tmp_path / "working", "RUN_ID": "smoke-test"}
    exec(compile(code, "notebook-setup", "exec"), namespace)
    assert namespace["RUN"].is_dir()
    with pytest.raises(FileExistsError, match="Run already exists"):
        exec(compile(code, "notebook-setup", "exec"), namespace)
