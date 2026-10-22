"""Offline, synthetic-fixture contracts for reviewed image intake. No models."""
import csv
import json
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest

from pmsa.artifacts import sha256_file
from pmsa.data.intake import inventory, build, _fingerprint, reserve_history
from pmsa.data.manifest import Manifest


def fixture_image(path, number):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), (number % 256, number // 256, 80)).save(path)
    return path


def registry(tmp_path, paths=()):
    output = tmp_path / "reserved.json"
    output.write_text(json.dumps({"schema_version": 1, "images": [
        {"sha256": sha256_file(path), "label": 0, **_fingerprint(path)} for path in paths]}))
    return output


def collection(tmp_path, per_class=20):
    root = tmp_path / "photos"
    for i in range(per_class):
        fixture_image(root / "real" / f"r{i}.png", i)
        fixture_image(root / "ai_generated" / f"f{i}.png", i + 100)
    snapshot = tmp_path / "intake"
    inventory(root, registry(tmp_path), snapshot)
    review(snapshot / "review.csv")
    return root, snapshot


def review(path, mutate=None):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(domain="people", processing="original_export", group_id=row["candidate_id"],
                   origin_kind="camera" if row["label"] == "0" else "fully_generated",
                   source="real" if row["label"] == "0" else "chatgpt_unknown_version",
                   prior_use="new", approved="yes")
    if mutate:
        mutate(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def freeze(snapshot, output, **kwargs):
    return build(snapshot / "inventory.json", snapshot / "review.csv", output, alpha=.2, **kwargs)


def test_inventory_reserves_renamed_and_lossless_reencoded_images(tmp_path):
    prior = fixture_image(tmp_path / "previous.png", 10)
    reserve = registry(tmp_path, [prior])
    root = tmp_path / "photos"
    root.mkdir()
    (root / "renamed.png").write_bytes(prior.read_bytes())
    info = PngInfo()
    info.add_text("different", "metadata")
    with Image.open(prior) as im:
        im.save(root / "reencoded.png", pnginfo=info)
    before = sha256_file(root / "renamed.png")
    report = inventory(root, reserve, tmp_path / "intake")
    assert report["counts"] == {"reserved-regression": 2}
    assert sha256_file(root / "renamed.png") == before
    with pytest.raises(FileExistsError):
        inventory(root, reserve, tmp_path / "intake")


def test_flat_unknown_filenames_do_not_infer_labels(tmp_path):
    root = tmp_path / "photos"
    fixture_image(root / "ChatGPT-looking-name.png", 10)
    output = tmp_path / "intake"
    inventory(root, registry(tmp_path), output)
    with (output / "review.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["label"] == "" and row["status"] == "ambiguous-folder"
    with pytest.raises(ValueError, match="Unapproved"):
        freeze(output, tmp_path / "splits")
    assert not (tmp_path / "splits").exists()


def test_declared_class_folder_aliases_are_case_and_space_insensitive(tmp_path):
    root = tmp_path / "photos"
    fixture_image(root / "REAL" / "camera.png", 10)
    fixture_image(root / "AI generated" / "generated.png", 20)
    fixture_image(root / "surreal" / "ambiguous.png", 30)
    output = tmp_path / "intake"
    report = inventory(root, registry(tmp_path), output)
    assert report["counts"] == {"pending-review": 2, "ambiguous-folder": 1}
    with (output / "review.csv").open() as handle:
        labels = {row["relative_path"]: row["label"] for row in csv.DictReader(handle)}
    assert labels == {"REAL/camera.png": "0", "AI generated/generated.png": "1", "surreal/ambiguous.png": ""}


def test_group_split_is_disjoint_deterministic_and_no_overwrite(tmp_path):
    root, snapshot = collection(tmp_path)
    before = {p: sha256_file(p) for p in root.rglob("*.png")}
    a, b = tmp_path / "split_a", tmp_path / "split_b"
    result = freeze(snapshot, a)
    freeze(snapshot, b)
    sets = []
    for role in ("train", "validation", "calibration", "test"):
        assert (a / f"manifest_{role}.csv").read_bytes() == (b / f"manifest_{role}.csv").read_bytes()
        manifest = Manifest.from_csv(a / f"manifest_{role}.csv")
        assert {r.label for r in manifest.records} == {0, 1}
        sets.append({r.group_id for r in manifest.records})
    assert all(not x & y for i, x in enumerate(sets) for y in sets[i + 1:])
    assert result["real_calibration_groups"] >= result["minimum_for_finite_conformal_threshold"]
    assert all(sha256_file(p) == digest for p, digest in before.items())
    with pytest.raises(FileExistsError):
        freeze(snapshot, a)


def test_copies_and_related_scene_stay_in_same_role(tmp_path):
    root, snapshot = collection(tmp_path)
    rows = review(snapshot / "review.csv", lambda rows: [row.update(group_id="same-burst")
                  for row in rows if row["relative_path"] in ("real/r0.png", "real/r1.png")])
    output = tmp_path / "splits"
    freeze(snapshot, output)
    roles = [role for role in ("train", "validation", "calibration", "test")
             if sum(Path(r.path).name in ("r0.png", "r1.png")
                    for r in Manifest.from_csv(output / f"manifest_{role}.csv").records)]
    assert len(roles) == 1


@pytest.mark.parametrize("defect,match", [("label", "disagreement"), ("review", "Unapproved"),
                                         ("domain", "domain"), ("processing", "processing"),
                                         ("prior", "Prior-use"), ("origin", "Origin kind")])
def test_incomplete_or_inconsistent_annotations_fail_before_output(tmp_path, defect, match):
    _, snapshot = collection(tmp_path)
    def mutate(rows):
        row = next(r for r in rows if r["label"] == "0")
        field, value = {"label": ("label", "1"), "review": ("approved", ""), "domain": ("domain", ""),
                        "processing": ("processing", "unknown"), "prior": ("prior_use", ""),
                        "origin": ("origin_kind", "fully_generated")}[defect]
        row[field] = value
    review(snapshot / "review.csv", mutate)
    with pytest.raises(ValueError, match=match):
        freeze(snapshot, tmp_path / "splits")
    assert not (tmp_path / "splits").exists()


def test_file_change_and_registry_change_fail_closed(tmp_path):
    root, snapshot = collection(tmp_path)
    fixture_image(root / "real/r0.png", 70)
    with pytest.raises(ValueError, match="Input changed"):
        freeze(snapshot, tmp_path / "splits")
    fixture_image(root / "real/r0.png", 0)
    (tmp_path / "reserved.json").write_text('{"schema_version": 1, "images": [], "changed": true}')
    with pytest.raises(ValueError, match="registry changed"):
        freeze(snapshot, tmp_path / "splits")


def test_reserved_csv_override_cannot_turn_test_photo_into_training_data(tmp_path):
    prior = fixture_image(tmp_path / "prior.png", 200)
    root = tmp_path / "photos"
    for i in range(20):
        fixture_image(root / "real" / f"r{i}.png", i)
        fixture_image(root / "ai_generated" / f"f{i}.png", i + 100)
    (root / "real/reserved.png").write_bytes(prior.read_bytes())
    snapshot = tmp_path / "intake"
    inventory(root, registry(tmp_path, [prior]), snapshot)
    review(snapshot / "review.csv", lambda rows: [r.update(status="pending-review", prior_use="new", approved="yes")
                                                  for r in rows if r["relative_path"] == "real/reserved.png"])
    result = freeze(snapshot, tmp_path / "splits")
    assert any(r["reason"] == "reserved-regression" for r in result["excluded"])
    assert all("reserved.png" not in (tmp_path / "splits" / f"manifest_{role}.csv").read_text()
               for role in ("train", "validation", "calibration", "test"))


def test_prior_related_component_and_graphics_are_excluded_from_primary(tmp_path):
    _, snapshot = collection(tmp_path, per_class=24)
    def mutate(rows):
        for row in rows:
            if row["relative_path"] in ("real/r0.png", "real/r1.png"):
                row["group_id"] = "known-scene"
            if row["relative_path"] == "real/r0.png":
                row["prior_use"] = "previously_used"
            if row["relative_path"] == "real/r2.png":
                row["origin_kind"] = "conventional_graphic"
    review(snapshot / "review.csv", mutate)
    result = freeze(snapshot, tmp_path / "splits")
    assert sum(r["reason"] == "component-related-to-prior-use" for r in result["excluded"]) == 2
    assert result["challenge_rows"] == 1


def test_original_and_ai_edit_pair_are_kept_together_in_challenge(tmp_path):
    _, snapshot = collection(tmp_path, per_class=24)
    def mutate(rows):
        for row in rows:
            if row["relative_path"] in ("real/r0.png", "ai_generated/f0.png"):
                row["group_id"] = "original-and-edit"
            if row["relative_path"] == "ai_generated/f0.png":
                row["origin_kind"] = "ai_edit"
    review(snapshot / "review.csv", mutate)
    result = freeze(snapshot, tmp_path / "splits")
    assert result["challenge_rows"] == 2
    challenge = Manifest.from_csv(tmp_path / "splits/manifest_challenge.csv")
    assert len({r.group_id for r in challenge.records}) == 1 and {r.label for r in challenge.records} == {0, 1}


def test_mixed_labels_without_disclosed_edit_fail(tmp_path):
    _, snapshot = collection(tmp_path)
    review(snapshot / "review.csv", lambda rows: [r.update(group_id="invalid-cross-label")
          for r in rows if r["relative_path"] in ("real/r0.png", "ai_generated/f0.png")])
    with pytest.raises(ValueError, match="Mixed-label"):
        freeze(snapshot, tmp_path / "splits")


def test_missing_class_domain_and_insufficient_calibration_fail(tmp_path):
    _, snapshot = collection(tmp_path)
    review(snapshot / "review.csv", lambda rows: [r.update(domain="objects") for r in rows if r["label"] == "1"])
    with pytest.raises(ValueError, match="real AND generated"):
        freeze(snapshot, tmp_path / "splits")
    review(snapshot / "review.csv")
    with pytest.raises(ValueError, match="independent real calibration"):
        build(snapshot / "inventory.json", snapshot / "review.csv", tmp_path / "splits", alpha=.05)


def test_reserved_group_annotation_protects_new_related_copy(tmp_path):
    prior = fixture_image(tmp_path / "prior.png", 200)
    root = tmp_path / "photos"
    for i in range(24):
        fixture_image(root / "real" / f"r{i}.png", i)
        fixture_image(root / "ai_generated" / f"f{i}.png", i + 100)
    (root / "real/old.png").write_bytes(prior.read_bytes())
    snapshot = tmp_path / "intake"
    inventory(root, registry(tmp_path, [prior]), snapshot)
    review(snapshot / "review.csv", lambda rows: [r.update(group_id="same-old-burst")
          for r in rows if r["relative_path"] in ("real/old.png", "real/r0.png")])
    result = freeze(snapshot, tmp_path / "splits")
    assert any(r["reason"] == "component-related-to-prior-use" for r in result["excluded"])


def test_outside_links_and_outputs_inside_images_are_rejected(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    outside = fixture_image(tmp_path / "outside.png", 10)
    (root / "linked.png").symlink_to(outside)
    reserve = registry(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        inventory(root, reserve, tmp_path / "intake")
    with pytest.raises(ValueError, match="outside"):
        inventory(root, reserve, root / "intake")


def test_reserve_history_preserves_all_labels_and_no_overwrite(tmp_path):
    a, b = tmp_path / "real_plan.json", tmp_path / "fake_plan.json"
    a.write_text(json.dumps({"inventory": {"images": [{"sha256": "a"*64}]}}))
    b.write_text(json.dumps({"new_fake_images": [{"sha256": "b"*64}]}))
    out = tmp_path / "reserved.json"
    result = reserve_history(a, b, out)
    assert [r["label"] for r in result["images"]] == [0, 1]
    with pytest.raises(FileExistsError):
        reserve_history(a, b, out)
