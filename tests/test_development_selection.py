"""Explicit development reuse must never manufacture a blind test."""
import csv
import json

import pytest
from PIL import Image

from pmsa.data.intake import select_development, inventory, _fingerprint, build_development
from pmsa.artifacts import sha256_file


def data(tmp_path):
    root = tmp_path / "photos"
    for folder, offset in (("REAL", 0), ("AI generated/gpt", 50), ("AI generated/gemini", 100)):
        (root / folder).mkdir(parents=True)
        for i in range(8):
            Image.new("RGB", (8, 8), (offset+i, 40, 70)).save(root / folder / f"{i}.png")
    oldreal, oldfake = root / "REAL/0.png", root / "AI generated/gpt/0.png"
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": 1, "images": [
        {"sha256": sha256_file(p), "label": label, **_fingerprint(p)}
        for p, label in ((oldreal, 0), (oldfake, 1))]}))
    (root / "AI generated/gpt/copy.png").write_bytes(oldfake.read_bytes())
    snapshot = tmp_path / "inventory"
    inventory(root, registry, snapshot)
    return root, snapshot / "inventory.json", {"gpt": "AI generated/gpt", "gemini": "AI generated/gemini"}


def test_explicit_reuse_retains_history_and_no_roles(tmp_path):
    root, snapshot, folders = data(tmp_path)
    out = tmp_path / "selection"
    result = select_development(snapshot, folders, out, per_generator=8, fully_generated=True)
    assert result["selected_counts"] == {"gemini": 8, "gpt": 8, "real": 7}
    assert result["untouched_test_created"] is False and result["role_manifests_created"] is False
    assert sum(r["source"] == "gpt" and r["status"] == "reserved-regression" for r in result["images"]) == 1
    assert not list(out.glob("manifest*"))
    with (out / "review.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(not r["approved"] and not r["group_id"] for r in rows)
    assert all(r["prior_use"] == "previously_used" and r["origin_kind"] == "fully_generated"
               for r in rows if r["label"] == "1")
    assert len({r["decoded_rgb_sha256"] for r in result["images"]}) == 23
    assert (root / "AI generated/gpt/copy.png").exists()
    with pytest.raises(FileExistsError):
        select_development(snapshot, folders, out, per_generator=8)


def test_content_rank_is_deterministic_and_seed_controlled(tmp_path):
    _, snapshot, folders = data(tmp_path)
    a = select_development(snapshot, folders, tmp_path / "a", per_generator=3)
    b = select_development(snapshot, folders, tmp_path / "b", per_generator=3)
    assert [r["sha256"] for r in a["images"]] == [r["sha256"] for r in b["images"]]
    assert all(not r["origin_kind"] for r in csv.DictReader((tmp_path / "a/review.csv").open()) if r["label"] == "1")
    with pytest.raises(ValueError, match="Insufficient"):
        select_development(snapshot, folders, tmp_path / "too_many", per_generator=9)
    assert not (tmp_path / "too_many").exists()


def test_mutated_input_and_overlapping_directories_fail(tmp_path):
    root, snapshot, folders = data(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        select_development(snapshot, {"a": "AI generated", "b": "AI generated/gpt"}, tmp_path / "overlap")
    (root / "REAL/1.png").write_bytes((root / "REAL/2.png").read_bytes())
    with pytest.raises(ValueError, match="Input changed"):
        select_development(snapshot, folders, tmp_path / "changed", per_generator=3)
    assert not (tmp_path / "changed").exists()


def test_source_conflict_is_not_silently_sampled(tmp_path):
    root, snapshot, folders = data(tmp_path)
    (root / "AI generated/gemini/7.png").write_bytes((root / "AI generated/gpt/7.png").read_bytes())
    fresh = tmp_path / "fresh"
    inventory(root, tmp_path / "registry.json", fresh)
    with pytest.raises(ValueError, match="across declared"):
        select_development(fresh / "inventory.json", folders, tmp_path / "conflict", per_generator=3)


def reviewed_selection(tmp_path):
    _, snapshot, folders = data(tmp_path)
    out = tmp_path / "selection"
    select_development(snapshot, folders, out, per_generator=8, fully_generated=True)
    with (out / "review.csv").open() as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for r in rows:
        r.update(domain="user_collection", group_id=r["candidate_id"], processing="curator_reported_export",
                 prior_use="new" if r["label"] == "0" else "previously_used", approved="yes")
    path = out / "reviewed.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out, path, fields, rows


def test_development_roles_preserve_history_and_group_boundaries(tmp_path):
    out, review, _, _ = reviewed_selection(tmp_path)
    result = build_development(out / "selection.json", review, tmp_path / "roles", alpha=.5)
    assert result["untouched_confirmation_test"] is False
    assert result["test_role"] == "internal-diagnostic-not-blind-confirmation"
    assert result["reused_generated_candidates"] == 16
    assert result["manifest_sha256"]["manifest_train.csv"] == sha256_file(tmp_path / "roles/manifest_train.csv")
    assert sum(v["rows"] for v in result["summary"].values()) == 23
    groups = []
    for role in ("train", "validation", "calibration", "test"):
        with (tmp_path / f"roles/manifest_{role}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert {r["source"] for r in rows} == {"real", "gpt", "gemini"}
        groups.append({r["group_id"] for r in rows})
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])


def test_previously_used_generated_cannot_be_relabelled_new(tmp_path):
    out, review, fields, rows = reviewed_selection(tmp_path)
    next(r for r in rows if r["label"] == "1")["prior_use"] = "new"
    with review.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="retain previously_used"):
        build_development(out / "selection.json", review, tmp_path / "roles", alpha=.5)
    assert not (tmp_path / "roles").exists()


def test_selection_and_review_source_changes_fail_before_roles(tmp_path):
    out, review, fields, rows = reviewed_selection(tmp_path)
    next(r for r in rows if r["label"] == "1")["source"] = "different_generator"
    with review.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="source changed"):
        build_development(out / "selection.json", review, tmp_path / "roles", alpha=.5)
    selection = json.loads((out / "selection.json").read_text())
    selection["images"][0]["sha256"] = "f" * 64
    (out / "tampered.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="does not match inventory"):
        build_development(out / "tampered.json", review, tmp_path / "roles", alpha=.5)
