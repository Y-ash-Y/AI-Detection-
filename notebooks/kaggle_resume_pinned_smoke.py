"""Paste this whole file into ONE new Kaggle code cell after the failed extraction.

Requires the existing notebook's smoke setup and audit. It re-extracts at an
explicit revision, preserving the unpinned caches. Run the four-head comparison
cell next, NOT the original extraction cell. No source edits, installs or pushes.
"""
from pathlib import Path
import json
from unittest.mock import patch

from huggingface_hub import try_to_load_from_cache
import pmsa.features.extract as extraction
from pmsa.data.manifest import Manifest
from pmsa.features import FeatureSet
from pmsa.features.cache import validate_alignment
from pmsa.artifacts import validate_stream_metadata, sha256_file
import yaml

if MODE != "smoke" or plan["mode"] != "smoke":
    raise ValueError("This recovery cell is ONLY for the small smoke run")

WEIGHTS = "google/siglip-base-patch16-224"
PIN = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
cached_config = try_to_load_from_cache(WEIGHTS, "config.json", revision=PIN)
if not isinstance(cached_config, str):
    raise ValueError("Expected model revision is not cached; stop and inspect")
cached_config = Path(cached_config)
if (not cached_config.is_file() or cached_config.parent.name != PIN
        or cached_config.parent.parent.name != "snapshots"):
    raise ValueError("Cannot verify the downloaded snapshot path")

old_cache = Path(CACHE)
new_cache = RUN / "feature_cache_pinned_01"
new_config = RUN / "config_pinned_01.yaml"
note = RUN / "pinning_recovery_01.json"
if any(p.exists() for p in (new_cache, new_config, note)):
    raise FileExistsError("Recovery attempt exists; do not overwrite it or rerun this cell")
manifests = {role: Manifest.from_csv(AUDIT / f"{role}.csv") for role in ("train", "test")}
if any(len(m) > 1000 for m in manifests.values()):
    raise ValueError("Unexpected image count for smoke; no extraction started")
for role, manifest in manifests.items():
    old_streams = [FeatureSet.load(old_cache / f"{name}_{role}_v3.npz")
                   for name in ("siglip", "forensic")]
    validate_alignment(old_streams)
    if len(old_streams[0]) != len(manifest):
        raise ValueError("Existing caches and audit manifest disagree")
    kw = old_streams[0].metadata["backbone_kwargs"]
    if kw.get("weights") != WEIGHTS or kw.get("use_fast") is not False:
        raise ValueError("Unexpected original model/processor; inspect before recovery")

# Use the original builder, but pin BEFORE it loads weights and the processor.
# Patching is temporary and scoped to this cell's Python extraction calls.
original_builder = extraction.build_backbone
def pinned_builder(name, **kwargs):
    if name == "siglip":
        kwargs.update(weights=WEIGHTS, revision=PIN, use_fast=False)
    return original_builder(name, **kwargs)

cfg = yaml.safe_load(Path(CONFIG).read_text())
cfg["feature"]["cache_dir"] = str(new_cache)
backbone_config = {b["name"]: b for b in cfg["feature"]["backbones"]}
if not {"siglip", "forensic"} <= backbone_config.keys():
    raise ValueError("Configuration lacks the required backbone settings")
new_cache.mkdir()
new_config.write_text(yaml.safe_dump(cfg))
recovery = {"status": "started", "weights": WEIGHTS, "revision": PIN,
            "processor_use_fast": False, "method": "explicit revision before model and processor loads",
            "source_snapshot_sha256": source_digest,
            "cached_config_sha256": sha256_file(cached_config),
            "original_caches": {str(p): sha256_file(p) for p in sorted(old_cache.glob("*.npz"))},
            "new_cache": str(new_cache), "new_config": str(new_config),
            "rows": {role: len(m) for role, m in manifests.items()}}
note.write_text(json.dumps(recovery, indent=2))
with patch.object(extraction, "build_backbone", side_effect=pinned_builder):
    for role, manifest in manifests.items():
        for name in ("siglip", "forensic"):
            extraction.extract_backbone(manifest, name, new_cache / f"{name}_{role}_v3.npz",
                device="cuda", batch_size=32, num_workers=0,
                weights=backbone_config[name].get("weights", ""),
                image_size=backbone_config[name]["image_size"],
                normalize_jpeg=True)
for role in manifests:
    streams = [FeatureSet.load(new_cache / f"{name}_{role}_v3.npz") for name in ("siglip", "forensic")]
    validate_alignment(streams)
    validate_stream_metadata({s.backbone: s.metadata for s in streams})
    if streams[0].metadata["backbone_kwargs"]["revision"] != PIN:
        raise ValueError("New cache did not record the requested revision")
    print(role, [(s.backbone, s.features.shape) for s in streams], "revision:", PIN)
if any(sha256_file(path) != digest for path, digest in recovery["original_caches"].items()):
    raise ValueError("Original caches changed unexpectedly")
recovery["status"] = "validated"
note.write_text(json.dumps(recovery, indent=2))
# Only switch the next evaluation cell to these inputs after every check passes.
CACHE, CONFIG = new_cache, new_config
archive_stage("features-pinned")
print("Pinned smoke extraction passed. Run the FOUR-HEAD COMPARISON cell next.")
