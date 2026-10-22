"""Explicit-role, private development replay. Never changes serving artifacts.

Previously inspected generator images are permitted for development, not called
blind confirmation. Only broad TRAIN and modern TRAIN enter optimization. Two
separate real-only calibration pools are reported, never pooled or test-tuned.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .artifacts import check_runtime, save_bundle, sha256_file, verify_bundle
from .calibration import Calibrator
from .data.manifest import Manifest
from .data.splits import group_ids
from .eval.metrics import bootstrap_ci, operating_report, roc_auc
from .features.cache import FeatureSet, validate_alignment
from .features.extract import extract_backbone
from .models.fusion import FusionDetector

ROLES = ("train", "validation", "calibration", "test")
NAMES = ("siglip", "forensic")


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def compatible_metadata(left, right):
    """CPU/CUDA build suffixes may differ, but all extractor versions must match.

    Compatibility is not bitwise identity across devices. Do not silently accept
    a different processor, image policy, model revision or core package version.
    """
    def core(metadata):
        result = {**metadata, "packages": dict(metadata["packages"])}
        for package in ("torch", "torchvision"):
            result["packages"][package] = result["packages"][package].split("+", 1)[0]
        return result
    return core(left) == core(right)


def prepared_roles(directory):
    directory = Path(directory)
    plan = json.loads((directory / "split_plan.json").read_text())
    if (plan.get("protocol") != "reused-generator-development-v1"
            or plan.get("untouched_confirmation_test") is not False):
        raise ValueError("Requires the explicitly declared reused development protocol")
    for name in (*ROLES, "challenge"):
        filename = f"manifest_{name}.csv"
        if sha256_file(directory / filename) != plan["manifest_sha256"][filename]:
            raise ValueError(f"Changed role manifest: {name}")
    manifests = {role: Manifest.from_csv(directory / f"manifest_{role}.csv") for role in ROLES}
    if any(not len(m) or set(r.label for r in m.records) != {0, 1} for m in manifests.values()):
        raise ValueError("All development roles require both classes")
    paths = [r.path for m in manifests.values() for r in m.records]
    groups = [r.group_id for m in manifests.values() for r in m.records]
    if len(set(paths)) != len(paths) or not all(groups) or len(set(groups)) != len(groups):
        # This pilot uses one representative per approved scene/prompt group.
        raise ValueError("Duplicate image/group or missing group in explicit roles")
    return plan, manifests


def load_broad(archive_path, baseline_directory):
    baseline_directory = Path(baseline_directory)
    bundle = verify_bundle(baseline_directory)
    if bundle["model_type"] != "fusion" or tuple(bundle["streams"]) != NAMES:
        raise ValueError("Replay expects a trusted SigLIP/forensic fusion baseline")
    check_runtime(bundle["streams"])
    split = json.loads((baseline_directory / "splits.json").read_text())
    sets = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in NAMES:
            filename = f"{name}_train_v3.npz"
            data = archive.read(f"artifacts/feature_cache/{filename}")
            if hashlib.sha256(data).hexdigest() != split["cache_sha256"][filename]:
                raise ValueError(f"Broad cache integrity failure: {name}")
            fs = FeatureSet.load(io.BytesIO(data))
            if fs.backbone != name or fs.metadata != bundle["streams"][name]:
                raise ValueError("Broad cache does not match baseline extractor")
            sets.append(fs)
    validate_alignment(sets)
    partitions = {role: np.asarray(split["partitions"][role]["original_indices"], dtype=int) for role in ROLES}
    all_ix = np.concatenate(list(partitions.values()))
    if (np.any(all_ix < 0) or np.any(all_ix >= len(sets[0]))
            or len(np.unique(all_ix)) != len(all_ix)):
        raise ValueError("Invalid/repeated broad partition indices")
    groups = sets[0].groups()
    membership = []
    for role, ix in partitions.items():
        if groups[ix].tolist() != split["partitions"][role]["groups"]:
            raise ValueError("Broad group membership mismatch")
        if len(set(groups[ix])) != len(ix) or set(np.unique(sets[0].labels[ix])) != {0, 1}:
            raise ValueError("Broad roles need independent representatives and both classes")
        membership.append(set(groups[ix]))
    if any(a & b for i, a in enumerate(membership) for b in membership[i+1:]):
        raise ValueError("Broad cross-role group leakage")
    return bundle, split, sets, partitions


def prepare_features(roles_dir, selection_file, archive_path, baseline_dir, output):
    """One local extraction over the role-ordered manifest; no role reshuffle."""
    output = Path(output)
    plan, manifests = prepared_roles(roles_dir)
    if sha256_file(selection_file) != plan["selection_sha256"]:
        raise ValueError("Changed frozen selection")
    selection = json.loads(Path(selection_file).read_text())
    root = Path(selection["root"]).resolve(strict=True)
    hashes = {str(root / row["relative_path"]): row["sha256"] for row in selection["images"]}
    records = [r for m in manifests.values() for r in m.records]
    expected = []
    for r in records:
        if not Path(r.path).resolve(strict=True).is_relative_to(root):
            raise ValueError("Image outside selected private collection")
        digest = sha256_file(r.path)
        if hashes.get(r.path) != digest:
            raise ValueError("Changed image bytes after fixed selection")
        expected.append(digest)
    if len(records) != len(hashes):
        raise ValueError("Roles do not cover the frozen selection exactly")
    bundle, _, broad, _ = load_broad(archive_path, baseline_dir)
    if set(expected) & set(broad[0].content_hashes):
        raise ValueError("Modern image also occurs in broad development data")
    manifest = Manifest(records)
    offsets, offset = {}, 0
    for role, m in manifests.items():
        offsets[role] = list(range(offset, offset+len(m)))
        offset += len(m)
    output.mkdir(parents=True, exist_ok=True)
    # Register an already installed decoder locally. No install, upload or HEIC
    # conversion is performed; the one accepted HEIF retains its original bytes.
    if any(Path(r.path).suffix.lower() == ".heif" for r in records):
        from pillow_heif import register_heif_opener
        register_heif_opener()
    files, sets = {}, []
    for name in NAMES:
        target = output / f"{name}_modern.npz"
        receipt_path = output / f"{name}_receipt.json"
        if target.exists():
            if not receipt_path.exists() or sha256_file(target) != json.loads(receipt_path.read_text())["sha256"]:
                raise ValueError("Existing extraction has no valid receipt; choose a new output")
            fs = FeatureSet.load(target)
        else:
            metadata = bundle["streams"][name]
            kw = metadata["backbone_kwargs"]
            fs = extract_backbone(manifest, name, target, device="cpu", batch_size=8,
                                  num_workers=0, image_size=kw["image_size"],
                                  weights=kw.get("weights", ""), revision=kw.get("revision"),
                                  use_fast=kw.get("use_fast", False), normalize_jpeg=True)
            write_new_json(receipt_path, {"sha256": sha256_file(target),
                                         "split_plan_sha256": sha256_file(Path(roles_dir)/"split_plan.json")})
        if (not compatible_metadata(fs.metadata, bundle["streams"][name])
                or fs.paths.tolist() != [r.path for r in records]
                or fs.labels.tolist() != [r.label for r in records]
                or fs.domain.tolist() != [r.domain for r in records]
                or fs.source.tolist() != [r.source for r in records]
                or fs.related_groups.tolist() != [r.group_id for r in records]
                or fs.content_hashes.tolist() != expected):
            raise ValueError("New cache/role/extractor mismatch")
        if any(sha256_file(r.path) != digest for r, digest in zip(records, expected)):
            raise ValueError("Source changed during extraction")
        files[name] = sha256_file(target)
        sets.append(fs)
    validate_alignment(sets)
    receipt = {"schema_version": 1, "feature_sha256": files, "role_indices": offsets,
               "roles_dir": str(Path(roles_dir).resolve()),
               "split_plan_sha256": sha256_file(Path(roles_dir)/"split_plan.json"),
               "selection_sha256": plan["selection_sha256"], "device": "cpu",
               "n_images": len(records), "decoder": "pillow-heif installed; registered for HEIF",
               "no_images_uploaded": True, "untouched_confirmation_test": False}
    write_new_json(output / "extraction.json", receipt)
    return receipt


def load_modern(directory, roles_dir, broad):
    directory = Path(directory)
    receipt = json.loads((directory / "extraction.json").read_text())
    _, manifests = prepared_roles(roles_dir)
    if receipt["split_plan_sha256"] != sha256_file(Path(roles_dir)/"split_plan.json"):
        raise ValueError("Feature receipt binds a different split plan")
    sets = []
    for name, base in zip(NAMES, broad):
        path = directory / f"{name}_modern.npz"
        if sha256_file(path) != receipt["feature_sha256"][name]:
            raise ValueError("Changed extracted features")
        fs = FeatureSet.load(path)
        if (fs.backbone != name or fs.dim != base.dim
                or not compatible_metadata(fs.metadata, base.metadata)):
            raise ValueError("New features incompatible with broad features")
        sets.append(fs)
    validate_alignment(sets)
    partitions = {role: np.asarray(receipt["role_indices"][role], dtype=int) for role in ROLES}
    flat = np.concatenate(list(partitions.values()))
    if sorted(flat.tolist()) != list(range(len(sets[0]))):
        raise ValueError("Modern roles must cover each row exactly once")
    for role, ix in partitions.items():
        subset = sets[0].select(ix)
        recs = manifests[role].records
        for field, actual in (("path", subset.paths), ("label", subset.labels),
                              ("domain", subset.domain), ("source", subset.source),
                              ("group_id", subset.related_groups)):
            if actual.tolist() != [getattr(r, field) for r in recs]:
                raise ValueError("New feature role membership mismatch")
    # A global identity graph catches raw/path/group links across archives/roles.
    global_groups = group_ids(np.concatenate([broad[0].paths, sets[0].paths]),
                             np.concatenate([broad[0].content_hashes, sets[0].content_hashes]),
                             np.concatenate([broad[0].related_groups, sets[0].related_groups]))
    if set(global_groups[:len(broad[0])]) & set(global_groups[len(broad[0]):]):
        raise ValueError("Modern data overlap broad development history")
    memberships = [set(global_groups[len(broad[0])+ix]) for ix in partitions.values()]
    if any(a & b for i, a in enumerate(memberships) for b in memberships[i+1:]):
        raise ValueError("Modern cross-role leakage")
    return sets, partitions, receipt


def compare_replay(archive_path, baseline_dir, roles_dir, feature_dir, output,
                   *, seeds=(0, 1, 2), epochs=15):
    """Fixed recipe; seed 0 is primary, other seeds assess stability, not winners."""
    import torch
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new comparison output; never overwrite a run")
    if tuple(seeds) != (0, 1, 2):
        raise ValueError("This comparison prespecifies seeds 0,1,2; no best-seed search")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    bundle, base_split, broad, bi = load_broad(archive_path, baseline_dir)
    modern, mi, receipt = load_modern(feature_dir, roles_dir, broad)
    alpha = .05
    real_b = bi["calibration"][broad[0].labels[bi["calibration"]] == 0]
    real_m = mi["calibration"][modern[0].labels[mi["calibration"]] == 0]
    settings = {"modern_weight": .25, "epochs": epochs, "lr": 1e-4,
                "weight_decay": 1e-4, "batch_size": 256}
    plan = {"protocol": "fixed-role-warm-start-replay-v1", "primary_seed": 0,
            "seeds": list(seeds), "settings": settings, "alpha": alpha,
            "selection": "Weighted validation BCE, including unchanged starting checkpoint",
            "baseline_bundle_sha256": sha256_file(Path(baseline_dir)/"bundle.json"),
            "base_cache_sha256": base_split["cache_sha256"], "modern_extraction": receipt,
            "broad_role_counts": {k: len(v) for k,v in bi.items()},
            "modern_role_counts": {k: len(v) for k,v in mi.items()},
            "real_calibration_counts": {"broad": len(real_b), "modern": len(real_m)},
            "source_sha256": {p.name: sha256_file(p) for p in
                              (Path(__file__), Path(__file__).parent/"models/fusion.py")},
            "untouched_confirmation_test": False, "deploy_changed": False,
            "caveats": ["Generated collections and broad diagnostics were previously inspected.",
                        "21 local real calibration examples do not establish reliable deployment FPR.",
                        "Metadata/raw identities cannot rule out all semantic or transformed-image leakage.",
                        "Collection labels do not establish matched scenes/devices/encodings.",
                        "No seed/hyperparameter/threshold chosen from diagnostic results."]}
    output.mkdir(parents=True, exist_ok=True)
    write_new_json(output/"plan.json", plan)

    def xs(sets, ix):
        return [fs.features[ix] for fs in sets]

    def describe(scores, labels, tau):
        auc = bootstrap_ci(roc_auc, scores, labels, n=300, seed=917)
        return {"operating": operating_report(scores, labels, tau),
                "auc": {"point": auc.point, "lo": auc.lo, "hi": auc.hi}}

    def diagnostics(model, broad_cal, modern_cal):
        result = {}
        for name, sets, ix in (("broad_internal", broad, bi["test"]),
                               ("modern_internal", modern, mi["test"])):
            scores, labels = model.score(xs(sets,ix)), sets[0].labels[ix]
            result[name] = {"broad_tau": describe(scores,labels,broad_cal.tau),
                            "local_tau": describe(scores,labels,modern_cal.tau)}
            if name == "modern_internal":
                result[name]["by_generator_at_local_tau"] = {
                    source: operating_report(scores[(sets[0].source[ix] == source) | (labels == 0)],
                                             labels[(sets[0].source[ix] == source) | (labels == 0)], modern_cal.tau)
                    for source in sorted(set(sets[0].source[ix][labels == 1]))}
        return result

    initial = FusionDetector.load(Path(baseline_dir)/"fusion.pt", device="cpu")
    broad_cal = Calibrator.load(Path(baseline_dir)/"calibrator.json")
    local_cal = Calibrator.fit(initial.score(xs(modern,real_m)), alpha, real_domain="mixed_user_collection")
    results = {"baseline": {"tau_broad": broad_cal.tau, "tau_local": local_cal.tau,
                             "diagnostics": diagnostics(initial,broad_cal,local_cal)}, "candidates": []}
    write_new_json(output/"baseline.json", results["baseline"])
    for seed in seeds:
        print(f"Fine-tuning existing head, fixed roles, training seed {seed}", flush=True)
        model = FusionDetector.load(Path(baseline_dir)/"fusion.pt", device="cpu")
        model.fit_replay(xs(broad,bi["train"]), broad[0].labels[bi["train"]],
                         xs(modern,mi["train"]), modern[0].labels[mi["train"]],
                         xs(broad,bi["validation"]), broad[0].labels[bi["validation"]],
                         xs(modern,mi["validation"]), modern[0].labels[mi["validation"]],
                         seed=seed, **settings)
        bc = Calibrator.fit(model.score(xs(broad,real_b)), alpha, real_domain="broad_mixed")
        mc = Calibrator.fit(model.score(xs(modern,real_m)), alpha, real_domain="mixed_user_collection")
        run = output/f"seed{seed}"
        run.mkdir()
        model.save(run/"fusion.pt")
        mc.save(run/"calibrator.json")
        bc.save(run/"broad_calibrator.json")
        write_new_json(run/"splits.json", {"development_ids": {
            field: np.concatenate([getattr(broad[0],field),getattr(modern[0],field)]).tolist()
            for field in ("paths","content_hashes","related_groups")},
            "base_partitions": base_split["partitions"], "modern_partitions": receipt["role_indices"],
            "training_seed": seed, "explicit_roles": True, "modern_index_offset": len(broad[0]),
            "calibrator_scope": "Local collection only; broad threshold reported separately"})
        save_bundle(run,"fusion.pt","fusion",{fs.backbone: fs.metadata for fs in modern})
        verify_bundle(run)
        reloaded = FusionDetector.load(run/"fusion.pt")
        np.testing.assert_allclose(model.score(xs(modern,mi["test"])),
                                   reloaded.score(xs(modern,mi["test"])), atol=1e-6, rtol=1e-6)
        np.savez_compressed(run/"predictions.npz",
                            modern_scores=model.score(xs(modern,mi["test"])),
                            modern_labels=modern[0].labels[mi["test"]],
                            modern_sources=modern[0].source[mi["test"]],
                            broad_scores=model.score(xs(broad,bi["test"])),
                            broad_labels=broad[0].labels[bi["test"]])
        result = {"seed":seed, "validation":model.fit_history,
                  "tau_broad":bc.tau,"tau_local":mc.tau,
                  "diagnostics": diagnostics(model,bc,mc)}
        write_new_json(run/"result.json",result)
        results["candidates"].append(result)
    write_new_json(output/"results.json",results)
    return results
