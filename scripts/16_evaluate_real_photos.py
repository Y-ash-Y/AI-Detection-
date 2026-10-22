"""Prepare then execute a private, frozen-threshold camera-photo diagnostic.

Use --prepare first; scores cannot influence sampling or model choice. This tool
never calibrates, uploads photos, publishes receipts, or changes deployment.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.artifacts import sha256_file, verify_bundle
from pmsa.photo_evaluation import inventory_photos, summarize_real_scores

MODEL_NAMES = [f"{name}_seed{seed}" for name in ("fusion", "siglip_mlp") for seed in (0, 1, 2)]


def json_write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def prepare(directory, models_root, out):
    if out.exists():
        raise FileExistsError("Plan directory exists; preserve it and choose a new name")
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass  # HEIC decode fails explicitly if the plugin is required.
    inventory = inventory_photos(directory, record_decode_failures=True)
    bundles = []
    for name in MODEL_NAMES:
        path = (models_root / name).resolve(strict=True)
        bundle = verify_bundle(path)
        if bundle["model_type"] != "fusion":
            raise ValueError("Only trusted tensor-only neural bundles accepted")
        bundles.append({"name": name, "path": str(path),
                        "bundle_sha256": sha256_file(path / "bundle.json")})
    plan = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256_file(__file__), "bundles": bundles,
            "primary_model": "fusion_seed0", "selection": "Default seed 0, not chosen from photo scores; all six fixed neural bundles reported, no best-seed promotion.",
            "labels": "User-attested camera-origin real images; normal processing/sharing possible; not independently authenticated.",
            "test_only": True, "fit_threshold": False, "uncertain_margin": .25,
            "inventory": inventory,
            "caveats": ["Small convenience sample, not representative of all devices or subjects.",
                        "Exact deduplication does not remove bursts, crops, or recompressions.",
                        "No target-real recalibration: all supplied photos are test-only.",
                        "Local macOS runtime matching is not cross-platform byte identity or a hosted-upload test."]}
    out.mkdir(parents=True)
    json_write(out / "plan.json", plan)
    print(f"Prepared {inventory['file_count']} files, {inventory['unique_decoded_count']} exact-unique readable images, {inventory['decode_failures']} input failures; no detector scores inspected.")


def execute(out):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import numpy as np
    import torch
    import pillow_heif
    pillow_heif.register_heif_opener()
    from PIL import Image
    from pmsa.inference import Detector
    from pmsa.parity import runtime_preflight

    plan_path = out / "plan.json"
    plan = json.loads(plan_path.read_text())
    if (out / "results.json").exists() or any(out.glob("*_scores.json")):
        raise FileExistsError("Scores already exist; do not repeatedly evaluate or overwrite this test")
    if plan["runner_sha256"] != sha256_file(__file__):
        raise ValueError("Runner changed since the plan was frozen")
    if not plan["test_only"] or plan["fit_threshold"]:
        raise ValueError("This workflow is test-only; no threshold fitting allowed")
    if [b["name"] for b in plan["bundles"]] != MODEL_NAMES:
        raise ValueError("Fixed model set changed")
    images = plan["inventory"]["images"]
    for row in images:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError("Photo changed since plan creation")
    for selected in plan["bundles"]:
        directory = Path(selected["path"])
        if sha256_file(directory / "bundle.json") != selected["bundle_sha256"]:
            raise ValueError("Bundle changed since plan creation")
        bundle = verify_bundle(directory)
        preflight = runtime_preflight(bundle)
        if not preflight["compatible"]:
            raise ValueError(preflight["reason"])
    torch.set_num_threads(4)
    results = []
    # Scoring is CPU-only and Hub is offline; no photo data ever leaves this process.
    for selected in plan["bundles"]:
        directory = Path(selected["path"])
        bundle = verify_bundle(directory)
        det = Detector(directory / bundle["model_file"], directory / "calibrator.json",
                       device="cpu", uncertain_margin=plan["uncertain_margin"])
        unique_scores, rows = {}, []
        for i, row in enumerate(images):
            if not row["decode_ok"]:
                rows.append({"index": i, "image_sha256": row["sha256"],
                             "status": "unsupported-input", "reason": row["reason"]})
                continue
            representative = row["representative"]
            if representative not in unique_scores:
                with Image.open(row["path"]) as image:
                    streams = det._features(image)
                score, contributions, bias = det._score_and_streams(streams)
                if (not np.isfinite(score) or not np.isclose(sum(contributions.values()) + bias, score,
                                                           atol=2e-4, rtol=1e-4)):
                    raise ValueError("Nonfinite score or contribution sum inconsistency")
                unique_scores[representative] = score
            rows.append({"index": i, "image_sha256": row["sha256"], "status": "scored",
                         "score": unique_scores[representative],
                         "threshold_exceeded": bool(unique_scores[representative] > det.calibrator.tau)})
        unique = [unique_scores[i] for i in plan["inventory"]["representative_indices"]]
        report = {"model": selected["name"],
                  "exact_unique": summarize_real_scores(unique, det.calibrator.tau),
                  "readable_files": summarize_real_scores([r["score"] for r in rows if r["status"] == "scored"], det.calibrator.tau),
                  "input_files": len(rows), "unsupported_inputs": plan["inventory"]["decode_failures"],
                  "rows": rows}
        json_write(out / f"{selected['name']}_scores.json", report)
        results.append(report)
        stats = report["exact_unique"]
        print(f"{selected['name']}: {stats['false_alarms']}/{stats['n_real']} false alarms", flush=True)
        del det
    json_write(out / "results.json", {"plan_sha256": sha256_file(plan_path),
               "primary_model": plan["primary_model"], "models": results,
               "caveats": plan["caveats"], "independence_verified": False,
               "deployment_changed": False, "photos_uploaded": False})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--photos")
    ap.add_argument("--models-root")
    args = ap.parse_args()
    if args.prepare:
        if not args.photos or not args.models_root:
            ap.error("--prepare requires --photos and --models-root")
        prepare(Path(args.photos), Path(args.models_root), Path(args.out))
    else:
        execute(Path(args.out))


if __name__ == "__main__":
    main()
