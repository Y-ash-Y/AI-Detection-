#!/usr/bin/env python
"""Check a trusted bundle's runtime and image/batch/cache parity; no training or deployment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.artifacts import verify_bundle
from pmsa.features import FeatureSet
from pmsa.inference import Detector
from pmsa.parity import image_serving_parity, runtime_preflight


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--images", nargs="+", help="Original image files; no labels are inferred")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--reference-cache", help="Optional cache directory containing ORIGINAL extraction")
    ap.add_argument("--tag", default="test_v3")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--out", required=True, help="New JSON receipt; never overwritten")
    args = ap.parse_args()
    out, directory = Path(args.out), Path(args.bundle)
    if out.exists():
        raise FileExistsError("Receipt exists; choose a new output filename")
    if not args.preflight_only and not args.images:
        ap.error("--images required unless --preflight-only")
    bundle = verify_bundle(directory)
    runtime = runtime_preflight(bundle)
    report = {"runtime": runtime, "status": "preflight-passed" if runtime["compatible"] else "blocked-runtime"}
    if runtime["compatible"] and not args.preflight_only:
        det = Detector(directory / bundle["model_file"], directory / "calibrator.json", device=args.device)
        reference = ([FeatureSet.load(Path(args.reference_cache) / f"{name}_{args.tag}.npz")
                      for name in bundle["streams"]] if args.reference_cache else None)
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        report.update(image_serving_parity(det, args.images, batch_size=args.batch_size,
                                          reference_sets=reference))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"{report['status']} -> {out}")
    if not runtime["compatible"]:
        print(runtime["reason"])
    return 0 if report["status"] in ("passed", "preflight-passed") else 2


if __name__ == "__main__":
    sys.exit(main())
