"""Stage a minimal, sanitized serving release. Does NOT authenticate or upload."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pmsa.artifacts import sha256_file, verify_bundle, validate_stream_metadata

SERVING_FILES = (
    "app.py", "pmsa/__init__.py", "pmsa/artifacts.py", "pmsa/inference.py",
    "pmsa/preprocessing.py", "pmsa/provenance.py", "pmsa/backbones/__init__.py",
    "pmsa/backbones/base.py", "pmsa/backbones/siglip_encoder.py", "pmsa/backbones/forensic.py",
    "pmsa/backbones/hub_revision.py", "pmsa/backbones/clip_encoder.py",
    "pmsa/backbones/dino_encoder.py", "pmsa/backbones/npr.py", "pmsa/models/__init__.py",
    "pmsa/models/fusion.py", "pmsa/models/linear_probe.py", "pmsa/calibration/__init__.py",
    "pmsa/calibration/conformal.py", "pmsa/calibration/np_threshold.py",
    "pmsa/utils/__init__.py", "pmsa/utils/device.py", "pmsa/utils/seed.py",
)
TEMPLATES = ("README.md", "requirements.txt", "space_app.py")


def prepare(repo, bundle_dir, output):
    repo, bundle_dir, output = (Path(p).resolve() for p in (repo,bundle_dir,output))
    if output.exists():
        raise FileExistsError("Preserve existing releases; choose a new --out")
    if not output.is_relative_to(repo / "outputs"):
        raise ValueError("Stage releases under this repository's ignored outputs directory")
    bundle = verify_bundle(bundle_dir)
    validate_stream_metadata(bundle["streams"])
    if bundle["model_type"] != "fusion" or list(bundle["streams"]) != ["siglip","forensic"]:
        raise ValueError("This release path requires the current two-stream fusion bundle")
    from pmsa.models import FusionDetector
    model = FusionDetector.load(bundle_dir/"fusion.pt",device="cpu")
    if [(s.name,s.dim) for s in model.specs] != [("siglip",768),("forensic",66)]:
        raise ValueError("Unexpected serving model dimensions")
    protected = {str(bundle_dir/name):sha256_file(bundle_dir/name) for name in
                 ("fusion.pt","calibrator.json","splits.json","bundle.json")}
    for name in SERVING_FILES:
        path = repo/name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked serving source: {name}")
    for name in TEMPLATES:
        if not (repo/"deployment/space"/name).is_file():
            raise ValueError(f"Missing Space template: {name}")
    output.mkdir(parents=True)
    for name in SERVING_FILES:
        target = output/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/name,target)
    for name in TEMPLATES:
        shutil.copyfile(repo/"deployment/space"/name,output/name)
    if (repo/"LICENSE").is_file() and not (repo/"LICENSE").is_symlink():
        shutil.copyfile(repo/"LICENSE",output/"LICENSE")
    (output/"model").mkdir()
    for name in ("fusion.pt","calibrator.json"):
        shutil.copyfile(bundle_dir/name,output/"model"/name)
    original_splits=json.loads((bundle_dir/"splits.json").read_text())
    partitions=original_splits.get("partitions",{})
    sanitized={"schema_version":1,"scope":"Sanitized public deployment receipt, not a reproducibility split manifest",
        "source_bundle_sha256":protected[str(bundle_dir/"bundle.json")],
        "source_splits_sha256":protected[str(bundle_dir/"splits.json")],
        "seed":original_splits.get("seed"),
        "partitions":{name:{k:int(value[k]) for k in ("n_real","n_fake") if k in value}
                      for name,value in partitions.items()},
        "identities_omitted":True,"note":"Original split IDs, image paths/hashes and predictions stay private."}
    (output/"model/splits.json").write_text(json.dumps(sanitized,indent=2,allow_nan=False)+"\n")
    released=copy.deepcopy(bundle)
    released["files"]={name:sha256_file(output/"model"/name) for name in
                       ("fusion.pt","calibrator.json","splits.json")}
    released["release"]={"source_bundle_sha256":protected[str(bundle_dir/"bundle.json")],
                         "model_and_calibrator_bytes_unchanged":True,"public_split_receipt_sanitized":True}
    (output/"model/bundle.json").write_text(json.dumps(released,indent=2,allow_nan=False)+"\n")
    verify_bundle(output/"model")
    for path,digest in protected.items():
        if sha256_file(path)!=digest:
            raise RuntimeError("Original artifact changed")
    files={str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest={"schema_version":1,"release_id":output.parent.name,"files":files,
        "source_bundle_sha256":protected[str(bundle_dir/"bundle.json")],
        "scope":"Serving files only; no photos, private documents, caches or original split identities",
        "probability_calibrated":False}
    (output/"release_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    result=prepare(Path(__file__).resolve().parents[1],args.bundle,args.out)
    print(f"Prepared {len(result['files'])+1} allowlisted serving files; originals unchanged; no upload.")


if __name__ == "__main__":
    main()
