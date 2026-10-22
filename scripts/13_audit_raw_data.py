#!/usr/bin/env python
"""Decode/hash manifests, reject cross-role identity overlap, and retain group IDs."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.artifacts import sha256_file
from pmsa.data.manifest import Manifest
from pmsa.data.raw_audit import audit_manifests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", nargs="+", required=True, help="ROLE:CSV, e.g. train:path.csv test:path.csv")
    ap.add_argument("--out", required=True, help="New audit directory; contains curated role CSVs")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Audit output exists; choose a NEW directory")
    manifests, files = {}, {}
    for spec in args.manifest:
        role, filename = spec.split(":", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", role) or role in manifests:
            raise ValueError("Manifest roles must be unique safe names")
        manifests[role] = Manifest.from_csv(filename)
        files[role] = {"path": filename, "sha256": sha256_file(filename)}
    report, curated = audit_manifests(manifests)
    report["input_manifests"] = files
    out.mkdir(parents=True)
    for role, manifest in curated.items():
        manifest.to_csv(out / f"{role}.csv")
    (out / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["summary"], indent=2))
    print("Audit passed; inspect domain/class counts and near-duplicates before full extraction.")


if __name__ == "__main__":
    main()
