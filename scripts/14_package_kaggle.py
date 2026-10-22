#!/usr/bin/env python
"""Private Kaggle handoff archive from an explicit source/artifact allowlist."""
import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.artifacts import sha256_file


def package(repo, output, run=None):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    checksum = output.with_suffix(output.suffix+".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("Archive/checksum already exists; choose a NEW output")
    files = []
    for folder in ("pmsa", "scripts", "examples", "configs", "tests", ".github"):
        for path in sorted((repo / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml") and not path.is_symlink():
                files.append((path, "source/"+str(path.relative_to(repo))))
    for filename in ("app.py", "kaggle_bootstrap.py", "README.md", "pyproject.toml", "requirements-app.txt", "requirements.txt",
                     "docs/MODEL_CARD.md", "docs/REPRODUCIBILITY.md", "docs/FRESH_EVALUATION.md", "docs/SERVING_VALIDATION.md",
                     "docs/ARCHITECTURE.md", "docs/ENGINEERING_STATUS.md", "docs/DEPLOYMENT.md", "docs/evidence/fresh_full_summary.json",
                     "docs/DATA_INTAKE.md", "configs/requirements-photo-eval.txt",
                     "notebooks/kaggle_fresh_evaluation.ipynb", "notebooks/kaggle_resume_pinned_smoke.py",
                     "notebooks/kaggle_serving_parity.py"):
        path = repo / filename
        if path.is_file() and not path.is_symlink():
            files.append((path, "source/"+filename))
    if run is not None:
        run = Path(run).resolve()
        if not run.is_dir() or not run.is_relative_to(repo / "outputs"):
            raise ValueError("--run must be an existing experiment directory under repo/outputs")
        for path in sorted(run.rglob("*")):
            if path.is_symlink():
                raise ValueError("Refusing experiment archive with symlinks")
            if path.is_file() and path.suffix in (".npz", ".json", ".csv", ".yaml", ".yml", ".txt", ".md", ".pt", ".pkl"):
                files.append((path, "artifacts/"+str(path.relative_to(run))))
    if not files:
        raise ValueError("No allowlisted files to archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent or accidental overwrites.
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        # Same manifest verifies both the ZIP and Kaggle's unpacked directory.
        source_hashes = {name.removeprefix("source/"): sha256_file(path)
                         for path, name in files if name.startswith("source/")}
        archive.writestr("source/source_manifest.json", json.dumps(
            {"schema_version": 1, "files": source_hashes}, sort_keys=True, indent=2))
    checksum.write_text(sha256_file(output)+"  "+output.name+"\n")
    return len(files)+1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--run", help="Optional experiment subdirectory under outputs to preserve")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    count = package(args.repo, args.out, args.run)
    print(f"Archived {count} allowlisted files -> {args.out}")
    print("Includes checksum. Excludes raw photos, .git, virtualenvs, private interview books and unselected caches.")


if __name__ == "__main__":
    main()
