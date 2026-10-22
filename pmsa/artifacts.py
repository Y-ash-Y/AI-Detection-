"""Integrity binding, NOT cryptographic provenance. Only use trusted model bundles."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_bundle(directory, model_file, model_type, streams):
    from importlib.metadata import version, PackageNotFoundError
    import platform
    directory = Path(directory)
    env = {"python": platform.python_version()}
    for p in ("numpy", "scikit-learn", "joblib", "torch", "torchvision", "transformers", "Pillow"):
        try:
            env[p] = version(p)
        except PackageNotFoundError:
            pass
    payload = {"schema_version": 1, "model_file": model_file, "model_type": model_type,
               "streams": streams, "environment": env, "probability_calibrated": False,
               "files": {n: sha256_file(directory / n)
                         for n in (model_file, "calibrator.json", "splits.json")}}
    (directory / "bundle.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    return payload


def verify_bundle(directory, checkpoint=None, calibrator=None):
    directory = Path(directory)
    if not (directory / "bundle.json").is_file():
        raise ValueError("Missing bundle.json: legacy artifacts are not automatically served. Retrain/export v3.")
    try:
        bundle = json.loads((directory / "bundle.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read bundle.json; use a complete trusted bundle") from exc
    if (not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int
            or bundle.get("schema_version") != 1
            or bundle.get("model_type") not in ("probe", "fusion")):
        raise ValueError("Unsupported model bundle")
    expected_model = "probe.pkl" if bundle["model_type"] == "probe" else "fusion.pt"
    if bundle.get("model_file") != expected_model:
        raise ValueError("Model type and filename disagree")
    files = bundle.get("files")
    if not isinstance(files, dict) or not {expected_model, "calibrator.json", "splits.json"} <= files.keys():
        raise ValueError("Incomplete artifact hashes")
    if not isinstance(bundle.get("streams"), dict) or not bundle["streams"]:
        raise ValueError("Bundle needs a nonempty stream configuration")
    for name, digest in files.items():
        if (not isinstance(name, str) or Path(name).name != name
                or name in (".", "..") or "\\" in name):
            raise ValueError("Unsafe artifact filename")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"Invalid SHA256 for artifact: {name}")
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing bundled artifact: {name}")
        if path.resolve().parent != directory.resolve():
            raise ValueError(f"Artifact escapes bundle directory: {name}")
        if sha256_file(path) != digest:
            raise ValueError(f"Artifact integrity mismatch: {name}")
    for supplied, name in ((checkpoint, expected_model), (calibrator, "calibrator.json")):
        if supplied is not None and Path(supplied).resolve() != (directory / name).resolve():
            raise ValueError("Selected artifact does not belong to bundle")
    return bundle


def validate_stream_metadata(streams):
    """Require reproducible preprocessing, not merely a cache-schema label."""
    from .preprocessing import ImagePolicy
    if not isinstance(streams, dict) or not streams:
        raise ValueError("Bundle needs at least one stream")
    policy = None
    for name, metadata in streams.items():
        if not isinstance(name, str) or not name or not isinstance(metadata, dict):
            raise ValueError("Invalid stream metadata")
        if metadata.get("schema_version") != 1:
            raise ValueError(f"Missing extraction metadata for {name}")
        image_policy = metadata.get("image_policy")
        if not isinstance(image_policy, dict):
            raise ValueError(f"Missing image policy for {name}")
        try:
            current = ImagePolicy(**image_policy)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid image policy for {name}") from exc
        if policy is not None and current != policy:
            raise ValueError("Feature streams use different image policies")
        policy = current
        packages = metadata.get("packages")
        if (not isinstance(metadata.get("backbone_kwargs"), dict)
                or not isinstance(packages, dict) or not packages
                or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                       for k, v in packages.items())):
            raise ValueError(f"Incomplete extraction configuration for {name}")
        if name in ("siglip", "clip_l14", "dino_b14"):
            kw = metadata["backbone_kwargs"]
            if (not isinstance(kw.get("weights"), str) or not kw["weights"]
                    or not isinstance(kw.get("revision"), str) or not kw["revision"]
                    or type(kw.get("use_fast")) is not bool):
                raise ValueError(f"Unpinned backbone/processor for {name}")
    return policy


def check_runtime(streams):
    """A different extractor version can change features; fail with clear advice."""
    from importlib.metadata import version, PackageNotFoundError
    problems = []
    for metadata in streams.values():
        for package, expected in metadata["packages"].items():
            try:
                actual = version(package)
            except PackageNotFoundError:
                actual = "not installed"
            # CPU/CUDA wheel build suffixes differ while the core extractor version
            # can match. This is compatibility, not a bitwise-determinism claim.
            comparable_actual = actual.split("+", 1)[0] if package in ("torch", "torchvision") else actual
            comparable_expected = expected.split("+", 1)[0] if package in ("torch", "torchvision") else expected
            if comparable_actual != comparable_expected:
                problems.append(f"{package}: expected {expected}, found {actual}")
    if problems:
        raise ValueError("Extraction runtime mismatch. Use the recorded environment or re-extract: "
                         + "; ".join(sorted(set(problems))))
