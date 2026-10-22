"""Standard-library source handoff checks, also embedded in the Kaggle notebook.

Hashes provide integrity, not publisher authenticity. Use only trusted sources.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def safe_source_name(name):
    if (not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(p in {"", ".", ".."} for p in name.split("/"))):
        raise ValueError("Unsafe source filename")
    return name


def load_source_snapshot(source_input):
    source_input = Path(source_input)
    archive_handle = None
    if source_input.is_dir():
        candidates = [p for p in (source_input, source_input / "source")
                      if (p / "source_manifest.json").is_file() and (p / "pyproject.toml").is_file()]
        if len(candidates) > 1:
            raise ValueError("Ambiguous source folders; choose the exact project folder")
        if not candidates:
            archive = source_input / "pmsa-kaggle-source.zip"
            if archive.is_file():
                return load_source_snapshot(archive)
            raise FileNotFoundError("Updated source manifest missing; upload the updated handoff package")
        root = candidates[0].resolve()

        def reader(name):
            name = safe_source_name(name)
            path = root / name
            if (not path.resolve().is_relative_to(root)
                    or any((root / Path(*Path(name).parts[:i])).is_symlink()
                           for i in range(1, len(Path(name).parts)+1))):
                raise ValueError("Unsafe source symlink")
            if not path.is_file() or path.stat().st_size > 50_000_000:
                raise ValueError("Missing or oversized source file")
            return path.read_bytes()

        manifest = json.loads(reader("source_manifest.json"))
    else:
        archive_handle = zipfile.ZipFile(source_input)
        try:
            entries = archive_handle.infolist()
            if len(entries) > 2000 or sum(p.file_size for p in entries) > 50_000_000:
                raise ValueError("Oversized source archive; use the small source handoff")
            if len({p.filename for p in entries}) != len(entries):
                raise ValueError("Duplicate archive members")
            for entry in entries:
                safe_source_name(entry.filename.rstrip("/"))
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Unsafe archive symlink")
            try:
                manifest = json.loads(archive_handle.read("source/source_manifest.json"))
            except KeyError as exc:
                raise ValueError("Updated source manifest missing; replace the old ZIP") from exc
            reader = lambda name: archive_handle.read("source/"+safe_source_name(name))
        except Exception:
            archive_handle.close()
            raise
    try:
        files = manifest.get("files")
        if (manifest.get("schema_version") != 1 or not isinstance(files, dict)
                or not 0 < len(files) <= 1000 or "pyproject.toml" not in files):
            raise ValueError("Invalid source manifest")
        payload, total = {}, 0
        for name, digest in sorted(files.items()):
            safe_source_name(name)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Invalid source digest")
            data = reader(name)
            total += len(data)
            if total > 50_000_000:
                raise ValueError("Oversized source snapshot")
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Source integrity mismatch: {name}")
            payload[name] = data
        identity = hashlib.sha256(json.dumps(files, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        return payload, identity
    finally:
        if archive_handle is not None:
            archive_handle.close()


def prepare_source_snapshot(source_input, workspace):
    payload, digest = load_source_snapshot(source_input)
    workspace = Path(workspace)
    root, marker = workspace / "source", workspace / "source-snapshot-sha256.txt"
    if workspace.exists():
        if not marker.is_file() or marker.read_text().strip() != digest:
            raise FileExistsError("Source workspace differs. Choose a NEW SOURCE_ROOT; do not delete old work.")
        for name, data in payload.items():
            target = root / name
            if (not target.resolve().is_relative_to(root.resolve()) or not target.is_file()
                    or target.is_symlink() or target.read_bytes() != data):
                raise ValueError(f"Existing source workspace changed: {name}; choose a NEW SOURCE_ROOT")
    else:
        workspace.mkdir(parents=True)
        for name, data in payload.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        marker.write_text(digest+"\n")
    return root, digest
