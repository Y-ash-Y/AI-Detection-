"""Resolve a Hub revision before loading weights and preprocessing assets.

Do not rely on Transformers' private config._commit_hash: some vision-only
configs do not retain it. The Hub's versioned snapshot path identifies the
immutable commit; keep the path lexical because its file may symlink to blobs/.
"""
from pathlib import Path
import re


def resolve_hub_revision(weights: str, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=weights, filename="config.json",
                                revision=revision or "main"))
    commit = path.parent.name
    if (path.name != "config.json" or path.parent.parent.name != "snapshots"
            or not re.fullmatch(r"[0-9a-f]{40}", commit) or not path.is_file()):
        raise ValueError("Cannot establish an immutable Hub snapshot; extraction stopped")
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision) and commit != revision:
        raise ValueError("Downloaded snapshot does not match the requested commit")
    return commit
