"""Bounded raw-image audit before feature extraction; no model downloads.

Exact encoded/decoded identities are not a perceptual near-duplicate detector.
Caller-supplied related IDs supplement these checks and must be curated.
"""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from ..artifacts import sha256_file
from ..preprocessing import ImagePolicy, load_image
from .manifest import Manifest, ImageRecord
from .splits import group_ids, independent_indices


def audit_manifests(manifests):
    if not manifests or any(not len(m) for m in manifests.values()):
        raise ValueError("Every input manifest must be nonempty")
    rows = [(role, record) for role, manifest in manifests.items() for record in manifest.records]
    if any(r.label not in (0, 1) for _, r in rows):
        raise ValueError("Labels must be 0=real or 1=fake")
    # Resolve and check all files before starting an expensive image/hash pass.
    paths = [str(Path(r.path).resolve(strict=True)) for _, r in rows]
    if any(not Path(p).is_file() for p in paths):
        raise ValueError("Manifest paths must be regular image files")
    facts, cache = [], {}
    for (role, r), path in zip(rows, paths):
        if path not in cache:
            try:
                image = load_image(path, ImagePolicy())
                pixel_hash = hashlib.sha256(
                    f"RGB:{image.width}:{image.height}:".encode()+image.tobytes()).hexdigest()
                cache[path] = {"sha256": sha256_file(path), "decoded_rgb_sha256": pixel_hash,
                               "width": image.width, "height": image.height,
                               "suffix": Path(path).suffix.lower()}
            except Exception as exc:
                raise ValueError(f"Cannot decode audit image: {path}") from exc
        facts.append({"role": role, "path": path, "label": r.label,
                      "domain": r.domain, "source": r.source, **cache[path]})
    labels = np.array([r.label for _, r in rows])
    related = [r.group_id for _, r in rows]
    raw_groups = group_ids(paths, [f["sha256"] for f in facts], related)
    groups = group_ids(paths, [f["decoded_rgb_sha256"] for f in facts], raw_groups)
    independent_indices(labels, groups)  # Reject conflicting labels before extraction.
    seen_roles = {}
    for (role, _), group in zip(rows, groups):
        seen_roles.setdefault(group, set()).add(role)
    if any(len(roles) > 1 for roles in seen_roles.values()):
        raise ValueError("Image identity/related group overlaps across dataset roles")
    curated = {}
    summary = {}
    for role in manifests:
        indices = [i for i, (name, _) in enumerate(rows) if name == role]
        curated[role] = Manifest([ImageRecord(paths[i], rows[i][1].label,
            rows[i][1].domain, rows[i][1].source, str(groups[i])) for i in indices])
        summary[role] = {"rows": len(indices), "independent_groups": len(set(groups[indices])),
            "labels": dict(Counter(str(labels[i]) for i in indices)),
            "domain_label": dict(Counter(f"{rows[i][1].domain}:{labels[i]}" for i in indices)),
            "source": dict(Counter(rows[i][1].source for i in indices)),
            "formats": dict(Counter(facts[i]["suffix"] for i in indices))}
    return {"schema_version": 1, "summary": summary, "images": facts,
        "cross_role_overlap": False, "caveats": [
            "Raw byte and exact decoded RGB identities do not exclude resized, recompressed or near-duplicate images.",
            "Related-image identifiers and source/generator labels are curator assertions, not verified provenance.",
            "Global class balance does not remove domain, semantic or compression shortcuts."]}, curated
