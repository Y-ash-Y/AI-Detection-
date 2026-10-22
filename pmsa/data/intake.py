"""Private, reviewed image intake and deterministic group-safe role manifests.

No scoring, fitting, downloads or mutation of source images. Folder labels are
curator assertions, never authenticated provenance. Already-inspected identities
remain reserved after renaming/moving or a lossless re-encoding.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..artifacts import sha256_file
from ..calibration.np_threshold import min_calibration_size
from ..photo_evaluation import IMAGE_SUFFIXES
from ..preprocessing import ImagePolicy, load_image
from .manifest import ImageRecord, Manifest
from .splits import group_ids

ROLES = ("train", "validation", "calibration", "test")
DEFAULT_FRACTIONS = (.5, .15, .2, .15)
REVIEW_FIELDS = ("candidate_id", "relative_path", "status", "label", "domain",
                 "source", "origin_kind", "device", "processing", "group_id",
                 "prior_use", "approved", "notes")
ORIGINS = {"camera": "0", "fully_generated": "1", "ai_edit": "1",
           "conventional_graphic": "0"}


def _json_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _new_dir(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve the existing intake/splits; choose a new output directory")
    return path


def reserve_history(real_plan, fake_plan, output):
    """Create an explicit reserved registry from trusted prior local receipts."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Reserved registry exists; do not overwrite")
    real = json.loads(Path(real_plan).read_text())
    fake = json.loads(Path(fake_plan).read_text())
    rows = []
    for label, records in ((0, real["inventory"]["images"]), (1, fake["new_fake_images"])):
        for row in records:
            digest = row["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid reserved image hash")
            rows.append({"sha256": digest, "decoded_rgb_sha256": row.get("decoded_rgb_sha256", ""),
                         "label": label, "reason": "already-inspected-regression",
                         "width": row.get("width"), "height": row.get("height"),
                         "group_id": row.get("group_id", ""), "dhash": row.get("dhash", "")})
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "images": rows,
              "source_receipts_sha256": [sha256_file(real_plan), sha256_file(fake_plan)]}
    _json_new(output, result)
    return result


def _reserved(path):
    registry = json.loads(Path(path).read_text())
    if registry.get("schema_version") != 1 or not isinstance(registry.get("images"), list):
        raise ValueError("Invalid reserved registry")
    by_raw, by_pixels = {}, {}
    for row in registry["images"]:
        if row.get("label") not in (0, 1):
            raise ValueError("Reserved registry labels must be 0 or 1")
        for field, mapping in (("sha256", by_raw), ("decoded_rgb_sha256", by_pixels)):
            value = row.get(field, "")
            if value:
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("Invalid registry digest")
                if value in mapping and mapping[value]["label"] != row["label"]:
                    raise ValueError("Reserved registry has conflicting labels")
                mapping[value] = row
    return registry, by_raw, by_pixels


def _fingerprint(path):
    # Native EXIF-oriented RGB identity, before optional model JPEG normalization.
    image = load_image(path, ImagePolicy())
    decoded = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()).hexdigest()
    small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = small[:, 1:] > small[:, :-1]
    return {"decoded_rgb_sha256": decoded, "width": image.width, "height": image.height,
            "dhash": np.packbits(bits).tobytes().hex()}


def inventory(root, reserved_path, output):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Image root must be a directory")
    output = _new_dir(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Keep private intake outputs outside the source-photo directory")
    registry, reserved_raw, reserved_pixels = _reserved(reserved_path)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError("No supported image files in the declared root")
    rows, reviews = [], []
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Image link escapes the declared root")
        relative = str(path.relative_to(root))
        digest = sha256_file(path)
        candidate = hashlib.sha256((relative + "\n" + digest).encode()).hexdigest()
        parent = "_".join(path.relative_to(root).parts[0].strip().lower().replace("_", " ").split())
        folder_label = "0" if parent == "real" else "1" if parent == "ai_generated" else ""
        fact = {"candidate_id": candidate, "relative_path": relative, "sha256": digest,
                "suffix": path.suffix.lower(), "folder_label": folder_label}
        reserved_row = reserved_raw.get(digest)
        if reserved_row:
            fact.update({k: reserved_row.get(k, "") for k in ("decoded_rgb_sha256", "dhash", "width", "height")})
            fact["status"] = "reserved-regression"
            # In particular, do not retry or change the old HEIC input policy.
        else:
            try:
                fact.update(_fingerprint(path))
                reserved_row = reserved_pixels.get(fact["decoded_rgb_sha256"])
                fact["status"] = "reserved-regression" if reserved_row else "pending-review" if folder_label else "ambiguous-folder"
            except Exception as exc:
                fact["status"] = "decode-error"
                fact["error_type"] = type(exc).__name__
        if reserved_row:
            fact["reserved_label"] = str(reserved_row["label"])
        label = fact.get("reserved_label", folder_label)
        rows.append(fact)
        reviews.append({"candidate_id": candidate, "relative_path": relative, "status": fact["status"],
                        "label": label, "domain": "", "source": "real" if label == "0" else "",
                        "origin_kind": "camera" if label == "0" else "", "device": "",
                        "processing": "", "group_id": "", "prior_use": "previously_used" if reserved_row else "",
                        "approved": "", "notes": ""})
    hints = []
    capped = False
    for i, a in enumerate(rows):
        if not a.get("dhash"):
            continue
        for j in range(i + 1, len(rows)):
            b = rows[j]
            if not b.get("dhash") or a.get("decoded_rgb_sha256") == b.get("decoded_rgb_sha256"):
                continue
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if distance <= 4:
                if len(hints) < 1000:
                    hints.append({"candidates": [a["candidate_id"], b["candidate_id"]], "hamming_distance": distance})
                else:
                    capped = True
    report = {"schema_version": 1, "root": str(root), "reserved_path": str(Path(reserved_path).resolve()),
              "reserved_sha256": sha256_file(reserved_path), "images": rows,
              "counts": dict(Counter(r["status"] for r in rows)),
              "related_review_hints": hints, "hint_list_capped": capped,
              "registry_images": len(registry["images"]),
              "caveats": ["Folder labels are assertions, not independently verified image provenance.",
                          "Exact RGB/byte identities cannot exclude crops, resizes, recompression or related scenes.",
                          "Difference hashes are review hints, not automatic semantic/scene grouping.",
                          "No input was scored, fitted, moved, uploaded or modified."]}
    output.mkdir(parents=True)
    _json_new(output / "inventory.json", report)
    with (output / "review.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(reviews)
    return report


def _allocation(n, fractions):
    if n < len(fractions):
        raise ValueError("Every domain/class stratum needs at least four independent scene/prompt groups")
    target = n * fractions
    counts = np.maximum(1, np.floor(target).astype(int))
    while counts.sum() > n:
        candidates = np.where(counts > 1)[0]
        counts[candidates[np.argmax((counts - target)[candidates])]] -= 1
    while counts.sum() < n:
        counts[np.argmax(target - counts)] += 1
    return counts


def select_development(inventory_path, generator_dirs, output, *, per_generator=50, seed=0,
                       fully_generated=False):
    """Explicit reuse protocol: make an unapproved, score-independent selection.

    This never emits training role manifests or a blind-test set. Generated
    collections must be explicitly designated as previously evaluated development
    data. The fresh-only ``build`` contract and reserved registry are unchanged.
    """
    output = _new_dir(output)
    if type(per_generator) is not int or per_generator < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive sample size and nonnegative integer seed required")
    if type(fully_generated) is not bool:
        raise ValueError("Fully-generated assertion must be explicit boolean")
    if not generator_dirs or any(not s.strip() or s == "real" for s in generator_dirs):
        raise ValueError("Explicit generator sources required")
    audit = json.loads(Path(inventory_path).read_text())
    if audit.get("schema_version") != 1:
        raise ValueError("Unsupported inventory")
    root = Path(audit["root"]).resolve(strict=True)
    if output.resolve().is_relative_to(root):
        raise ValueError("Keep selection outputs outside source images")
    if sha256_file(audit["reserved_path"]) != audit["reserved_sha256"]:
        raise ValueError("Reserved registry changed; prepare a new inventory")
    _, reserved_raw, reserved_pixels = _reserved(audit["reserved_path"])
    folders = {}
    for source, name in generator_dirs.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("Generator folders must be explicit relative subdirectories")
        folder = (root / relative).resolve(strict=True)
        if not folder.is_dir() or not folder.is_relative_to(root):
            raise ValueError("Generator folder escapes the declared root")
        folders[source] = folder
    folder_values = list(folders.values())
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(folder_values) for b in folder_values[i+1:]):
        raise ValueError("Generator directories must not overlap")
    pools = {source: [] for source in folders}
    pools["real"] = []
    excluded, identity_sources = [], {}
    seen_candidates = set()
    for fact in audit["images"]:
        if fact["candidate_id"] in seen_candidates:
            raise ValueError("Duplicate candidate identities")
        seen_candidates.add(fact["candidate_id"])
        relative = Path(fact["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid inventory path")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != fact["sha256"]:
            raise ValueError("Input changed or escapes inventory root")
        sources = [s for s, folder in folders.items() if path.is_relative_to(folder)]
        source = sources[0] if sources else "real" if fact["folder_label"] == "0" else None
        if source is None:
            excluded.append({"candidate_id": fact["candidate_id"], "reason": "outside-designated-generator-folders"})
            continue
        label = "0" if source == "real" else "1"
        if fact["folder_label"] != label or fact.get("reserved_label", label) != label:
            raise ValueError("Declared source and folder/prior label conflict")
        protected = (fact["sha256"] in reserved_raw or fact.get("decoded_rgb_sha256") in reserved_pixels
                     or fact["status"] == "reserved-regression")
        if fact["status"] == "decode-error" or (source == "real" and protected):
            excluded.append({"candidate_id": fact["candidate_id"],
                             "reason": "decode-error" if fact["status"] == "decode-error" else "earlier-real-diagnostic"})
            continue
        current = _fingerprint(path)
        if current["decoded_rgb_sha256"] != fact.get("decoded_rgb_sha256"):
            raise ValueError("Decoded identity changed")
        for digest in (fact["sha256"], current["decoded_rgb_sha256"]):
            if digest in identity_sources and identity_sources[digest] != source:
                raise ValueError("Identical image appears across declared sources/classes")
            identity_sources[digest] = source
        pools[source].append({**fact, **current, "source": source, "label": label})
    selected, pool_counts = [], {}
    for source, rows in sorted(pools.items()):
        # Exact decoded pixels merge byte duplicates and metadata-only re-exports.
        canonical = {}
        for row in sorted(rows, key=lambda r: r["relative_path"]):
            identity = row["decoded_rgb_sha256"]
            if identity in canonical:
                excluded.append({"candidate_id": row["candidate_id"], "reason": "exact-identity-copy"})
            else:
                canonical[identity] = row
        pool_counts[source] = len(canonical)
        if source != "real" and len(canonical) < per_generator:
            raise ValueError(f"Insufficient unique images in {source}: {len(canonical)}")
        def rank(row):
            return hashlib.sha256(f"{seed}:{source}:{row['decoded_rgb_sha256']}".encode()).hexdigest()
        ranked = sorted(canonical.values(), key=rank)
        chosen = ranked if source == "real" else ranked[:per_generator]
        selected.extend(chosen)
        excluded.extend({"candidate_id": r["candidate_id"], "reason": "not-selected-content-ranked-sample"}
                        for r in ranked[len(chosen):])
    if not pools["real"]:
        raise ValueError("No eligible real-photo candidates")
    reviews = []
    for row in selected:
        camera = row["source"] == "real"
        reviews.append({"candidate_id": row["candidate_id"], "relative_path": row["relative_path"],
                        "status": "development-pending-review", "label": row["label"], "domain": "",
                        "source": row["source"], "origin_kind": "camera" if camera else "fully_generated" if fully_generated else "",
                        "device": "", "processing": "", "group_id": "",
                        "prior_use": "" if camera else "previously_used", "approved": "",
                        "notes": "User reports camera origin; editing history and related scenes need review."
                                 if camera else "User confirmed earlier zero-shot use. Generator version and generative-edit status need review."})
    selected_ids = {r["candidate_id"] for r in selected}
    plan = {"schema_version": 1, "protocol": "reused-generator-development-v1",
            "seed": seed, "per_generator": per_generator, "root": str(root),
            "inventory_path": str(Path(inventory_path).resolve()),
            "inventory_sha256": sha256_file(inventory_path), "reserved_sha256": audit["reserved_sha256"],
            "generator_dirs": {s: str(f.relative_to(root)) for s, f in folders.items()},
            "fully_generated_user_assertion": fully_generated,
            "pool_unique_images": pool_counts,
            "selected_counts": dict(Counter(r["source"] for r in selected)),
            "images": selected, "excluded": excluded,
            "related_review_hints": [h for h in audit["related_review_hints"]
                                     if selected_ids.intersection(h["candidates"])],
            "approval_status": "pending-origin-processing-domain-and-related-group-review",
            "role_manifests_created": False, "untouched_test_created": False,
            "caveats": ["Every selected generated image belongs to a previously evaluated development collection.",
                        "Unselected images from those same evaluated collections are not an untouched confirmation test.",
                        "Related-group review is required; exact identity deduplication is not scene independence.",
                        "Sampling uses seed/source/decoded-pixel hashes, never model scores or hand-picked successes.",
                        "Ordinary edited camera photos and AI-edited photos need explicit scope review.",
                        "This is not training-ready and does not bypass the fresh-only build command."]}
    output.mkdir(parents=True)
    _json_new(output / "selection.json", plan)
    with (output / "review.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(reviews)
    return plan


def build(inventory_path, review_path, output, *, seed=0, alpha=.05, fractions=DEFAULT_FRACTIONS):
    return _build_roles(inventory_path, review_path, output, seed=seed, alpha=alpha, fractions=fractions)


def build_development(selection_path, review_path, output, *, seed=0, alpha=.05,
                      fractions=DEFAULT_FRACTIONS):
    """Explicitly reused development roles; the test role is diagnostic only."""
    selection = json.loads(Path(selection_path).read_text())
    if selection.get("protocol") != "reused-generator-development-v1" or selection.get("schema_version") != 1:
        raise ValueError("Explicit reused-development selection required")
    if sha256_file(selection["inventory_path"]) != selection["inventory_sha256"]:
        raise ValueError("Selection inventory changed")
    return _build_roles(selection["inventory_path"], review_path, output, seed=seed,
                        alpha=alpha, fractions=fractions,
                        development={**selection, "selection_sha256": sha256_file(selection_path)})


def _build_roles(inventory_path, review_path, output, *, seed, alpha, fractions, development=None):
    output = _new_dir(output)
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.shape != (4,) or not np.isfinite(fractions).all() or (fractions <= 0).any()
            or not np.isclose(fractions.sum(), 1)):
        raise ValueError("Four positive fractions summing to one required")
    minimum = min_calibration_size(alpha)
    audit = json.loads(Path(inventory_path).read_text())
    if audit.get("schema_version") != 1:
        raise ValueError("Unsupported inventory")
    root = Path(audit["root"]).resolve(strict=True)
    if output.resolve().is_relative_to(root):
        raise ValueError("Output must not be inside the source images")
    if sha256_file(audit["reserved_path"]) != audit["reserved_sha256"]:
        raise ValueError("Reserved registry changed; prepare a new inventory")
    _, reserved_raw, reserved_pixels = _reserved(audit["reserved_path"])
    facts = {r["candidate_id"]: r for r in audit["images"]}
    if len(facts) != len(audit["images"]):
        raise ValueError("Duplicate candidate identities")
    reused_ids = set()
    if development is not None:
        selected = development["images"]
        if len({r["candidate_id"] for r in selected}) != len(selected):
            raise ValueError("Duplicate development selection identity")
        selected_facts = {}
        for item in selected:
            fact = facts.get(item["candidate_id"])
            if (fact is None or any(item.get(k) != fact.get(k) for k in
                                   ("sha256", "decoded_rgb_sha256", "relative_path", "folder_label"))):
                raise ValueError("Development selection does not match inventory")
            if item["source"] != "real":
                folder = Path(development["generator_dirs"][item["source"]])
                if (folder.is_absolute() or ".." in folder.parts or folder == Path(".")
                        or not Path(fact["relative_path"]).is_relative_to(folder)
                        or fact["folder_label"] != "1" or fact.get("reserved_label", "1") != "1"):
                    raise ValueError("Invalid designated generator selection")
                reused_ids.add(item["candidate_id"])
            elif fact["folder_label"] != "0":
                raise ValueError("Invalid selected real label")
            selected_facts[item["candidate_id"]] = fact
        facts = selected_facts
    with Path(review_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(REVIEW_FIELDS) <= set(reader.fieldnames or []):
            raise ValueError("Incomplete review columns")
        annotations = [{k: str(v or "").strip() for k, v in row.items()} for row in reader]
    if len(annotations) != len(facts) or {r["candidate_id"] for r in annotations} != set(facts):
        raise ValueError("Review must contain every inventory candidate exactly once")
    reviewed, excluded = [], []
    reserved_related = {r.get("group_id", "") for r in _reserved(audit["reserved_path"])[0]["images"]
                        if development is None or r["label"] == 0} - {""}
    reserved_related.update(row["group_id"] for row in annotations
                            if row["group_id"] and facts[row["candidate_id"]]["status"] == "reserved-regression"
                            and row["candidate_id"] not in reused_ids)
    for row in annotations:
        fact = facts[row["candidate_id"]]
        relative = Path(fact["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or row["relative_path"] != fact["relative_path"]:
            raise ValueError("Review identity/path changed")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != fact["sha256"]:
            raise ValueError("Input changed or escapes inventory root")
        protected = (fact["sha256"] in reserved_raw or fact.get("decoded_rgb_sha256") in reserved_pixels
                     or fact["status"] == "reserved-regression")
        if (protected and row["candidate_id"] not in reused_ids) or fact["status"] == "decode-error":
            excluded.append({"candidate_id": row["candidate_id"], "reason": "reserved-regression" if protected else "decode-error"})
            continue  # CSV edits cannot override reserved exclusions.
        if row["approved"] != "yes":
            raise ValueError("Unapproved fresh image; review labels, related groups and prior use before splitting")
        if row["label"] not in ("0", "1") or (fact["folder_label"] and row["label"] != fact["folder_label"]):
            raise ValueError("Invalid label or disagreement with the declared real/ai_generated folder")
        if row["origin_kind"] not in ORIGINS or ORIGINS[row["origin_kind"]] != row["label"]:
            raise ValueError("Origin kind and label must agree; disclose generative edits separately")
        if row["prior_use"] not in ("new", "previously_used", "related_to_prior"):
            raise ValueError("Prior-use review is required")
        if row["candidate_id"] in reused_ids and row["prior_use"] != "previously_used":
            raise ValueError("Reused generated images must retain previously_used history")
        if development is not None:
            expected_source = next(r["source"] for r in development["images"] if r["candidate_id"] == row["candidate_id"])
            if row["source"] != expected_source:
                raise ValueError("Reviewed source changed from development selection")
        for field in ("domain", "source", "processing", "group_id"):
            if not row[field] or row[field].lower() in ("unknown", "pending", "?"):
                raise ValueError(f"Reviewed {field} required")
        if row["label"] == "0" and row["source"] != "real":
            raise ValueError("Camera source must be real; put device details in the device field")
        if row["label"] == "1" and row["source"] == "real":
            raise ValueError("Generated images need their asserted generator source")
        # Re-decode fresh inputs to prevent edited inventory pixel identities.
        current = _fingerprint(path)
        if current["decoded_rgb_sha256"] != fact.get("decoded_rgb_sha256"):
            raise ValueError("Decoded input identity changed")
        if row["group_id"] in reserved_related:
            row["prior_use"] = "related_to_prior"
        reviewed.append({**row, "path": str(path), "sha256": fact["sha256"], **current})
    if not reviewed:
        raise ValueError("No fresh reviewed images; reserved files cannot be development data")
    raw_groups = group_ids([r["path"] for r in reviewed], [r["sha256"] for r in reviewed],
                          [r["group_id"] for r in reviewed])
    groups = group_ids([r["path"] for r in reviewed], [r["decoded_rgb_sha256"] for r in reviewed], raw_groups)
    components = defaultdict(list)
    for row, group in zip(reviewed, groups):
        components[str(group)].append(row)
    strata, challenges, records = defaultdict(list), [], {}
    for group, members in components.items():
        if any(r["prior_use"] != "new" and not
               (r["candidate_id"] in reused_ids and r["prior_use"] == "previously_used") for r in members):
            excluded.extend({"candidate_id": r["candidate_id"], "reason": "component-related-to-prior-use"} for r in members)
            continue
        if any(r["origin_kind"] in ("ai_edit", "conventional_graphic") for r in members):
            challenges.extend(ImageRecord(r["path"], int(r["label"]), r["domain"], r["source"], group) for r in members)
            continue  # Original/edit pairs stay together outside main photographic binary protocol.
        if len({r["label"] for r in members}) != 1:
            raise ValueError("Mixed-label related component: disclose original/edit relationships as a separate challenge")
        if len({r["domain"] for r in members}) != 1:
            raise ValueError("A related scene/prompt component has conflicting domain annotations")
        key = (members[0]["domain"], members[0]["label"])
        if development is not None:
            if len({r["source"] for r in members}) != 1:
                raise ValueError("Related component crosses designated generator sources")
            key = (*key, members[0]["source"])
        strata[key].append(group)
        records[group] = [ImageRecord(r["path"], int(r["label"]), r["domain"], r["source"], group) for r in members]
    if not strata:
        raise ValueError("No fresh primary-scope groups after exclusions")
    domain_classes = {(key[0], key[1]) for key in strata}
    domains = {key[0] for key in strata}
    if any((domain, "0") not in domain_classes or (domain, "1") not in domain_classes for domain in domains):
        raise ValueError("Every primary content domain must contain real AND generated examples; match coverage before splitting")
    partitions = {role: [] for role in ROLES}
    rng = np.random.default_rng(seed)
    for key in sorted(strata):
        shuffled = rng.permutation(sorted(strata[key]))
        cuts = np.cumsum(_allocation(len(shuffled), fractions))[:-1]
        for role, chunk in zip(ROLES, np.split(shuffled, cuts)):
            for group in chunk:
                partitions[role].extend(records[str(group)])
    group_sets = [{r.group_id for r in partitions[role]} for role in ROLES]
    if any(a & b for i, a in enumerate(group_sets) for b in group_sets[i+1:]):
        raise AssertionError("Cross-role image-group overlap")
    real_cal_groups = {r.group_id for r in partitions["calibration"] if r.label == 0}
    if len(real_cal_groups) < minimum:
        raise ValueError(f"Need at least {minimum} independent real calibration groups at alpha={alpha:g}; have {len(real_cal_groups)}")
    summary = {role: {"rows": len(items), "independent_groups": len({r.group_id for r in items}),
                       "real_groups": len({r.group_id for r in items if r.label == 0}),
                       "fake_groups": len({r.group_id for r in items if r.label == 1}),
                       "domain_label": dict(Counter(f"{r.domain}:{r.label}" for r in items)),
                       "sources": dict(Counter(r.source for r in items)),
                       "formats_by_label": dict(Counter(f"{r.label}:{Path(r.path).suffix.lower()}" for r in items))}
               for role, items in partitions.items()}
    plan = {"schema_version": 1, "seed": seed, "roles": list(ROLES), "fractions": fractions.tolist(),
            "inventory_sha256": sha256_file(inventory_path), "review_sha256": sha256_file(review_path),
            "reserved_sha256": audit["reserved_sha256"], "summary": summary,
            "excluded": excluded, "challenge_rows": len(challenges),
            "real_calibration_groups": len(real_cal_groups), "alpha": alpha,
            "minimum_for_finite_conformal_threshold": minimum, "cross_role_group_overlap": False,
            "selection": "Fixed before scoring; domain/class-stratified related-group assignment",
            "near_duplicate_review": "Curator-approved scene/prompt groups; difference hashes are hints only",
            "caveats": ["Finite calibration size is not proof of exchangeability or domain-specific FPR control.",
                        "Small test strata and correlated scenes cannot establish deployment readiness.",
                        "Group-safe manifests do not remove semantic, device, format or generator-source shortcuts.",
                        "Only real calibration representatives may set thresholds; validation selects settings; test is untouched.",
                        "AI edits/conventional graphics are challenge-only; mixed-label challenges need a separate grouped evaluation."]}
    if development is not None:
        plan.update(protocol="reused-generator-development-v1",
                    selection_sha256=development["selection_sha256"],
                    reused_generated_candidates=len(reused_ids),
                    untouched_confirmation_test=False,
                    test_role="internal-diagnostic-not-blind-confirmation",
                    selection="Fixed before scoring; domain/class/source-stratified curator-group assignment")
        plan["caveats"].extend([
            "Generated images were previously evaluated; test-role power is an internal diagnostic, not fresh generalization evidence.",
            "Domain can be one broad user collection; no matched-semantic-content claim follows from that annotation.",
            "Scene/prompt independence and origins are curator assertions, not independently authenticated facts."])
    output.mkdir(parents=True)
    for role, items in partitions.items():
        Manifest(items).to_csv(output / f"manifest_{role}.csv")
    Manifest(challenges).to_csv(output / "manifest_challenge.csv")
    plan["manifest_sha256"] = {p.name: sha256_file(p) for p in sorted(output.glob("manifest_*.csv"))}
    _json_new(output / "split_plan.json", plan)
    return plan
