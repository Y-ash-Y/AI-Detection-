"""Private, fixed-threshold real-photo diagnostics; never fit on these test images."""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import sha256_file
from .eval.metrics import binomial_rate
from .preprocessing import ImagePolicy, load_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"}


def inventory_photos(directory, *, record_decode_failures=False):
    """Decode all eligible files, grouping ONLY exact decoded RGB duplicates.

    The 64-bit difference hash is a review hint, not a provenance/independence
    certificate. No EXIF/GPS metadata or pixels are exported.
    """
    root = Path(directory).resolve(strict=True)
    paths = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError("No supported photos in the supplied directory")
    rows, representatives, seen = [], [], {}
    for path in paths:
        # Reject escaped links rather than reading unrelated personal files.
        if not path.resolve().is_relative_to(root):
            raise ValueError("Photo link escapes the requested directory")
        try:
            image = load_image(path, ImagePolicy())
        except Exception as exc:
            if not record_decode_failures:
                raise
            rows.append({"path": str(path), "sha256": sha256_file(path),
                         "suffix": path.suffix.lower(), "decode_ok": False,
                         "error_type": type(exc).__name__,
                         "reason": "Multi-frame images are not supported" if str(exc) == "Multi-frame images are not supported" else "Image decode/policy failed; inspect locally"})
            continue
        decoded = hashlib.sha256(
            f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()).hexdigest()
        small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        bits = small[:, 1:] > small[:, :-1]
        dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
        index = len(rows)
        if decoded not in seen:
            seen[decoded] = index
            representatives.append(index)
        rows.append({"path": str(path), "sha256": sha256_file(path), "decode_ok": True,
                     "decoded_rgb_sha256": decoded, "width": image.width,
                     "height": image.height, "suffix": path.suffix.lower(),
                     "dhash": f"{dhash:016x}", "representative": seen[decoded]})
    hints = []
    for i in representatives:
        for j in representatives:
            if j <= i:
                continue
            distance = (int(rows[i]["dhash"], 16) ^ int(rows[j]["dhash"], 16)).bit_count()
            if distance <= 4:
                hints.append({"indices": [i, j], "hamming_distance": distance})
    decoded_count = sum(r["decode_ok"] for r in rows)
    if not decoded_count:
        raise ValueError("No decodable photos; no operating-point estimate possible")
    return {"images": rows, "representative_indices": representatives,
            "file_count": len(rows), "unique_decoded_count": len(representatives),
            "decoded_file_count": decoded_count, "decode_failures": len(rows) - decoded_count,
            "exact_decoded_duplicates": decoded_count - len(representatives),
            "formats": dict(Counter(r["suffix"] for r in rows)),
            "related_image_review_hints": hints,
            "independence_verified": False}


def summarize_real_scores(scores, tau, uncertain_margin=.25):
    """Count false alarms using strict score>tau; abstention is not a TN."""
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or not np.isfinite([tau, uncertain_margin]).all() or uncertain_margin < 0):
        raise ValueError("Nonempty finite scores and finite frozen threshold required")
    false_alarms = int((scores > tau).sum())
    uncertain = np.abs(scores - tau) <= uncertain_margin
    return {"n_real": len(scores), "false_alarms": false_alarms,
            "fpr": binomial_rate(false_alarms, len(scores)),
            "tau": float(tau), "uncertain_margin": float(uncertain_margin),
            "ui_uncertain": int(uncertain.sum()),
            "ui_ai_signal": int(((scores > tau) & ~uncertain).sum()),
            "ui_no_ai_signal": int(((scores <= tau) & ~uncertain).sum()),
            "scope": "Real-only diagnostic. No TPR/AUC, probability, or deployment guarantee.",
            "interval_note": "95% Wilson interval conditional on fixed threshold and independent photos; related shots can invalidate this assumption."}
