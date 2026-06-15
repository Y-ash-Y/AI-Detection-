"""Layer 1 — provenance / watermark check (best-effort, dependency-light).

The most reliable real-world signal that an image came from a top generator is the
provenance metadata those vendors embed:
  - OpenAI (GPT-Image / DALL-E): C2PA content credentials.
  - Google (Nano Banana / Gemini / Imagen): SynthID watermark + C2PA.
  - Adobe Firefly, others: C2PA.

When that metadata is present, it's near-certain evidence. When it's absent the
result is INCONCLUSIVE (it may have been stripped by a screenshot or re-save), and
we fall back to the forensic model (Layer 2). This module does a lightweight byte +
EXIF/XMP scan — it does not cryptographically verify C2PA manifests (that needs the
`c2pa` native lib), so treat hits as strong hints, not proofs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# marker -> human-readable source
_MARKERS = {
    b"c2pa": "C2PA content credentials",
    b"jumbf": "C2PA/JUMBF manifest",
    b"contentcredentials": "C2PA content credentials",
    b"synthid": "Google SynthID watermark",
    b"made with google ai": "Google AI",
    b"imagen": "Google Imagen",
    b"gemini": "Google Gemini",
    b"dall-e": "OpenAI DALL-E",
    b"dalle": "OpenAI DALL-E",
    b"gpt-image": "OpenAI GPT-Image",
    b"openai": "OpenAI",
    b"midjourney": "Midjourney",
    b"stable diffusion": "Stable Diffusion",
    b"stability.ai": "Stability AI",
    b"firefly": "Adobe Firefly",
    b"adobe firefly": "Adobe Firefly",
}


@dataclass
class Provenance:
    found: bool
    signals: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"found": self.found, "signals": self.signals, "note": self.note}


def check_provenance(path: str | Path, scan_bytes: int = 2_000_000) -> Provenance:
    """Scan a file for embedded AI-provenance markers (C2PA / SynthID / vendor tags)."""
    path = Path(path)
    hits: set[str] = set()
    try:
        raw = path.read_bytes()[:scan_bytes].lower()
        for marker, label in _MARKERS.items():
            if marker in raw:
                hits.add(label)
    except OSError:
        pass

    # EXIF/XMP via PIL (Software/Description fields sometimes name the generator)
    try:
        from PIL import Image

        with Image.open(path) as im:
            meta = " ".join(str(v) for v in (im.info or {}).values()).lower()
            exif = im.getexif()
            meta += " " + " ".join(str(v) for v in exif.values()).lower()
            for marker, label in _MARKERS.items():
                if marker.decode() in meta:
                    hits.add(label)
    except Exception:
        pass

    if hits:
        return Provenance(
            found=True, signals=sorted(hits),
            note="Embedded AI-provenance metadata found — strong evidence of AI origin.",
        )
    return Provenance(
        found=False, signals=[],
        note="No provenance metadata (may be stripped) — deferring to forensic model.",
    )
