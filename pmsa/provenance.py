"""Unverified metadata keyword hints, NOT a C2PA or SynthID verifier.

Markers can be spoofed; content credentials can occur on camera photographs.
Absence proves nothing. A byte scan cannot detect an invisible SynthID watermark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# marker -> human-readable source
_MARKERS = {
    b"c2pa": "C2PA content credentials",
    b"jumbf": "C2PA/JUMBF manifest",
    b"contentcredentials": "C2PA content credentials",
    b"synthid": "SynthID text reference (watermark not verified)",
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
    verified: bool = False

    def as_dict(self) -> dict:
        return {"found": self.found, "signals": self.signals, "note": self.note,
                "verified": self.verified}


def check_provenance(path: str | Path, scan_bytes: int = 2_000_000) -> Provenance:
    """Scan bounded file bytes/metadata for hints; never verify origin."""
    if scan_bytes < 1:
        raise ValueError("scan_bytes must be positive")
    path = Path(path)
    hits: set[str] = set()
    try:
        with path.open("rb") as f:
            raw = f.read(scan_bytes).lower()
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
            note="Unverified metadata keywords found. They may be spoofed and do not establish AI origin.",
        )
    return Provenance(
        found=False, signals=[],
        note="No recognized metadata hints. Absence does not establish real or AI origin.",
    )
