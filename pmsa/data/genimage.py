"""Scan GenImage-style directories into a Manifest.

Expected GenImage layout (per generator):

    <root>/<generator>/<split>/ai/*.png       -> fake, source=<generator>
    <root>/<generator>/<split>/nature/*.JPEG  -> real, source="real"

All `nature` images come from ImageNet regardless of generator, so reals share
domain="imagenet". This is exactly the setting where the REAL domain is fixed and
only the generator varies — the (B) column of the shift matrix.

For real-domain-shift cells, scan a different real source with scan_real_dir(...,
domain="ffhq" / "chameleon"), and a separate in-the-wild fake set with
scan_fake_dir(...).
"""
from __future__ import annotations

from pathlib import Path

from .manifest import ImageRecord, Manifest

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _images(d: Path):
    return [p for p in d.rglob("*") if p.suffix.lower() in _IMG_EXT]


def scan_genimage(root: str | Path, generators=None, split="train",
                  per_class_limit: int | None = None,
                  real_domain="imagenet") -> Manifest:
    root = Path(root)
    if generators is None:
        generators = [p.name for p in root.iterdir() if p.is_dir()]
    recs: list[ImageRecord] = []
    for gen in generators:
        ai = root / gen / split / "ai"
        nature = root / gen / split / "nature"
        for p in _limit(_images(ai), per_class_limit):
            recs.append(ImageRecord(str(p), 1, real_domain, gen))
        for p in _limit(_images(nature), per_class_limit):
            recs.append(ImageRecord(str(p), 0, real_domain, "real"))
    return Manifest(recs)


def scan_real_dir(directory: str | Path, domain: str,
                  limit: int | None = None) -> Manifest:
    """A pure-real directory from some domain (FFHQ, Chameleon reals, your own)."""
    recs = [ImageRecord(str(p), 0, domain, "real")
            for p in _limit(_images(Path(directory)), limit)]
    return Manifest(recs)


def scan_fake_dir(directory: str | Path, source: str, domain="wild",
                  limit: int | None = None) -> Manifest:
    """A pure-fake directory from one generator (frontier / in-the-wild set)."""
    recs = [ImageRecord(str(p), 1, domain, source)
            for p in _limit(_images(Path(directory)), limit)]
    return Manifest(recs)


def _limit(items, n):
    return items if n is None else items[:n]
