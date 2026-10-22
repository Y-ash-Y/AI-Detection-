"""Image manifest + splitting. The manifest is the single source of truth that
both the extraction pass and every experiment read from.

A record tags each image with:
  label  : 1 fake / 0 real
  domain : the REAL-image domain ("imagenet", "ffhq", "chameleon"...). For fakes
           this is the real domain the generator was conditioned to imitate.
  source : the generator ("sdxl", "midjourney"...) for fakes, "real" for reals.

These three tags are what make the shift matrix expressible: hold `domain` fixed
and vary `source` => generator shift; vary `domain` => real-domain shift.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
from pathlib import Path

import numpy as np


@dataclass
class ImageRecord:
    path: str
    label: int      # 1 fake, 0 real
    domain: str     # real-image domain
    source: str     # generator name, or "real"
    group_id: str = ""  # shared original/edit or near-duplicate identifier, if known


class Manifest:
    def __init__(self, records: list[ImageRecord]):
        self.records = records

    def __len__(self):
        return len(self.records)

    @property
    def generators(self) -> list[str]:
        return sorted({r.source for r in self.records if r.label == 1})

    @property
    def domains(self) -> list[str]:
        return sorted({r.domain for r in self.records})

    def filter(self, *, label=None, domain=None, source=None,
               exclude_source=None) -> "Manifest":
        recs = self.records
        if label is not None:
            recs = [r for r in recs if r.label == label]
        if domain is not None:
            recs = [r for r in recs if r.domain == domain]
        if source is not None:
            recs = [r for r in recs if r.source == source]
        if exclude_source is not None:
            recs = [r for r in recs if r.source != exclude_source]
        return Manifest(recs)

    # ---- protocols ------------------------------------------------------
    def leave_one_generator_out(self, held_out: str) -> tuple["Manifest", "Manifest"]:
        """LOGO with disjoint real groups; raw-file auditing is separate."""
        from .splits import group_ids, logo_split
        groups = group_ids([r.path for r in self.records], related=[r.group_id for r in self.records])
        tr, te = logo_split([r.label for r in self.records], [r.source for r in self.records], groups, held_out)
        return tuple(Manifest([self.records[i] for i in ix]) for ix in (tr, te))

    def three_way_split(self, frac=(0.7, 0.15, 0.15), seed=0
                        ) -> tuple["Manifest", "Manifest", "Manifest"]:
        """Train / calibration / test. Calibration is the real-only pool used to
        fit tau; keeping it disjoint from test is required for an honest FPR."""
        from .splits import group_ids, grouped_split
        groups = group_ids([r.path for r in self.records], related=[r.group_id for r in self.records])
        parts = grouped_split(groups, frac, seed)
        return tuple(Manifest([self.records[i] for i in p]) for p in parts)

    # ---- io -------------------------------------------------------------
    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "domain", "source", "group_id"])
            w.writeheader()
            for r in self.records:
                w.writerow(asdict(r))

    @classmethod
    def from_csv(cls, path: str | Path) -> "Manifest":
        with open(path, newline="") as f:
            recs = [ImageRecord(path=row["path"], label=int(row["label"]),
                                domain=row["domain"], source=row["source"], group_id=row.get("group_id", ""))
                    for row in csv.DictReader(f)]
        return cls(recs)
