#!/usr/bin/env python
"""Phase 1: the heavy extraction pass. Run on the GPU that holds the data (Kaggle T4).

Builds a manifest (GenImage + any extra real/fake dirs), then caches one npz per
backbone. Everything downstream reads these npz and never touches a GPU again.

Example (Kaggle):
    python scripts/01_extract_features.py \
        --genimage-root /kaggle/input/genimage \
        --generators stable_diffusion_v_1_4 sdxl midjourney \
        --per-class-limit 5000 \
        --split train --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pmsa.config import Config
from pmsa.data import scan_genimage, scan_real_dir, scan_fake_dir
from pmsa.data.manifest import Manifest
from pmsa.features import extract_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--genimage-root")
    ap.add_argument("--generators", nargs="*", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-class-limit", type=int, default=None)
    ap.add_argument("--real-dir", nargs="*", default=[],
                    help="extra pure-real dirs as DOMAIN:PATH (shift domains)")
    ap.add_argument("--fake-dir", nargs="*", default=[],
                    help="extra pure-fake dirs as SOURCE:PATH (in-the-wild)")
    ap.add_argument("--tag", default="train", help="cache filename tag")
    ap.add_argument("--manifest", default=None,
                    help="extract from a prebuilt manifest CSV (skips arg-based scan)")
    ap.add_argument("--normalize-jpeg", action="store_true",
                    help="re-encode every image to JPEG q90 (controls format bias)")
    ap.add_argument("--real-limit", type=int, default=None,
                    help="cap images per --real-dir")
    ap.add_argument("--fake-limit", type=int, default=None,
                    help="cap images per --fake-dir")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = args.device or "cuda"

    if args.manifest:
        manifest = Manifest.from_csv(args.manifest)
        print(f"manifest: {len(manifest)} images from {args.manifest}")
        cache_dir = Path(cfg.feature.cache_dir)
        for bb in cfg.feature.backbones:
            if not bb.enabled:
                continue
            extract_backbone(
                manifest, bb.name, cache_dir / f"{bb.name}_{args.tag}.npz",
                device=device, batch_size=cfg.feature.batch_size,
                num_workers=cfg.feature.num_workers, weights=bb.weights,
                image_size=bb.image_size, normalize_jpeg=args.normalize_jpeg)
        return

    records = []
    if args.genimage_root:
        m = scan_genimage(args.genimage_root, args.generators, args.split,
                          args.per_class_limit)
        records += m.records
        print(f"genimage: {len(m)} images, generators={m.generators}")
    for spec in args.real_dir:
        domain, path = spec.split(":", 1)
        m = scan_real_dir(path, domain, limit=args.real_limit)
        records += m.records
        print(f"real[{domain}]: {len(m)} images")
    for spec in args.fake_dir:
        source, path = spec.split(":", 1)
        m = scan_fake_dir(path, source, limit=args.fake_limit)
        records += m.records
        print(f"fake[{source}]: {len(m)} images")

    manifest = Manifest(records)
    if len(manifest) == 0:
        raise SystemExit("empty manifest — pass --genimage-root and/or dirs")

    cache_dir = Path(cfg.feature.cache_dir)
    manifest.to_csv(cache_dir / f"manifest_{args.tag}.csv")
    print(f"manifest: {len(manifest)} total -> {cache_dir}/manifest_{args.tag}.csv")

    for bb in cfg.feature.backbones:
        if not bb.enabled:
            continue
        out = cache_dir / f"{bb.name}_{args.tag}.npz"
        extract_backbone(
            manifest, bb.name, out, device=device,
            batch_size=cfg.feature.batch_size,
            num_workers=cfg.feature.num_workers,
            weights=bb.weights, image_size=bb.image_size,
            normalize_jpeg=args.normalize_jpeg,
        )


if __name__ == "__main__":
    main()
