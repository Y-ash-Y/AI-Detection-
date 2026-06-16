#!/usr/bin/env python
"""Build a diverse, balanced training manifest + a held-out Chameleon test manifest.

Diversity on two axes (content type x generator family) so the detector can't cheat
on "faces" or "diffusion-only":

  TRAIN reals : GenImage nature (objects/scenes) + FFHQ (faces) + COCO (scenes)
  TRAIN fakes : GenImage ai (7 generators) + StyleGAN2 fakefaces (faces)
  TEST        : Chameleon test/{0_real,1_fake}  (in-the-wild — NEVER trained on)

Each record is tagged label / domain / source so the pipeline can do leave-one-out,
per-domain shift studies, etc. Outputs two CSV manifests for scripts/01 --manifest.

Example (Kaggle):
  python scripts/build_manifest.py \
    --genimage-root /kaggle/input/datasets/yangsangtai/tiny-genimage \
    --ffhq-dir      /kaggle/input/datasets/arnaud58/flickrfaceshq-dataset-ffhq \
    --fakefaces-dir /kaggle/input/datasets/hyperclaw79/fakefaces \
    --coco-dir      /kaggle/input/datasets/xthink/coco-2017-val-images/val2017 \
    --per-generator 2000 --per-source 6000
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pmsa.data import scan_genimage, scan_real_dir, scan_fake_dir
from pmsa.data.manifest import Manifest


def _summary(m: Manifest, name: str):
    by_label = Counter(r.label for r in m.records)
    by_src = Counter(r.source for r in m.records)
    by_dom = Counter(r.domain for r in m.records)
    print(f"\n[{name}] {len(m)} images  | real={by_label[0]} fake={by_label[1]}")
    print("  by domain:", dict(by_dom))
    print("  by source:", dict(by_src))


def _balance(m: Manifest, seed: int) -> Manifest:
    """Subsample the majority class so real and fake counts match."""
    reals = [r for r in m.records if r.label == 0]
    fakes = [r for r in m.records if r.label == 1]
    k = min(len(reals), len(fakes))
    rng = np.random.default_rng(seed)
    reals = [reals[i] for i in rng.permutation(len(reals))[:k]]
    fakes = [fakes[i] for i in rng.permutation(len(fakes))[:k]]
    return Manifest(reals + fakes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genimage-root")
    ap.add_argument("--ffhq-dir")
    ap.add_argument("--fakefaces-dir")
    ap.add_argument("--coco-dir")
    ap.add_argument("--chameleon-dir",
                    help="folder containing 0_real/ and 1_fake/ (Chameleon/test)")
    ap.add_argument("--per-generator", type=int, default=2000,
                    help="cap per GenImage generator per class")
    ap.add_argument("--per-source", type=int, default=6000,
                    help="cap for FFHQ / fakefaces / COCO")
    ap.add_argument("--per-test", type=int, default=None, help="cap per Chameleon class")
    ap.add_argument("--balance", action="store_true", help="equalize real/fake in train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-train", default="feature_cache/manifest_diverse_train.csv")
    ap.add_argument("--out-test", default="feature_cache/manifest_chameleon_test.csv")
    args = ap.parse_args()

    train_records = []
    if args.genimage_root:
        m = scan_genimage(args.genimage_root, per_class_limit=args.per_generator)
        train_records += m.records
        print(f"genimage: {len(m)} (generators={m.generators})")
    if args.ffhq_dir:
        m = scan_real_dir(args.ffhq_dir, domain="faces", limit=args.per_source)
        train_records += m.records
        print(f"ffhq real: {len(m)}")
    if args.fakefaces_dir:
        m = scan_fake_dir(args.fakefaces_dir, source="stylegan2", domain="faces",
                          limit=args.per_source)
        train_records += m.records
        print(f"stylegan2 fake: {len(m)}")
    if args.coco_dir:
        m = scan_real_dir(args.coco_dir, domain="coco", limit=args.per_source)
        train_records += m.records
        print(f"coco real: {len(m)}")

    train = Manifest(train_records)
    if args.balance:
        train = _balance(train, args.seed)
    _summary(train, "TRAIN")
    train.to_csv(args.out_train)
    print(f"-> {args.out_train}")

    if args.chameleon_dir:
        c = Path(args.chameleon_dir)
        real = scan_real_dir(c / "0_real", domain="chameleon", limit=args.per_test)
        fake = scan_fake_dir(c / "1_fake", source="chameleon", domain="chameleon",
                             limit=args.per_test)
        test = Manifest(real.records + fake.records)
        _summary(test, "TEST (Chameleon, held out)")
        test.to_csv(args.out_test)
        print(f"-> {args.out_test}")


if __name__ == "__main__":
    main()
