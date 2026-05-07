"""
Build feature cache from a dataset.

Run this ONCE before training. Saves all features to .npz so you
never re-run CLIP/DINO during training experiments.

Usage:
    python -m image.pipeline.build_features \\
        --real  data/real \\
        --fake  data/fake \\
        --out   feature_cache/features.npz

    # Or for CIFAKE:
    python -m image.pipeline.build_features \\
        --cifake /path/to/cifake \\
        --split  train \\
        --out    feature_cache/cifake_train.npz
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from image.pipeline.feature_extraction import FeaturePipeline
from image.data.dataset_loader import load_dataset, load_cifake


def build(dataset, output_path: str, batch_size: int = 1):
    pipeline = FeaturePipeline()

    clip_list, dino_list, forensic_list, labels = [], [], [], []

    failed = 0
    for path, label in tqdm(dataset, desc="Extracting features"):
        try:
            c, d, f = pipeline.extract(path)
            clip_list.append(c)
            dino_list.append(d)
            forensic_list.append(f)
            labels.append(label)
        except Exception as e:
            print(f"[SKIP] {path}: {e}")
            failed += 1

    print(f"\nExtracted {len(labels)} samples ({failed} failed)")
    if not labels:
        raise RuntimeError("No features were extracted; refusing to save an empty feature cache.")

    print(f"Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip_arr = np.array(clip_list, dtype=np.float32)
    dino_arr = np.array(dino_list, dtype=np.float32)
    forensic_arr = np.array(forensic_list, dtype=np.float32)
    label_arr = np.array(labels, dtype=np.float32)

    np.savez(
        output_path,
        clip=clip_arr,
        dino=dino_arr,
        forensic=forensic_arr,
        label=label_arr,
    )

    print(f"Saved. Shapes:")
    print(f"  clip     : {clip_arr.shape}")
    print(f"  dino     : {dino_arr.shape}")
    print(f"  forensic : {forensic_arr.shape}")
    print(f"  labels   : {label_arr.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",   type=str, default=None)
    parser.add_argument("--fake",   type=str, default=None)
    parser.add_argument("--cifake", type=str, default=None,
                        help="Path to CIFAKE root directory")
    parser.add_argument("--split",  type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--out",    type=str,
                        default="feature_cache/features.npz")
    args = parser.parse_args()

    if args.cifake:
        dataset = load_cifake(args.cifake, split=args.split)
    elif args.real and args.fake:
        dataset = load_dataset(args.real, args.fake)
    else:
        raise ValueError("Provide either --cifake or both --real and --fake")

    build(dataset, args.out)
