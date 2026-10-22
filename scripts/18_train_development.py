"""Private local modern-generator replay; no deployment, installs or uploads."""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pmsa.development import prepare_features, compare_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "compare"))
    parser.add_argument("--roles", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--broad-archive", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.stage == "extract":
        if not args.selection:
            parser.error("extract requires --selection")
        import torch
        torch.set_num_threads(4)
        prepare_features(args.roles,args.selection,args.broad_archive,args.baseline,args.features)
    else:
        if not args.out:
            parser.error("compare requires --out")
        compare_replay(args.broad_archive,args.baseline,args.roles,args.features,args.out)


if __name__ == "__main__":
    main()
