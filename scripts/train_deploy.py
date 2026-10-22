#!/usr/bin/env python
"""Train a disjoint-split v3 detector. Defaults never overwrite legacy outputs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.config import Config
from pmsa.training import train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="diversetrain_v3")
    ap.add_argument("--model", choices=["probe", "fusion"], default="probe")
    ap.add_argument("--backbone", default="siglip")
    ap.add_argument("--backbones", nargs="+", default=["siglip", "forensic"])
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/deploy_v3")
    ap.add_argument("--exploratory-legacy", action="store_true")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    names = [args.backbone] if args.model == "probe" else args.backbones
    report = train(cfg, names, args.tag, args.out, args.model, args.alpha,
                   args.device, args.exploratory_legacy)
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.out}")
    if not report["exploratory_legacy"]:
        print(f"PMSA_BUNDLE={args.out} python app.py")


if __name__ == "__main__":
    main()
