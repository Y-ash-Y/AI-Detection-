"""Prepare private reviewed, group-safe data; never scores/trains/deploys."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.data.intake import reserve_history, inventory, build, select_development, build_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reserve = commands.add_parser("reserve", help="Reserve already-inspected image identities")
    reserve.add_argument("--real-plan", required=True)
    reserve.add_argument("--fake-plan", required=True)
    reserve.add_argument("--out", required=True)
    scan = commands.add_parser("inventory", help="Create a no-scoring review sheet")
    scan.add_argument("--input", required=True)
    scan.add_argument("--reserved", required=True)
    scan.add_argument("--out", required=True)
    split = commands.add_parser("build", help="Freeze four disjoint roles after manual review")
    split.add_argument("--inventory", required=True)
    split.add_argument("--review", required=True)
    split.add_argument("--out", required=True)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--alpha", type=float, default=.05)
    split.add_argument("--fractions", type=float, nargs=4, default=(.5, .15, .2, .15),
                       metavar=("TRAIN", "VALIDATION", "CALIBRATION", "TEST"))
    select = commands.add_parser("select-development", help="Explicit reuse: sample evaluated generators for development, not blind testing")
    select.add_argument("--inventory", required=True)
    select.add_argument("--generator-dir", action="append", required=True, metavar="SOURCE:RELATIVE_DIR")
    select.add_argument("--per-generator", type=int, default=50)
    select.add_argument("--seed", type=int, default=0)
    select.add_argument("--fully-generated", action="store_true", help="User explicitly attests generation from scratch")
    select.add_argument("--out", required=True)
    develop = commands.add_parser("build-development", help="Freeze explicit reused-development roles; test is diagnostic, not blind")
    develop.add_argument("--selection", required=True)
    develop.add_argument("--review", required=True)
    develop.add_argument("--out", required=True)
    develop.add_argument("--seed", type=int, default=0)
    develop.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    if args.command == "reserve":
        result = reserve_history(args.real_plan, args.fake_plan, args.out)
        print(f"Reserved {len(result['images'])} previously inspected image records")
    elif args.command == "inventory":
        result = inventory(args.input, args.reserved, args.out)
        print(f"Inventory: {result['counts']}; no images scored or moved")
    elif args.command == "select-development":
        folders = {}
        for item in args.generator_dir:
            if ":" not in item:
                parser.error("--generator-dir requires SOURCE:RELATIVE_DIR")
            source, folder = item.split(":", 1)
            if source in folders:
                parser.error("Repeated generator source")
            folders[source] = folder
        result = select_development(args.inventory, folders, args.out,
                                    per_generator=args.per_generator, seed=args.seed,
                                    fully_generated=args.fully_generated)
        print(f"Development review selection: {result['selected_counts']}; no scoring, training or blind-test set")
    elif args.command == "build-development":
        result = build_development(args.selection, args.review, args.out, seed=args.seed, alpha=args.alpha)
        print(f"Development roles: {result['summary']}; internal diagnostic only, no model fitted")
    else:
        result = build(args.inventory, args.review, args.out, seed=args.seed,
                       alpha=args.alpha, fractions=args.fractions)
        print(f"Frozen roles: {result['summary']}; no model fitted")


if __name__ == "__main__":
    main()
