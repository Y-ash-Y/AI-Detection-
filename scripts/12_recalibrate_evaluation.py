#!/usr/bin/env python
"""Supplemental target-real calibration; primary frozen-threshold results stay intact."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.calibration import Calibrator
from pmsa.eval.metrics import operating_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--real-calibration", type=int, default=2000)
    ap.add_argument("--split-seed", type=int, default=20261002)
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out, run, control = Path(args.out), Path(args.run), Path(args.control)
    if out.exists():
        raise FileExistsError("Supplement output exists; choose a new directory")
    entries = json.loads((run / "results.json").read_text())+json.loads((control / "results.json").read_text())
    reference = None
    cal_ix, test_ix, results = None, None, []
    out.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
        "target_real_calibration_n": args.real_calibration, "alpha": args.alpha, "seed": args.split_seed,
        "reason": "Supplement after frozen-threshold analysis; assess target-real recalibration/power cost.",
        "selection": "Uniform random real-only calibration subset; all other real examples and all fake examples test-only.",
        "scope": "Not a zero-shot threshold. No classifier training or test-fake calibration."}, indent=2))
    for entry in entries:
        name, seed = entry["model"], entry["seed"]
        directory = (control if name == "siglip_mlp" else run) / f"{name}_seed{seed}"
        with np.load(directory / "external_predictions.npz", allow_pickle=False) as data:
            scores, labels, paths = data["scores"], data["labels"], data["paths"]
        if reference is None:
            reference = (labels.copy(), paths.copy())
            real = np.flatnonzero(labels == 0)
            if not 0 < args.real_calibration < len(real):
                raise ValueError("Need nonempty, disjoint real calibration and test pools")
            perm = np.random.default_rng(args.split_seed).permutation(real)
            cal_ix = np.sort(perm[:args.real_calibration])
            test_ix = np.sort(np.concatenate([perm[args.real_calibration:], np.flatnonzero(labels == 1)]))
            if np.intersect1d(cal_ix, test_ix).size or np.any(labels[cal_ix] != 0):
                raise ValueError("Target calibration/test leakage")
            (out / "splits.json").write_text(json.dumps({"calibration_indices": cal_ix.tolist(),
                "test_indices": test_ix.tolist(), "calibration_paths": paths[cal_ix].tolist(),
                "n_calibration_real": len(cal_ix), "n_test_real": int((labels[test_ix] == 0).sum()),
                "n_test_fake": int((labels[test_ix] == 1).sum())}, indent=2))
        if not np.array_equal(labels, reference[0]) or not np.array_equal(paths, reference[1]):
            raise ValueError("Supplement prediction order mismatch")
        cal = Calibrator.fit(scores[cal_ix], args.alpha, "conformal", "chameleon-supplemental-calibration")
        previous = entry["external"]["operating"]["tau"]
        results.append({"model": name, "seed": seed, "tau_base": previous, "tau_recalibrated": cal.tau,
            "base_on_same_test": operating_report(scores[test_ix], labels[test_ix], previous),
            "recalibrated_on_same_test": operating_report(scores[test_ix], labels[test_ix], cal.tau)})
    summary = {}
    for name in sorted(set(r["model"] for r in results)):
        rows = [r for r in results if r["model"] == name]
        summary[name] = {}
        for setting in ("base_on_same_test", "recalibrated_on_same_test"):
            summary[name][setting] = {metric: {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}
                for metric, values in {m: [r[setting][m]["point"] for r in rows] for m in ("fpr", "tpr")}.items()}
    (out / "results.json").write_text(json.dumps({"runs": results, "summary": summary,
        "caveats": ["Reuses a previously viewed benchmark; supplemental, exploratory evidence only.",
                    "No raw-image hashes; unknown near-duplicates can violate independence.",
                    "Target calibration needs trusted real images representative of deployment; it cannot guarantee arbitrary future domains.",
                    "Mean/min/max across seeds are not confidence intervals; per-run Wilson intervals are conditional on each threshold."]}, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
