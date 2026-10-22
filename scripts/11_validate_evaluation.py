#!/usr/bin/env python
"""Independent result reconciliation and paired bootstrap for a recovered-cache run."""
import argparse
import csv
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmsa.artifacts import sha256_file
from pmsa.eval.metrics import roc_auc, tpr_at_fpr, operating_report
from pmsa.features import FeatureSet


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run, control, out = Path(args.run), Path(args.control), Path(args.out)
    if out.exists():
        raise FileExistsError("Validation output exists; choose a new path")
    if args.bootstrap < 1:
        raise ValueError("Positive bootstrap count required")
    plan = read(run / "plan.json")
    extra = read(control / "plan.json")
    if any(plan[k] != extra[k] for k in ("seeds", "epochs", "alpha", "shared_group_reference", "bootstrap_resamples")):
        raise ValueError("Control experiment settings differ")
    for root in (run, control):
        for path, digest in read(root / "preserved_artifacts.json").items():
            if sha256_file(path) != digest:
                raise ValueError("Original artifact changed")
    entries = read(run / "results.json") + read(control / "results.json")
    predictions, checks, reference_splits = {}, [], {}
    baseline_labels, baseline_paths = None, None
    for item in entries:
        name, seed = item["model"], item["seed"]
        root = control if name == "siglip_mlp" else run
        directory = root / f"{name}_seed{seed}"
        parts = read(directory / "splits.json")["partitions"]
        if seed in reference_splits and parts != reference_splits[seed]:
            raise ValueError("Fit/validation/calibration/test partitions differ across ablations")
        reference_splits[seed] = parts
        with np.load(directory / "external_predictions.npz", allow_pickle=False) as data:
            scores, labels, paths = data["scores"], data["labels"], data["paths"]
        if baseline_labels is None:
            baseline_labels, baseline_paths = labels.copy(), paths.copy()
        if not np.array_equal(labels, baseline_labels) or not np.array_equal(paths, baseline_paths):
            raise ValueError("External rows differ across models/seeds")
        predictions[name, seed] = scores
        tau = item["external"]["operating"]["tau"]
        real, fake, flagged = labels == 0, labels == 1, scores > tau
        direct = {"fpr": float(flagged[real].sum()/real.sum()),
                  "tpr": float(flagged[fake].sum()/fake.sum()),
                  "accuracy": float((flagged == fake).sum()/len(labels))}
        for metric, value in direct.items():
            if not np.isclose(value, item["external"]["operating"][metric]["point"], atol=1e-12):
                raise ValueError("Independent operating-rate reconciliation failed")
        checks.append({"model": name, "seed": seed, "direct_rates": direct,
                       "confusion_counts": {"tn": int((~flagged & real).sum()), "fp": int((flagged & real).sum()),
                                            "fn": int((~flagged & fake).sum()), "tp": int((flagged & fake).sum())}})
    manifests = {}
    for tag, filename in (("diversetrain", "manifest_diverse_train.csv"),
                          ("chameleontest", "manifest_chameleon_test.csv")):
        fs = FeatureSet.load(Path(args.cache_dir) / f"siglip_{tag}.npz")
        path = Path(args.cache_dir) / filename
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(fs):
            raise ValueError("Manifest row count differs")
        for key in ("path", "label", "domain", "source"):
            column = "paths" if key == "path" else "labels" if key == "label" else key
            if not np.array_equal(np.array([row[key] for row in rows]), getattr(fs, column).astype(str)):
                raise ValueError(f"Manifest mismatch: {key}")
        manifests[tag] = {"rows": len(rows), "sha256": sha256_file(path), "all_fields_match_cache": True}
    rng = np.random.default_rng(20261002)
    pos, neg = np.flatnonzero(baseline_labels == 1), np.flatnonzero(baseline_labels == 0)
    pairs = [("fusion", "siglip_probe"), ("fusion", "siglip_mlp")]
    stats = {f"{a}_minus_{b}": {"auc": [], "tpr_at_5pct": []} for a, b in pairs}
    for i in range(args.bootstrap):
        ix = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        lab = baseline_labels[ix]
        measures = {name: {"auc": float(np.mean([roc_auc(predictions[name, s][ix], lab) for s in plan["seeds"]])),
                           "tpr_at_5pct": float(np.mean([tpr_at_fpr(predictions[name, s][ix], lab, .05) for s in plan["seeds"]]))}
                    for name in ("fusion", "siglip_probe", "siglip_mlp")}
        for a, b in pairs:
            for metric in ("auc", "tpr_at_5pct"):
                stats[f"{a}_minus_{b}"][metric].append(measures[a][metric]-measures[b][metric])
        if i % 500 == 0:
            print(f"Paired bootstrap {i}/{args.bootstrap}", flush=True)
    deltas = {}
    for a, b in pairs:
        key = f"{a}_minus_{b}"
        deltas[key] = {}
        for metric in ("auc", "tpr_at_5pct"):
            fn = roc_auc if metric == "auc" else lambda s, l: tpr_at_fpr(s, l, .05)
            point = float(np.mean([fn(predictions[a, s], baseline_labels)-fn(predictions[b, s], baseline_labels)
                                   for s in plan["seeds"]]))
            low, high = np.quantile(stats[key][metric], [.025, .975])
            deltas[key][metric] = {"point": point, "lo": float(low), "hi": float(high)}
    # Inspect real-domain false alarms on the untouched internal test partitions.
    development = FeatureSet.load(Path(args.cache_dir) / "siglip_diversetrain.npz")
    domains_by_path = dict(zip(development.paths, development.domain))
    domain_rates = []
    for item in entries:
        name, seed = item["model"], item["seed"]
        root = control if name == "siglip_mlp" else run
        with np.load(root / f"{name}_seed{seed}" / "test_predictions.npz", allow_pickle=False) as d:
            scores, labels, paths = d["scores"], d["labels"], d["paths"]
        domains = np.array([domains_by_path[p] for p in paths])
        for domain in sorted(set(domains)):
            mask = domains == domain
            operating = operating_report(scores[mask], labels[mask], item["training"]["operating"]["tau"])
            domain_rates.append({"model": name, "seed": seed, "domain": str(domain), "operating": operating})
    payload = {"verified_at": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "manifest_reconciliation": manifests, "shared_splits_and_external_order_verified": True,
        "original_artifacts_unchanged": True, "paired_deltas": deltas, "internal_domain_rates": domain_rates,
        "interval_scope": "Paired 95% percentile bootstrap over independent image representatives, conditional on the three fitted heads; not uncertainty over all possible training runs.",
        "caveats": ["Legacy exact feature identities are proxies, not raw-image hashes; unknown near-duplicates may remain.",
                    "Chameleon has been evaluated during prior project development; this is not a newly blinded benchmark.",
                    "The nonlinear SigLIP control was added after the initial comparison; all control seeds are reported.",
                    "Differences compare frozen learned heads; forensic causal value cannot be proved by ranking alone.",
                    "TPR@5% here uses test-derived thresholds, not the frozen deployed calibrator."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(deltas, indent=2), flush=True)


if __name__ == "__main__":
    main()
