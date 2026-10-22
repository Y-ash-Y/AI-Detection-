"""Paste into ONE NEW cell after the completed full evaluation.

Attach the NEW private serving-parity source package first. Keep the original raw
datasets attached. This does not rerun full extraction/training, edit the old source,
change thresholds, install libraries, push Git or update a Space. It processes
only 16 fixed benchmark images and writes new engineering receipts.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

# Change these only if the full evaluation was restored elsewhere.
PARITY_RUN = Path("/kaggle/working/pmsa_fresh_source_v3/source/outputs/fresh-full-01")
PARITY_SOURCE = None  # auto-find the attached NEW source/ folder; set explicitly if ambiguous
PARITY_WORKSPACE = Path("/kaggle/working/pmsa_parity_tools_01")
PARITY_OUTPUT = Path("/kaggle/working/pmsa-serving-parity-01.zip")

if PARITY_OUTPUT.exists() or PARITY_OUTPUT.with_suffix(".zip.sha256").exists():
    raise FileExistsError("Parity archive already exists; download it or use a new output name")
if json.loads((PARITY_RUN / "plan.json").read_text()).get("mode") != "full":
    raise ValueError("Point PARITY_RUN at the completed FULL run, not smoke")
if PARITY_SOURCE is None:
    candidates = [p.parent for p in Path("/kaggle/input").rglob("source_manifest.json")
                  if (p.parent / "scripts/15_check_serving_parity.py").is_file()
                  and (p.parent / "pmsa/parity.py").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Attach the NEW parity source package or set PARITY_SOURCE explicitly. Candidates: {candidates}")
    PARITY_SOURCE = candidates[0]
PARITY_SOURCE = Path(PARITY_SOURCE)

# Verify the trusted bootstrap before executing it; it then verifies all source.
manifest = json.loads((PARITY_SOURCE / "source_manifest.json").read_text())
bootstrap = PARITY_SOURCE / "kaggle_bootstrap.py"
if hashlib.sha256(bootstrap.read_bytes()).hexdigest() != manifest["files"]["kaggle_bootstrap.py"]:
    raise ValueError("Bootstrap integrity mismatch")
spec = importlib.util.spec_from_file_location("parity_bootstrap", bootstrap)
bootstrap_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap_module)
parity_repo, parity_source_digest = bootstrap_module.prepare_source_snapshot(PARITY_SOURCE, PARITY_WORKSPACE)

# Separate interpreter avoids old pmsa modules cached in this notebook kernel.
code = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch
from pmsa.artifacts import verify_bundle
from pmsa.features import FeatureSet
from pmsa.inference import Detector
from pmsa.parity import runtime_preflight, image_serving_parity

run, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    raise FileExistsError("Parity receipts exist; preserve them and choose a new workspace")
cache = run / "feature_cache"
sets = {n: FeatureSet.load(cache / f"{n}_test_v3.npz") for n in ("siglip", "forensic")}
rng = np.random.default_rng(20261008)
base = sets["siglip"]
selected = np.sort(np.concatenate([rng.choice(np.flatnonzero(base.labels == y), 8, replace=False) for y in (0, 1)]))
paths = base.paths[selected].tolist()
if any(not Path(p).is_file() for p in paths):
    raise FileNotFoundError("Original image mounts are missing. Reattach them; do not re-extract or substitute unrelated images.")
bundle = verify_bundle(run / "evaluation/fusion_seed0")
preflight = runtime_preflight(bundle)
if not preflight["compatible"]:
    raise ValueError(preflight["reason"])
out.mkdir()
torch.set_num_threads(4)
device = "cuda" if torch.cuda.is_available() else "cpu"
reports = []
for name, names in (("siglip_mlp", ["siglip"]), ("fusion", ["siglip", "forensic"])):
    for seed in (0, 1, 2):
        directory = run / f"evaluation/{name}_seed{seed}"
        bundle = verify_bundle(directory)
        det = Detector(directory / bundle["model_file"], directory / "calibrator.json", device=device)
        report = image_serving_parity(det, paths, batch_size=8, reference_sets=[sets[n] for n in names])
        report.update(model=name, seed=seed)
        reports.append(report)
        (out / f"{name}_seed{seed}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(name, seed, report["status"], flush=True)
        del det
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
summary = {"all_passed": all(r["status"] == "passed" for r in reports), "images_per_model": len(paths),
           "models_checked": len(reports), "scope": "Engineering sample, not a new accuracy/FPR evaluation; no models selected or deployed."}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
if not summary["all_passed"]:
    raise RuntimeError("Parity failed; inspect receipts and do not deploy these bundles")
'''
parity_receipts = PARITY_WORKSPACE / "receipts"
subprocess.run([sys.executable, "-c", code, str(PARITY_RUN), str(parity_receipts)],
               cwd=parity_repo, check=True)
(parity_receipts / "source.json").write_text(json.dumps({"source_snapshot_sha256": parity_source_digest}, indent=2))
with ZipFile(PARITY_OUTPUT, "x", compression=ZIP_DEFLATED) as archive:
    for path in sorted(parity_receipts.glob("*.json")):
        archive.write(path, path.name)
digest = hashlib.sha256(PARITY_OUTPUT.read_bytes()).hexdigest()
PARITY_OUTPUT.with_suffix(".zip.sha256").write_text(digest + "  " + PARITY_OUTPUT.name + "\n")
print("DOWNLOAD:", PARITY_OUTPUT, "and", PARITY_OUTPUT.with_suffix(".zip.sha256"))
print("No training, deployment, threshold changes or raw-image export occurred.")
