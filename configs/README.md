# Experiment configuration

[default.yaml](default.yaml) selects the current SigLIP and forensic streams,
workspace paths and training, calibration and evaluation defaults.
[Experiment overrides](experiments/README.md) change individual sections;
[pmsa/config.py](../pmsa/config.py) loads them. Check CLI defaults as well:
the evaluation's 5% reference target is not a promise that every script or
config uses the same target.

[requirements-photo-eval.txt](requirements-photo-eval.txt) pins the recorded
macOS Python 3.12 fresh-full evaluation runtime. Use a separate environment;
do not overwrite the app's environment. Generic dependency ranges are in
[pyproject.toml](../pyproject.toml); [Space pins](../deployment/space/requirements.txt)
describe a different, standalone CPU deployment.
