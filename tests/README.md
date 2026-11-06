# Offline regression tests

From the repository root:

```sh
python -m pip install -e '.[dev,app]'
python -m pytest -q
python examples/offline_smoke.py
```

Use `.[dev]` for the lightweight suite; optional neural/app groups skip when
their dependencies are unavailable. CI exercises Python 3.11/3.12 with CPU
Torch and offline model settings. No benchmark, Hugging Face login or GPU is
required for these tests.

Coverage includes grouped splits and reviewed intake, feature identity and
cache alignment, threshold ties and false-positive budgets, artifact contract
checks, synthetic end-to-end workflows, fixed-revision recovery, serving parity,
app input handling and allowlisted Space staging. Passing tests verify
implementation behavior, not detector accuracy or a live deployment.
See [CI](../.github/workflows/tests.yml) and the
[evaluation record](../reference/EVALUATION.md).
