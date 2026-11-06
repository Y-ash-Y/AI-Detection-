# Offline example

Run from the repository root after installing `.[dev]`:

```sh
python examples/offline_smoke.py
```

[offline_smoke.py](offline_smoke.py) creates synthetic fixture images and uses
the forensic-only path to exercise extraction, probe fitting, threshold
calibration and prediction. It uses disposable artifacts, needs no pretrained
weights, and does not measure accuracy on real or AI-generated images.
For automated verification, see the [test guide](../tests/README.md).
