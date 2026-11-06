# Shared utilities

[device.py](device.py) handles compute device selection;
[seed.py](seed.py) provides shared seed setup for reproducible experiments.
Record the seed and runtime used for each run. A fixed seed does not promise
byte-identical execution across different hardware or library versions.

See the [default configuration](../../configs/default.yaml) and
[experiment workflow](../../scripts/README.md).
