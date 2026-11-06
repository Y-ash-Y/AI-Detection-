# Continuous integration

[tests.yml](tests.yml) runs offline regression checks on Python 3.11 and 3.12
with CPU Torch, the synthetic pipeline example and Python compilation. It uses
read-only repository permissions. It does not use private datasets, retrain a
serving model or deploy the Space.

See [tests](../../tests/README.md) for local commands. A local pass does not
establish that hosted CI ran for the same revision.
