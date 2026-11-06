# Local feature caches

This directory is a workspace for extracted features and associated manifests.
Only this guide and the placeholder belong in Git. Caches can contain private
paths, image identities and source labels even when raw images are absent.

Generate caches with [01_extract_features.py](../scripts/01_extract_features.py).
Keep the cache files and extraction records together. Do not delete a cache as
routine cleanup: reproducing it can require the original data, revision and
GPU extraction pass. See the [cache format](../pmsa/features/README.md).
