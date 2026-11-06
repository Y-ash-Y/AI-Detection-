# Extraction and caches

[extract.py](extract.py) connects the shared image policy and feature extractors.
[cache.py](cache.py) defines the NPZ format: feature rows, labels, paths,
domains, sources, stream identity and extraction metadata. New caches can also
carry image hashes and related-group IDs. Stream alignment checks these identities
and row order before fusion.

Use [01_extract_features.py](../../scripts/01_extract_features.py) to generate
caches. Preserve them and their extraction records. Legacy archives without
provenance can support exploratory comparisons, but do not establish equivalence
to current image inference. See the [local cache convention](../../feature_cache/README.md).
