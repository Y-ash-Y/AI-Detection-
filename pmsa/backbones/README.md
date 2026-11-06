# Feature extractors

- [SigLIP](siglip_encoder.py): Google's frozen base image encoder (768 features
  for the current serving model). This repository trains detector heads, not
  the backbone. Use the revision and processor settings recorded with a bundle.
- [Forensic](forensic.py): deterministic 66-dimensional spectral, color and
  residual statistics. No pretrained weights are needed for this stream.
- [Hub revision helper](hub_revision.py): resolves model revisions so feature
  extraction can record a fixed identity.

Training and inference must use the same extractor and preprocessing settings.
The earlier CLIP, DINOv2 and NPR variants are not active implementations.
See the [feature cache guide](../features/README.md) before mixing streams.
