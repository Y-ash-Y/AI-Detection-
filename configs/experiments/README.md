# Experiment overrides

[logo_genimage.yaml](logo_genimage.yaml) holds the leave-one-generator-out
GenImage experiment settings. It is a protocol configuration, not a deployment
configuration or a record of measured results. Use it with the appropriate
[experiment script and input cache](../../scripts/README.md); inspect
`--help` for tags and generator selection.

Overrides are shallow within each section. Review effective values and pick a
new output path for new runs.
