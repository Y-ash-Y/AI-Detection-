# Serving and release staging

For local inference from the repository root, install
`requirements-app.txt` and pass a trusted bundle directory with
`PMSA_BUNDLE=/path/to/bundle python app.py`. The default bundle location is
`outputs/deploy_v3`. The current fusion bundle pairs `fusion.pt`,
`calibrator.json`, `splits.json` and `bundle.json`. Use files from the same
run and a compatible runtime; hashes verify pairing and integrity, not publisher
identity. Do not load untrusted serialized models.

[space/](space/README.md) contains the standalone Hugging Face Space card,
entry point and pinned CPU dependencies. They are not the generic local app
environment. For an approved, trusted two-stream fusion bundle, stage a
separate payload locally:

```sh
python scripts/19_prepare_space.py --bundle /path/to/trusted-bundle \
  --out outputs/new-space-release/payload
```

The [staging script](../scripts/19_prepare_space.py) copies only allowlisted
serving files and sanitizes split identities without changing the source model
or calibrator. It refuses to overwrite a prior payload and does not upload.
Inspect the staged files before a separate deployment. See the [Space card](space/README.md)
for user-facing model limits and upload privacy guidance.
