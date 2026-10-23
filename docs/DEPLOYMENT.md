# Minimal Hugging Face serving release

The release was verified on 2026-10-23. The Space is running; all 32 published
file hashes matched the staged payload, a synthetic HTTP upload succeeded, and
invalid input was rejected. The hosted score matched the local CPU score within
the existing parity tolerance. GitHub CI passed for the serving-code commit.
See `evidence/deployment_summary.json` for the exact revisions and scope.

The subsequent UI-only release uses a clear result card, upload preview and
collapsed Developer analysis panel, without an app benchmark tab. Its 32 remote
file hashes and synthetic HTTP regression passed; model/calibrator bytes were
unchanged. See `evidence/ui_release_summary.json` for its revision and scope.
The upload target and preview occupy one shared panel: a valid upload replaces
the drop area in place, and Clear restores it. Inference still reads the original
file rather than the display thumbnail.

Deployment is separate from source publication and model improvement. The Space
uses the fixed fresh-full fusion seed 0 reference and its unchanged calibrator,
not a best diagnostic seed or an unpromoted modern-data candidate.

## Release contents

`scripts/19_prepare_space.py` stages an explicit serving-source allowlist and the
templates under `deployment/space/`. It refuses an existing destination and stages
only below ignored `outputs/`. The model and calibrator retain their original
bytes. A new, sanitized deployment split receipt removes all image paths, image
hashes, group IDs, prediction rows and sample indices. Its changed hash is bound
by the release copy's bundle; original experiment files are not altered.

The released `release_manifest.json` identifies the source bundle and serving
files. It is an integrity manifest, not publisher authentication or proof of
detection reliability. No personal images, PDFs, interview material, training
caches or original private split records belong in the Space.

```bash
python scripts/19_prepare_space.py \
  --bundle /path/to/trusted/fusion_seed0 \
  --out outputs/new_space_release/payload
```

This command does not log in or upload. Publishing requires an authenticated
account with write access to the intended Space; never put tokens in source,
commit messages or README files. Review the exact payload file/hash manifest
and the remote revision before uploading.

## Runtime and entrypoint

The Space template selects Python 3.12, Gradio 6.18.0 and exact extractor versions.
It includes the same MCP/OAuth extras automatically installed by the Space SDK,
with their compatible Pydantic 2.12.5 pin. These extras are runtime dependencies;
this application does not enable an MCP server or change its detection method.
Linux CPU Torch wheels differ in build suffix from the original CUDA extraction,
which the existing core-version compatibility rule permits. Keep the backbone
and processor pinned to the recorded immutable Hub revision; do not edit a bundle
to bypass mismatches. Build-time preloading uses that same revision.

`space_app.py` selects the staged `model/` directory, disables legacy checkpoint
overrides, forces CPU serving, and disables Gradio analytics. It does not change
Space hardware, paid settings, token storage or the trained model. The main
library/app remains usable independently of this hosting wrapper.

## Verification and recovery

Before publication, verify bundle integrity, run regression tests and send a
non-sensitive synthetic image through the staged HTTP interface. This tests
serving, not classification accuracy. Also check invalid uploads fail clearly.
After publication, verify the build status, deployed revision, uploaded file
hashes, HTTP interface and the same synthetic test. Local macOS success alone
does not validate a Linux-hosted build.

Preserve the previous Space revision and a private backup before replacing
obsolete files. A failed release can be restored from that known revision without
rewriting Git history or touching original experiment artifacts.

The app does not train on or publish uploads, but Gradio temporarily caches them
and hosting logs can have separate retention. Never promise zero retention or
use personal photos for a public deployment smoke test.
