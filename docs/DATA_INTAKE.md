# Reviewed modern-image development workflow

This prepares data, not a new detector. It never trains, scores, uploads, moves
or changes original photos. Use local ignored `outputs/` for all personal image
lists, review sheets and manifests; do not commit them or personal embeddings.

## Folder layout

```text
pmsa-real-photos/
  real/          camera-origin photos, no generative edits
  ai_generated/  generated exports; explicitly disclose any AI edits
```

Folder names propose labels; they do not authenticate provenance. Folders
named `REAL/` and `AI generated/` are also accepted (case, spaces and underscores
are normalized in the folder name, without renaming source files). Existing files
may be moved into these folders by their owner. Moving or renaming a previously
tested image does not make it new. The reserved registry checks encoded hashes
and exact decoded RGB identities, including lossless re-encoding. Cropped,
resized or recompressed versions require human related-scene/prior-use review.

Conventional computer graphics/logos are not automatically AI-generated. Keep
them outside the primary photographic binary experiment. AI-edited originals
and their edited versions form a separate, grouped challenge, not independent
camera-versus-fully-generated samples.

## Collection goal

For an initial head-training pilot, aim for roughly 100 NEW camera photos and
100 NEW fully generated images, with multiple independent scenes/prompts in
each class. This is a starting collection target, not a statistically adequate
final benchmark or promise of a particular detection rate. Shared scenes/bursts
reduce effective sample size. More than one device or generator helps coverage,
but devices/prompts are not treated as independent merely because they differ.

Include people/groups, objects, indoor/outdoor scenes and lighting conditions in
BOTH classes. Record sharing, ordinary editing and export history; avoid making
all reals WhatsApp JPEGs while all fakes are clean studio portraits/PNG exports.
The existing model JPEG normalization does not remove all acquisition biases.
Record actual generator/model version when known; use an explicit source such
as `chatgpt_unknown_version` if only the service is known. Do not invent a model
version from a filename. Licensed sourced images require a recorded usable
license in `notes`; camera permission and generator origin remain user assertions.

Do not use the already-inspected personal diagnostic images, Chameleon test
images, previous frontier benchmark examples or related variants as fresh
development/holdout examples. The tool automatically protects the registered
personal identities; other historical uses must be disclosed by the curator.

Previously evaluated examples CAN be intentionally re-designated as development
data. This is a different protocol, documented below; prior-use history must never
be changed to `new`, and reused examples are not an untouched confirmation test.

## Three stages

Use the matching local environment when available. The commands below are
terminal commands; Kaggle notebook commands require `!python`, and paths must
be updated to that notebook's explicitly attached datasets. No network or
backbone downloads are required for these stages.

1. Reserve historical examples ONCE from the trusted private receipts:

```bash
outputs/photo_eval_env_01/bin/python scripts/17_prepare_data.py reserve \
  --real-plan outputs/real_photo_test_01/plan.json \
  --fake-plan outputs/chatgpt_six_diagnostic_01/plan.json \
  --out outputs/data_intake_01/reserved.json
```

2. Create a new immutable inventory and editable review sheet AFTER adding files:

```bash
outputs/photo_eval_env_01/bin/python scripts/17_prepare_data.py inventory \
  --input /absolute/path/to/pmsa-real-photos \
  --reserved outputs/data_intake_01/reserved.json \
  --out outputs/data_intake_01/batch_01
```

The output is `inventory.json` and `review.csv`. Choose a NEW batch directory
after any addition, move or change; never overwrite a snapshot. Inventory is
no-scoring intake, so it does not reveal model outcomes before split decisions.
Exact duplicates are linked later; difference hashes only provide bounded
review hints and cannot establish authenticity or scene independence. Existing
protected unsupported inputs are not re-decoded or counted as correct samples.

3. Review metadata, then freeze the four roles:

```bash
outputs/photo_eval_env_01/bin/python scripts/17_prepare_data.py build \
  --inventory outputs/data_intake_01/batch_01/inventory.json \
  --review outputs/data_intake_01/batch_01/review.csv \
  --out outputs/data_intake_01/splits_01 --seed 0 --alpha 0.05
```

Reserved files never enter development, even if their CSV status/approval is
edited. Unapproved or contradictory new rows stop the build before writing any
splits. Decode failures and prior-use components are explicitly excluded. Input
and registry changes require a new inventory instead of silently changing the
experiment. All source photos remain in place.

## Review columns

| Field | Meaning and allowed practice |
| --- | --- |
| `candidate_id`, `relative_path`, `status` | Inventory identity; do not edit identities/paths. Status cannot override protection. |
| `label` | `0` camera/conventional graphic or `1` generated/edit; must match declared class folder. |
| `domain` | Shared coarse content category, e.g. `people`, `objects`, `landscape`; both classes required per primary category. |
| `source` | `real` for cameras, asserted generator for generated images; never use `real` for fakes. |
| `origin_kind` | `camera`, `fully_generated`, `ai_edit`, or `conventional_graphic`. The latter two are challenge-only. |
| `device` | Actual device description if known; blank is an explicitly unknown device, not inferred from image style. |
| `processing` | Actual known processing, e.g. `camera_original`, `whatsapp_shared`, `ordinary_edit`, `original_generator_export`. Unknown processing needs investigation. |
| `group_id` | Same ID for bursts, the same scene/session variants, copies, original/edit pairs, or related prompt/reference variants. Namespace unrelated scenes clearly. |
| `prior_use` | `new`, `previously_used`, or `related_to_prior`. “New” means no earlier training, tuning, calibration or evaluated collection use. |
| `approved` | `yes` only after reviewing provenance claim, fields, prior use and related-image hints. Approval is a curator assertion, not automated certification. |
| `notes` | Permissions/licenses, editing details, exact model version, source records and unresolved qualifications. |

For independent scenes, use distinct group IDs. Do not give every burst frame a
different ID to inflate group counts. Annotate a reserved old image's group ID
when a new variant shares its scene: protection then propagates to that group.
Exact bytes/RGB also merge groups automatically. Disclosure of one prior-used
component member excludes the entire component. Original/edit pairs stay
together in the challenge manifest, preventing the original from entering main
training while its edited version appears in the challenge set.

## Split protocol and outputs

Default group fractions are 50% training, 15% validation, 20% calibration and 15%
test. Assignment is deterministic for the seed, stratified by content domain and
class, and never divides an exact/annotated component across roles. At least four
independent primary groups per domain/class stratum are required. Small strata
are rounded to retain each role, so exact fractions are not promised.

Outputs:

- `manifest_train.csv`: fitting weights and training normalization only.
- `manifest_validation.csv`: choose hyperparameters/checkpoints, not thresholds.
- `manifest_calibration.csv`: only independent REAL representatives set FPR thresholds.
- `manifest_test.csv`: frozen held-out evaluation, no fitting/tuning.
- `manifest_challenge.csv`: optional graphics/AI-edit components; mixed-label groups
  require a separate grouped evaluation, not the current IID representative code.
- `split_plan.json`: frozen identities, seed, role counts, class/domain/source/
  format composition, explicit exclusions and limitations.

At alpha 0.05, at least 19 independent real calibration groups are needed merely
for a finite conformal threshold. That is NOT a reliable deployment guarantee;
coverage needs exchangeability, and a small real holdout has wide uncertainty.
The builder checks this minimum without scoring images. Test-only metrics and
source/model selection must not feed back into choosing splits or seeds.

These role manifests are compatible with the extraction manifest schema. The
current legacy training entry point RANDOMLY RE-SPLITS one development cache;
do not silently pass these roles through it or claim the role boundaries were
honored. The next training-stage integration must load these explicit roles,
share them between regularized and fusion candidates, and keep model randomness
separate from split randomness. That integration/training is not part of intake.

On Kaggle, private data must be intentionally attached by the user. Local
absolute source paths do not exist there; regenerate inventory on the attached
root, preserving reviewed relative-file/group annotations after verifying byte
identities. Never upload private manifests or photos automatically.

## Explicit reuse of evaluated generator collections

When the owner authorizes using earlier evaluated generator collections for
development, use `select-development` rather than editing reserved exclusions:

```bash
outputs/photo_eval_env_01/bin/python scripts/17_prepare_data.py select-development \
  --inventory outputs/data_intake_01/batch_05/inventory.json \
  --generator-dir 'chatgpt_unknown_version:AI generated/chatgpt_images' \
  --generator-dir 'nano_banana_unknown_version:AI generated/nano_banana' \
  --per-generator 50 --seed 0 --fully-generated \
  --out outputs/modern_development_01
```

Folder mappings are explicit origin assertions. `--fully-generated` requires an
owner assertion that the images were generated from scratch. Versions remain
unknown unless supported by source records. The selection checks input identity,
deduplicates exact decoded pixels, and uses seed/source/pixel hashes for ranking,
not detector scores. It excludes older real-photo diagnostics and loose copies
outside the designated generator directories. Selected generated files retain
`previously_used` history. No image is scored, copied, moved or uploaded.

Review `review.csv`, preserve it, and save a NEW reviewed copy. Use the origin,
processing and scene/prompt assertions supplied by the curator; explicitly retain
unknown devices/model versions and qualifications. A broad `mixed_user_collection`
domain is acceptable for a pilot, but does not establish matched semantic content
or device distributions. Independence is an assumption, not proven by unique
filenames/prompts. AI edits or conventional graphics are separate challenge items.

```bash
outputs/photo_eval_env_01/bin/python scripts/17_prepare_data.py build-development \
  --selection outputs/modern_development_01/selection.json \
  --review outputs/modern_development_01/reviewed_01.csv \
  --out outputs/modern_development_01/roles_01 --seed 0 --alpha 0.05
```

This generates fixed, disjoint train/validation/calibration/internal-diagnostic
manifests, stratified by domain/class/source. Related components stay together.
The nominal `manifest_test.csv` is explicitly marked **internal diagnostic**, not
an untouched confirmation test. Leftover images from the same previously
evaluated collection are not a fresh confirmation set either. Future final claims
need a genuinely new, independently collected evaluation set. All split-plan
records must travel with the manifests; a CSV alone omits the prior-use caveats.

The default fresh-only `build` command remains unchanged. The legacy trainer
still re-splits one cache. Use `scripts/18_train_development.py` for this protocol;
do not launch the legacy trainer on the combined selected collection.

## Fixed-role replay training

The explicit-role development workflow verifies manifest checksums, selection
identity, raw image hashes, broad-cache hashes, pinned extractor configuration,
feature alignment and cross-role identity/group overlap. It extracts locally,
offline, without uploading private images. Its CPU extraction uses the recorded
slow processor and JPEG-quality-90 policy. CPU/CUDA wheel suffix compatibility
does not imply bitwise device equivalence; earlier serving-parity checks remain
the relevant evidence for the starting model.

Run `scripts/18_train_development.py --help` for its `extract` and `compare`
stages. Required inputs are the roles directory, frozen selection (extraction),
trusted baseline bundle, recovered broad-cache archive and NEW private output
directories. Preserve receipts; never overwrite completed runs.

Comparison warm-starts copies of the original fusion head. Frozen backbones and
normalization remain unchanged. Every broad training row is replayed per epoch;
modern training rows are sampled with replacement. The objective is 75% mean
broad binary cross-entropy (BCE) plus 25% mean modern BCE. Repeated draws do not
increase the independent sample count. The same weighting applies to validation
BCE; the unchanged starting checkpoint is also eligible. Fifteen epochs, learning
rate 0.0001, weight decay 0.0001 and batch size 256 are fixed before diagnostics.
Seed 0 is the prespecified primary candidate; seeds 1/2 assess stability, not a
best-seed search. This is head adaptation, not backbone training from scratch.

Broad calibration and local real calibration are kept separate, never pooled.
Only real calibration rows set each threshold. The local threshold is stored in
the candidate's `calibrator.json`; `broad_calibrator.json` is a separate reference
threshold. Local calibration with only 21 real rows is a pilot, not deployment
certification. Both thresholds are reported on diagnostics rather than chosen
from test outcomes. Generated internal-test images were previously inspected.
Original broad test partitions are also an already-inspected diagnostic. No
live app or serving checkpoint is changed by this workflow. Candidate artifacts
need genuinely new confirmation data and a separate serving validation decision
before any promotion.
