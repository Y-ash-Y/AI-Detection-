# Data intake and splits

[manifest.py](manifest.py) records image paths, binary labels (0 real, 1 fake),
domains, sources and optional related-image groups.
[raw_audit.py](raw_audit.py) checks raw inputs and cross-role overlap;
[splits.py](splits.py) partitions by group so duplicates and related variants
do not straddle fitting, validation, calibration and test.
[genimage.py](genimage.py) handles GenImage layout;
[intake.py](intake.py) supports private review and role assignment.

Use [17_prepare_data.py](../../scripts/17_prepare_data.py) for reviewed intake
and [13_audit_raw_data.py](../../scripts/13_audit_raw_data.py) before extraction.
Hashes do not replace review for visually related images, mixed labels or
licensing. Raw images, manifests with private paths and review sheets stay local.
