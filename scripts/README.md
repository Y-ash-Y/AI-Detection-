# Pipeline commands

Run from the repository root in the appropriate environment. Each entry point
has `--help`; inputs and output locations vary by stage. Use a new output path
for new experiments and keep generated files private.

| Workflow | Entry points |
| --- | --- |
| Assemble and review data | [build_manifest.py](build_manifest.py), [17_prepare_data.py](17_prepare_data.py) |
| Audit decoded files and roles | [13_audit_raw_data.py](13_audit_raw_data.py) |
| Extract aligned features | [01_extract_features.py](01_extract_features.py) |
| Fit a v3 serving bundle | [train_deploy.py](train_deploy.py) |
| Evaluate frozen model and threshold on external caches | [eval_testset.py](eval_testset.py) |
| Compare recovered heads and reconcile the results | [10_evaluate_recovered.py](10_evaluate_recovered.py), [11_validate_evaluation.py](11_validate_evaluation.py), [12_recalibrate_evaluation.py](12_recalibrate_evaluation.py) |
| Package or verify a run | [14_package_kaggle.py](14_package_kaggle.py), [15_check_serving_parity.py](15_check_serving_parity.py) |
| Private photo diagnostics or candidate development | [16_evaluate_real_photos.py](16_evaluate_real_photos.py), [18_train_development.py](18_train_development.py) |
| Stage a minimal Space release locally | [19_prepare_space.py](19_prepare_space.py) |

Earlier protocol scripts [02](02_train_baseline.py) and [03](03_train_fusion.py)
fit baseline and fusion heads. [04](04_score_and_calibrate.py) exports scores
and group identities; [05](05_shift_experiment.py) uses disjoint calibration and
test groups to compare thresholds. Older score archives without group identities
need a new export. [06](06_real_domain_shift.py), [07](07_few_shot_recovery.py),
[08](08_shift_generality.py) and [09](09_frontier_fewshot.py) explore shift,
few-shot recovery and cross-generator behavior. Treat their outputs as separate
experiments, not updates to the deployed model.

Start without data with the [offline example](../examples/README.md). For the
recorded fresh benchmark use the [Kaggle guide](../notebooks/README.md).
Staging and packaging do not upload anything or authenticate to an external
service. Review archive contents before transferring them.
