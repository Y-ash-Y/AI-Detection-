# Kaggle workflows

- [kaggle_fresh_evaluation.ipynb](kaggle_fresh_evaluation.ipynb) is the fresh
  evaluation notebook. It requires the appropriate reviewed dataset inputs and
  source handoff; follow its setup and integrity checks before extraction.
- [kaggle_resume_pinned_smoke.py](kaggle_resume_pinned_smoke.py) is a targeted
  recovery cell for an earlier smoke run, not a standalone training workflow.
- [kaggle_serving_parity.py](kaggle_serving_parity.py) checks a fixed image sample
  against existing evaluation artifacts. It does not retrain or deploy.

Use the [private archive packager](../scripts/14_package_kaggle.py) only when
preparing a new handoff. Inspect inputs and outputs for private images and
identifiers before upload. The repository does not bundle benchmark data,
model weights or personal collections. Recovered legacy outputs are documented
separately from the [fresh evaluation](../reference/EVALUATION.md).
