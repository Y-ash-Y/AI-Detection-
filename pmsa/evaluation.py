"""Frozen-threshold evaluation, separate from fitting and recalibration."""
import numpy as np

from .artifacts import validate_stream_metadata
from .data.splits import independent_indices
from .eval import full_report
from .eval.metrics import operating_report
from .features.cache import validate_alignment


def check_external_overlap(feature_set, development_ids):
    """Fail on observed overlap; this cannot detect unannotated near-duplicates."""
    for key in ("paths", "content_hashes", "related_groups"):
        values = getattr(feature_set, key)
        if values is None:
            continue
        target = {str(x) for x in values if str(x)}
        development = {str(x) for x in development_ids.get(key, []) if str(x)}
        if target & development:
            raise ValueError(f"External test overlaps development data: {key}")


def evaluate_cached(model, sets, calibrator, bundle, development_ids, *,
                    exploratory_legacy=False, bootstrap_n=2000, ci=.95, seed=0,
                    fpr_targets=(.01, .05), group_reference=None):
    validate_alignment(sets)
    if not bundle and not exploratory_legacy:
        raise ValueError("Evaluation needs an artifact bundle, or explicit exploratory legacy mode")
    names = [s.backbone for s in sets]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate evaluation streams")
    complete = all(s.content_hashes is not None and np.all(s.content_hashes != "")
                   and s.metadata.get("schema_version") == 1 for s in sets)
    if not complete and not exploratory_legacy:
        raise ValueError("Test cache lacks image hashes/extraction metadata")
    if bundle:
        if names != list(bundle["streams"]):
            raise ValueError("Test streams differ from model bundle")
        for s in sets:
            if s.metadata and s.metadata != bundle["streams"][s.backbone]:
                raise ValueError(f"Test extraction configuration differs: {s.backbone}")
    if complete:
        validate_stream_metadata({s.backbone: s.metadata for s in sets})
    if not development_ids and not exploratory_legacy:
        raise ValueError("Development split identities are required for overlap checks")
    base = sets[0]
    if group_reference is not None:
        validate_alignment([group_reference, *sets])
    check_external_overlap(base, development_ids)
    reference = group_reference if group_reference is not None else base
    ix = independent_indices(base.labels, reference.groups())
    unique_sets = [s.select(ix) for s in sets]
    labels = unique_sets[0].labels
    features = [s.features for s in unique_sets]
    scores = model.score(features if hasattr(model, "specs") else features[0])
    ranking = (full_report(scores, labels, fpr_targets, bootstrap_n, ci, seed)
               if set(np.unique(labels)) == {0, 1} else None)
    report = {"protocol": "external-frozen-threshold-v3",
              "exploratory_legacy": not complete or not bool(bundle) or not bool(development_ids),
              "original_rows": len(base), "independent_representatives": len(ix),
              "ranking": ranking, "operating": operating_report(scores, labels, calibrator.tau, ci),
              "notes": ["No recalibration or fitting on this external test set.",
                        "Ranking TPR@FPR is descriptive and uses test-real thresholds; deployed rates use the frozen threshold.",
                        "Bootstrap/Wilson intervals assume independent representatives, not repeated crops/prompts.",
                        "Observed identity checks do not rule out unannotated near-duplicates.",
                        "Single-class sets cannot establish AUC, false-positive rate or overall generalization."]}
    return report, scores, labels, unique_sets[0].paths
