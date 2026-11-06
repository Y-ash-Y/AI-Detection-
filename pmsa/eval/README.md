# Evaluation metrics

[metrics.py](metrics.py) computes ranking and operating-point metrics and
bootstrap summaries. [shift_matrix.py](shift_matrix.py) compares real-domain
and generator shifts. [External evaluation](../evaluation.py) evaluates a
frozen model and threshold and checks observed overlap.

Keep AUC, test-derived TPR at a false-positive budget, and observed FPR/TPR at
a separately calibrated threshold distinct. A good ranking does not ensure a
useful threshold, and uncertainty intervals depend on the sampling protocol.
See the [recorded results and limits](../../reference/EVALUATION.md).
