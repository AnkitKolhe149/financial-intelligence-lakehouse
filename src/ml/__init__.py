"""Unsupervised anomaly detection over Silver transactions.

**Why unsupervised.** In a real oversight environment, labelled fraud is scarce,
arrives months late, and covers only the fraud that was *caught*. A supervised
model trained on it learns to find the fraud that was already being found.
Isolation Forest needs no labels and flags statistically unusual activity, which
is the honest description of what the platform can actually detect.

**Labels are still used - for evaluation only.** The generator writes a held-out
ground-truth file (investigation outcomes). It is never a feature and never
touches training; it is joined afterwards to report precision@k and PR-AUC, so
the model's usefulness is measured rather than asserted.

**Temporal integrity is enforced, not hoped for.** Transaction data is a time
series. Every historical feature is computed from strictly prior rows, and the
train/validation/test split is chronological. Both rules are enforced in code
(see :mod:`src.ml.features`) because leakage in fraud modelling produces
excellent offline metrics and a useless production model.
"""

from src.ml.anomaly_model import AnomalyModel, ScoringResult, train_anomaly_model
from src.ml.features import FEATURE_COLUMNS, build_features, temporal_split
from src.ml.model_registry import ModelRegistry

__all__ = [
    "FEATURE_COLUMNS",
    "AnomalyModel",
    "ModelRegistry",
    "ScoringResult",
    "build_features",
    "temporal_split",
    "train_anomaly_model",
]
