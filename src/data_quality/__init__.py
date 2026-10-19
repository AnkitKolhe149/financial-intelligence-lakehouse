"""Data quality: measurement, not enforcement.

An important distinction runs through this package. :mod:`src.silver.validation`
*enforces* - it decides which records may proceed. This package *measures* - it
scores a dataset across six dimensions and records the result so that quality
becomes a time series rather than an opinion.

The two are separate because they answer different questions. Validation asks
"may this record proceed?"; quality asks "how healthy is this dataset, and is it
getting better or worse?" A dataset can be 100% valid (everything invalid was
quarantined) and still be unhealthy (40% of the day's records were quarantined).
Only the second question predicts an incident.

Dimensions scored: completeness, validity, uniqueness, integrity, freshness and
volume, combined into one weighted score with configurable weights.
"""

from src.data_quality.base_check import CheckResult, CheckSeverity, DataQualityCheck, Dimension
from src.data_quality.scoring import QualityScorecard, score_dataset
from src.data_quality.validator import DataQualityValidator, run_quality_suite

__all__ = [
    "CheckResult",
    "CheckSeverity",
    "DataQualityCheck",
    "DataQualityValidator",
    "Dimension",
    "QualityScorecard",
    "run_quality_suite",
    "score_dataset",
]
