"""The check abstraction.

Every quality check is an object with a name, a dimension, a severity and an
``evaluate`` method returning a :class:`CheckResult`. Making checks objects
rather than functions buys three things:

* **A registry.** Suites are declared as data (see :mod:`src.data_quality.suites`),
  so adding a check to a dataset does not mean editing pipeline code.
* **Uniform results.** Every check reports the same fields - records evaluated,
  records failed, observed value, threshold, pass/fail - which is what makes a
  single ``data_quality_checks`` table possible.
* **A severity contract.** ``blocking`` fails the pipeline; ``warning`` is
  recorded and execution continues; ``info`` is measurement only. Without an
  explicit severity, every check is implicitly blocking and the platform stops
  on trivia.

Checks return a **score in [0, 1]** as well as a boolean, because "99.2% of rows
have an account id" is a far more actionable statement than "the completeness
check failed".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd


class Dimension(StrEnum):
    """The six quality dimensions that make up the overall score."""

    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    INTEGRITY = "integrity"
    FRESHNESS = "freshness"
    VOLUME = "volume"
    #: Schema and distribution checks are recorded but do not carry weight in
    #: the headline score; they are early-warning signals, not health measures.
    SCHEMA = "schema"
    DISTRIBUTION = "distribution"


class CheckSeverity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


@dataclass
class CheckResult:
    """The outcome of one check against one dataset."""

    check_name: str
    dimension: Dimension
    severity: CheckSeverity
    passed: bool
    score: float
    records_evaluated: int
    records_failed: int
    message: str
    column_name: str | None = None
    threshold: float | None = None
    observed_value: float | None = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)

    def to_row(self, run_id: str, dataset: str) -> dict[str, Any]:
        """The shape stored in ``data_quality_checks``."""
        return {
            "run_id": run_id,
            "dataset": dataset,
            "check_name": self.check_name,
            "dimension": str(self.dimension),
            "column_name": self.column_name,
            "severity": str(self.severity),
            "passed": bool(self.passed),
            "score": round(float(self.score), 6),
            "records_evaluated": int(self.records_evaluated),
            "records_failed": int(self.records_failed),
            "threshold": self.threshold,
            "observed_value": self.observed_value,
            "message": self.message,
            "evaluated_at": self.evaluated_at,
        }

    @property
    def score_pct(self) -> float:
        return round(self.score * 100, 2)


@dataclass
class CheckContext:
    """Everything a check may need beyond the frame itself.

    Passing a context rather than a pile of keyword arguments means a new check
    that needs, say, the historical baseline does not change every other check's
    signature.
    """

    dataset: str
    layer: str = "silver"
    #: Key values of referenced dimensions, for integrity checks.
    references: dict[str, pd.Series] = field(default_factory=dict)
    #: Prior runs of this dataset, for volume and distribution comparisons.
    history: pd.DataFrame | None = None
    #: A previously-captured numeric profile, for drift detection.
    baseline: dict[str, Any] = field(default_factory=dict)
    settings: Any = None


class DataQualityCheck(ABC):
    """Base class for every check."""

    #: Human-readable, stable. Stored in the metadata table and used in alerts,
    #: so renaming one breaks historical comparison - treat as an identifier.
    name: str = "unnamed_check"
    dimension: Dimension = Dimension.VALIDITY
    severity: CheckSeverity = CheckSeverity.WARNING
    #: Minimum score for the check to pass, in [0, 1].
    threshold: float = 1.0

    def __init__(
        self,
        *,
        name: str | None = None,
        severity: CheckSeverity | None = None,
        threshold: float | None = None,
    ) -> None:
        if name:
            self.name = name
        if severity:
            self.severity = severity
        if threshold is not None:
            self.threshold = threshold

    @abstractmethod
    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        """Run the check and return its result."""

    # -- helpers for subclasses --------------------------------------------

    def _result(
        self,
        *,
        passed: bool,
        score: float,
        records_evaluated: int,
        records_failed: int,
        message: str,
        column_name: str | None = None,
        observed_value: float | None = None,
        **details: Any,
    ) -> CheckResult:
        return CheckResult(
            check_name=self.name,
            dimension=self.dimension,
            severity=self.severity,
            passed=passed,
            score=max(0.0, min(1.0, float(score))),
            records_evaluated=int(records_evaluated),
            records_failed=int(records_failed),
            message=message,
            column_name=column_name,
            threshold=self.threshold,
            observed_value=observed_value,
            details=details,
        )

    def _skip(self, message: str, column_name: str | None = None) -> CheckResult:
        """A check that could not run.

        Scored as 1.0 and passing, but flagged in details. A skipped check must
        not silently drag the score down - that would make an absent optional
        column indistinguishable from a genuine quality problem - but it is
        recorded so nobody mistakes 'not run' for 'passed'.
        """
        return CheckResult(
            check_name=self.name,
            dimension=self.dimension,
            severity=CheckSeverity.INFO,
            passed=True,
            score=1.0,
            records_evaluated=0,
            records_failed=0,
            message=f"SKIPPED: {message}",
            column_name=column_name,
            threshold=self.threshold,
            observed_value=None,
            details={"skipped": True},
        )
