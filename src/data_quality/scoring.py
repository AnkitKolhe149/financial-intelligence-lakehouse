"""Weighted data-quality scoring.

Individual check results are aggregated into one number per dimension, and the
six weighted dimensions into one headline score. The aggregation rules are
deliberate:

* **Within a dimension**, checks are averaged weighted by the number of records
  they evaluated. A check over 500,000 rows should count for more than one over
  a 36-row reference table; a plain mean would let a tiny dataset's perfect
  score paper over the fact table's problems.
* **Across dimensions**, configured weights apply (``quality.score_weights``).
  Completeness and validity carry the most weight because they most directly
  determine whether a number in a report is right.
* **Dimensions with no checks are dropped and the remaining weights are
  renormalised**, so a dataset without a timestamp is not penalised on freshness
  for having no timestamp to be stale.

A dimension that scored no checks is reported as ``None`` rather than 100%,
because "not measured" and "perfect" must never look the same on a dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.core.config import Settings, get_settings
from src.data_quality.base_check import CheckResult, CheckSeverity, Dimension

#: Dimensions that contribute to the headline score.
SCORED_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension.COMPLETENESS,
    Dimension.VALIDITY,
    Dimension.UNIQUENESS,
    Dimension.INTEGRITY,
    Dimension.FRESHNESS,
    Dimension.VOLUME,
)


@dataclass
class QualityScorecard:
    """The quality result for one dataset in one run."""

    dataset: str
    layer: str
    run_id: str
    overall_score: float
    dimension_scores: dict[str, float | None]
    checks: list[CheckResult] = field(default_factory=list)
    records_evaluated: int = 0
    records_failed: int = 0
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "PASS"

    @property
    def checks_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def blocking_failures(self) -> list[CheckResult]:
        """Failed checks that are allowed to stop a pipeline."""
        return [c for c in self.checks if not c.passed and c.severity == CheckSeverity.BLOCKING]

    def to_result_row(self) -> dict[str, Any]:
        """The shape stored in ``data_quality_results``."""
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "layer": self.layer,
            "evaluated_at": self.evaluated_at,
            "overall_score": round(self.overall_score, 4),
            "completeness": _round(self.dimension_scores.get("completeness")),
            "validity": _round(self.dimension_scores.get("validity")),
            "uniqueness": _round(self.dimension_scores.get("uniqueness")),
            "integrity": _round(self.dimension_scores.get("integrity")),
            "freshness": _round(self.dimension_scores.get("freshness")),
            "volume": _round(self.dimension_scores.get("volume")),
            "records_evaluated": self.records_evaluated,
            "records_failed": self.records_failed,
            "checks_total": len(self.checks),
            "checks_passed": self.checks_passed,
            "status": self.status,
        }

    def to_check_rows(self) -> list[dict[str, Any]]:
        return [check.to_row(self.run_id, self.dataset) for check in self.checks]

    def report(self) -> str:
        """Human-readable scorecard, printed by the demo and the CLI."""
        lines = [
            f"Data Quality - {self.dataset} ({self.layer})",
            f"Run: {self.run_id}",
            "-" * 46,
        ]
        for dimension in SCORED_DIMENSIONS:
            score = self.dimension_scores.get(str(dimension))
            rendered = "not measured" if score is None else f"{score:6.2f}%"
            lines.append(f"  {str(dimension).capitalize():<14} {rendered}")
        lines.append("-" * 46)
        lines.append(f"  {'Overall':<14} {self.overall_score:6.2f}%   [{self.status}]")
        lines.append(f"  {'Checks':<14} {self.checks_passed}/{len(self.checks)} passed")
        if self.blocking_failures:
            lines.append("  Blocking failures:")
            lines.extend(f"    - {c.check_name}: {c.message}" for c in self.blocking_failures)
        return "\n".join(lines)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def score_dataset(
    dataset: str,
    checks: list[CheckResult],
    run_id: str,
    *,
    layer: str = "silver",
    records_evaluated: int = 0,
    settings: Settings | None = None,
) -> QualityScorecard:
    """Aggregate check results into a weighted scorecard."""
    settings = settings or get_settings()
    weights = settings.quality.score_weights

    dimension_scores: dict[str, float | None] = {}
    for dimension in SCORED_DIMENSIONS:
        relevant = [
            c for c in checks if c.dimension == dimension and not c.details.get("skipped", False)
        ]
        dimension_scores[str(dimension)] = _weighted_dimension_score(relevant)

    # Renormalise over the dimensions that were actually measured.
    measured = {name: score for name, score in dimension_scores.items() if score is not None}
    active_weight = sum(weights.get(name, 0.0) for name in measured)
    if measured and active_weight > 0:
        overall = (
            sum(score * weights.get(name, 0.0) for name, score in measured.items()) / active_weight
        )
    else:
        overall = 100.0

    thresholds = settings.quality.thresholds
    if overall < thresholds.overall_fail:
        status = "FAIL"
    elif overall < thresholds.overall_warn:
        status = "WARN"
    else:
        status = "PASS"

    # A blocking check that failed downgrades the status regardless of the
    # headline number: 99.4% overall is not reassuring if every transaction id
    # is duplicated.
    if any(not c.passed and c.severity == CheckSeverity.BLOCKING for c in checks):
        status = "FAIL" if status != "FAIL" else status

    records_failed = sum(c.records_failed for c in checks)

    return QualityScorecard(
        dataset=dataset,
        layer=layer,
        run_id=run_id,
        overall_score=round(overall, 4),
        dimension_scores=dimension_scores,
        checks=checks,
        records_evaluated=records_evaluated,
        records_failed=records_failed,
        status=status,
    )


def _weighted_dimension_score(checks: list[CheckResult]) -> float | None:
    """Record-count-weighted mean of a dimension's checks, as a percentage."""
    if not checks:
        return None
    total_records = sum(max(c.records_evaluated, 0) for c in checks)
    if total_records == 0:
        return round(sum(c.score for c in checks) / len(checks) * 100, 4)
    weighted = sum(c.score * max(c.records_evaluated, 0) for c in checks)
    return round(weighted / total_records * 100, 4)


def aggregate_platform_score(scorecards: list[QualityScorecard]) -> float:
    """One number for the whole platform, weighted by records evaluated.

    This is the figure the dashboard headline and the demo summary show. It is
    weighted rather than averaged for the same reason as within a dimension: the
    transaction fact table's health is what matters most.
    """
    if not scorecards:
        return 100.0
    total = sum(max(card.records_evaluated, 1) for card in scorecards)
    return round(
        sum(card.overall_score * max(card.records_evaluated, 1) for card in scorecards) / total,
        2,
    )
