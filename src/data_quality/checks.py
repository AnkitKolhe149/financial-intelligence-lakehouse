"""Concrete data-quality checks.

Nine check types covering the six scored dimensions plus schema and
distribution monitoring. Each is vectorised and each reports a graded score
rather than a bare pass/fail, so a dataset that is 99.1% complete is
distinguishable from one that is 62% complete - both "fail" a strict threshold,
but only one is an emergency.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data_quality.base_check import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    DataQualityCheck,
    Dimension,
)

# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------


class CompletenessCheck(DataQualityCheck):
    """Required columns must be populated.

    Scored as the proportion of populated cells across the checked columns, so
    one column being 50% null in a ten-column check does not read the same as
    every column being 95% populated.
    """

    dimension = Dimension.COMPLETENESS
    severity = CheckSeverity.BLOCKING

    def __init__(self, columns: list[str], threshold: float = 0.99, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.columns = columns
        self.name = kwargs.get("name") or f"completeness_{'_'.join(columns[:3])}"
        self.threshold = threshold

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        present = [c for c in self.columns if c in frame.columns]
        if not present:
            return self._skip(f"none of {self.columns} present in {context.dataset}")
        if frame.empty:
            return self._skip("dataset is empty")

        total_cells = len(frame) * len(present)
        null_cells = int(frame[present].isna().sum().sum())
        score = 1.0 - (null_cells / total_cells) if total_cells else 1.0
        # A row is "failed" if any checked column is null.
        failed_rows = int(frame[present].isna().any(axis=1).sum())

        per_column = {c: float(1 - frame[c].isna().mean()) for c in present}
        worst = min(per_column, key=per_column.get) if per_column else None

        return self._result(
            passed=score >= self.threshold,
            score=score,
            records_evaluated=len(frame),
            records_failed=failed_rows,
            observed_value=round(score * 100, 4),
            column_name=worst,
            message=(
                f"{score * 100:.2f}% of required cells populated across {len(present)} columns"
                + (f"; weakest column '{worst}' at {per_column[worst] * 100:.2f}%" if worst else "")
            ),
            per_column={k: round(v, 6) for k, v in per_column.items()},
        )


# ---------------------------------------------------------------------------
# uniqueness
# ---------------------------------------------------------------------------


class UniquenessCheck(DataQualityCheck):
    """A business key must identify exactly one record."""

    dimension = Dimension.UNIQUENESS
    severity = CheckSeverity.BLOCKING

    def __init__(self, column: str, threshold: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.column = column
        self.name = kwargs.get("name") or f"uniqueness_{column}"
        self.threshold = threshold

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        if self.column not in frame.columns:
            return self._skip(f"column '{self.column}' absent", self.column)
        if frame.empty:
            return self._skip("dataset is empty", self.column)

        populated = frame[self.column].dropna()
        duplicates = int(populated.duplicated().sum())
        score = 1.0 - (duplicates / len(populated)) if len(populated) else 1.0

        offenders = populated.value_counts().head(5).to_dict() if duplicates else {}
        return self._result(
            passed=score >= self.threshold,
            score=score,
            records_evaluated=len(populated),
            records_failed=duplicates,
            column_name=self.column,
            observed_value=float(duplicates),
            message=(
                f"{duplicates:,} duplicate values in '{self.column}' "
                f"({score * 100:.4f}% unique)"
            ),
            top_offenders={str(k): int(v) for k, v in offenders.items()},
        )


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------


class RangeCheck(DataQualityCheck):
    """A numeric column must fall inside an allowed range."""

    dimension = Dimension.VALIDITY
    severity = CheckSeverity.BLOCKING

    def __init__(
        self,
        column: str,
        minimum: float | None = None,
        maximum: float | None = None,
        *,
        exclusive_min: bool = False,
        threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.column = column
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_min = exclusive_min
        self.name = kwargs.get("name") or f"range_{column}"
        self.threshold = threshold

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        if self.column not in frame.columns:
            return self._skip(f"column '{self.column}' absent", self.column)
        values = pd.to_numeric(frame[self.column], errors="coerce").dropna()
        if values.empty:
            return self._skip(f"column '{self.column}' has no numeric values", self.column)

        violations = pd.Series(False, index=values.index)
        if self.minimum is not None:
            violations |= (
                (values <= self.minimum) if self.exclusive_min else (values < self.minimum)
            )
        if self.maximum is not None:
            violations |= values > self.maximum

        failed = int(violations.sum())
        score = 1.0 - failed / len(values)
        bounds = f"[{self.minimum}, {self.maximum}]"
        return self._result(
            passed=score >= self.threshold,
            score=score,
            records_evaluated=len(values),
            records_failed=failed,
            column_name=self.column,
            observed_value=float(values.min()) if failed else None,
            message=f"{failed:,} of {len(values):,} values outside {bounds} in '{self.column}'",
            minimum=float(values.min()),
            maximum=float(values.max()),
            mean=float(values.mean()),
        )


class AllowedValuesCheck(DataQualityCheck):
    """A column's values must belong to a closed set."""

    dimension = Dimension.VALIDITY
    severity = CheckSeverity.WARNING

    def __init__(
        self, column: str, allowed: list[str], threshold: float = 0.99, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.column = column
        self.allowed = {str(v).upper() for v in allowed}
        self.name = kwargs.get("name") or f"allowed_values_{column}"
        self.threshold = threshold

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        if self.column not in frame.columns:
            return self._skip(f"column '{self.column}' absent", self.column)
        values = frame[self.column].dropna().astype("string").str.upper()
        if values.empty:
            return self._skip(f"column '{self.column}' is entirely null", self.column)

        offending = ~values.isin(self.allowed)
        failed = int(offending.sum())
        score = 1.0 - failed / len(values)
        unexpected = sorted(set(values[offending].tolist()))[:10]

        return self._result(
            passed=score >= self.threshold,
            score=score,
            records_evaluated=len(values),
            records_failed=failed,
            column_name=self.column,
            observed_value=float(failed),
            message=(
                f"{failed:,} values in '{self.column}' outside the approved set"
                + (f": {', '.join(unexpected[:5])}" if unexpected else "")
            ),
            unexpected_values=unexpected,
        )


class TimestampSanityCheck(DataQualityCheck):
    """Timestamps must be parseable and not in the future."""

    dimension = Dimension.VALIDITY
    severity = CheckSeverity.BLOCKING

    def __init__(self, column: str, threshold: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.column = column
        self.name = kwargs.get("name") or f"timestamp_sanity_{column}"
        self.threshold = threshold

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        if self.column not in frame.columns or frame.empty:
            return self._skip(f"column '{self.column}' absent or dataset empty", self.column)

        values = pd.to_datetime(frame[self.column], errors="coerce", utc=True)
        unparseable = int(values.isna().sum() - frame[self.column].isna().sum())
        future = int((values > pd.Timestamp(datetime.now(UTC))).sum())
        failed = max(unparseable, 0) + future
        score = 1.0 - failed / len(frame)

        return self._result(
            passed=score >= self.threshold,
            score=score,
            records_evaluated=len(frame),
            records_failed=failed,
            column_name=self.column,
            observed_value=float(future),
            message=(
                f"'{self.column}': {max(unparseable, 0):,} unparseable, {future:,} in the future"
            ),
            earliest=str(values.min()),
            latest=str(values.max()),
        )


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------


class ReferentialIntegrityCheck(DataQualityCheck):
    """Foreign keys must resolve against the referenced dimension."""

    dimension = Dimension.INTEGRITY
    severity = CheckSeverity.BLOCKING

    def __init__(
        self,
        column: str,
        reference_table: str,
        threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.column = column
        self.reference_table = reference_table
        self.name = kwargs.get("name") or f"referential_{column}"
        self.threshold = threshold

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        if self.column not in frame.columns:
            return self._skip(f"column '{self.column}' absent", self.column)
        reference = context.references.get(self.reference_table)
        if reference is None or reference.empty:
            return self._skip(
                f"reference dataset '{self.reference_table}' unavailable", self.column
            )

        values = frame[self.column].dropna().astype("string")
        if values.empty:
            return self._skip(f"column '{self.column}' is entirely null", self.column)

        known = set(reference.dropna().astype(str))
        orphans = ~values.isin(known)
        failed = int(orphans.sum())
        score = 1.0 - failed / len(values)
        examples = sorted(set(values[orphans].tolist()))[:5]

        return self._result(
            passed=score >= self.threshold,
            score=score,
            records_evaluated=len(values),
            records_failed=failed,
            column_name=self.column,
            observed_value=float(failed),
            message=(
                f"{failed:,} values in '{self.column}' do not exist in " f"{self.reference_table}"
            ),
            reference_table=self.reference_table,
            examples=examples,
        )


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------


class FreshnessCheck(DataQualityCheck):
    """The newest record must be recent enough to be useful.

    Scored as a linear decay past the threshold rather than a cliff: data that
    is one hour late is not as broken as data that is a week late, and the score
    should say so.
    """

    dimension = Dimension.FRESHNESS
    severity = CheckSeverity.WARNING

    def __init__(self, column: str, max_age_hours: float = 48, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.column = column
        self.max_age_hours = max_age_hours
        self.name = kwargs.get("name") or f"freshness_{column}"
        self.threshold = 1.0

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        if self.column not in frame.columns or frame.empty:
            return self._skip(f"column '{self.column}' absent or dataset empty", self.column)

        values = pd.to_datetime(frame[self.column], errors="coerce", utc=True).dropna()
        if values.empty:
            return self._skip(f"column '{self.column}' has no valid timestamps", self.column)

        latest = values.max()
        age_hours = (pd.Timestamp(datetime.now(UTC)) - latest).total_seconds() / 3600
        if age_hours <= self.max_age_hours:
            score = 1.0
        else:
            # Full credit at the threshold, zero at four times the threshold.
            overshoot = (age_hours - self.max_age_hours) / (self.max_age_hours * 3)
            score = max(0.0, 1.0 - overshoot)

        stale = int((values < latest - pd.Timedelta(days=365)).sum())
        return self._result(
            passed=age_hours <= self.max_age_hours,
            score=score,
            records_evaluated=len(values),
            records_failed=0 if age_hours <= self.max_age_hours else len(values),
            column_name=self.column,
            observed_value=round(age_hours, 2),
            message=(
                f"newest record is {age_hours:.1f}h old "
                f"(threshold {self.max_age_hours:.0f}h); latest={latest:%Y-%m-%d %H:%M}"
            ),
            latest=str(latest),
            earliest=str(values.min()),
            older_than_a_year=stale,
        )


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------


class VolumeCheck(DataQualityCheck):
    """Daily record counts must not deviate wildly from their trailing median.

    Two details make this check usable rather than noisy:

    **Median, not mean.** A single outlier day (the September fiscal-year-end
    surge) would drag a mean-based baseline upward and then mask a genuine drop
    the following week.

    **Count-aware tolerance.** A flat "±40%" rule is statistically wrong at low
    volumes: if a dataset averages 14 records a day, ordinary Poisson noise
    produces ±40% swings constantly and the check fires every run. Below
    ``min_median_for_pct`` the tolerance therefore widens to a Poisson band
    (median ± k·sqrt(median)), which is the actual expected variation of a count
    process. Above it, the configured percentage applies. Without this, the
    check would be technically correct and operationally useless - the classic
    way monitoring gets switched off.
    """

    dimension = Dimension.VOLUME
    severity = CheckSeverity.WARNING

    def __init__(
        self,
        date_column: str,
        max_deviation_pct: float = 40.0,
        min_days: int = 5,
        min_median_for_pct: float = 50.0,
        poisson_sigma: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.date_column = date_column
        self.max_deviation_pct = max_deviation_pct
        self.min_days = min_days
        #: Below this median daily volume, percentage deviation is meaningless.
        self.min_median_for_pct = min_median_for_pct
        self.poisson_sigma = poisson_sigma
        self.name = kwargs.get("name") or "volume_daily_counts"
        self.threshold = 0.9

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        if self.date_column not in frame.columns or frame.empty:
            return self._skip(f"column '{self.date_column}' absent or dataset empty")

        dates = pd.to_datetime(frame[self.date_column], errors="coerce", utc=True).dt.date
        counts = dates.value_counts().sort_index()
        if len(counts) < self.min_days:
            return self._skip(
                f"only {len(counts)} distinct days; need {self.min_days} for a baseline"
            )

        median = float(counts.median())
        if median <= 0:
            return self._skip("median daily volume is zero")

        deviation_pct = (counts - median).abs() / median * 100

        if median >= self.min_median_for_pct:
            tolerance = median * self.max_deviation_pct / 100
            regime = f"±{self.max_deviation_pct:.0f}%"
        else:
            # Poisson: variance equals the mean, so the standard deviation of a
            # daily count around `median` is sqrt(median).
            tolerance = max(self.poisson_sigma * np.sqrt(median), median * 0.5)
            regime = f"±{self.poisson_sigma:.0f} sigma Poisson (low-volume regime)"

        anomalous = (counts - median).abs() > tolerance
        anomalous_days = counts[anomalous]
        score = 1.0 - len(anomalous_days) / len(counts)

        worst_day = deviation_pct.idxmax()
        return self._result(
            passed=score >= self.threshold,
            score=score,
            records_evaluated=int(counts.sum()),
            records_failed=int(anomalous_days.sum()) if len(anomalous_days) else 0,
            column_name=self.date_column,
            observed_value=round(float(deviation_pct.max()), 2),
            message=(
                f"{len(anomalous_days)} of {len(counts)} days outside {regime} "
                f"of the median {median:,.0f}/day; worst {worst_day} "
                f"at {deviation_pct.max():.0f}% deviation"
            ),
            median_daily_volume=median,
            tolerance=round(float(tolerance), 2),
            regime=regime,
            days_evaluated=len(counts),
            anomalous_days=[str(d) for d in anomalous_days.index[:10]],
        )


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


class SchemaCheck(DataQualityCheck):
    """The dataset must carry the expected columns.

    Missing expected columns fail. **Unexpected** columns are reported but do
    not fail: additive evolution is normal and Bronze is deliberately permissive.
    A platform that fails a load because the source added a field is a platform
    people route around.
    """

    dimension = Dimension.SCHEMA
    severity = CheckSeverity.BLOCKING

    def __init__(self, expected_columns: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.expected_columns = expected_columns
        self.name = kwargs.get("name") or "schema_conformance"
        self.threshold = 1.0

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        actual = set(frame.columns)
        expected = set(self.expected_columns)
        missing = sorted(expected - actual)
        unexpected = sorted(c for c in actual - expected if not c.startswith("_"))

        score = 1.0 - len(missing) / len(expected) if expected else 1.0
        return self._result(
            passed=not missing,
            score=score,
            records_evaluated=len(frame),
            records_failed=0,
            observed_value=float(len(missing)),
            message=(
                f"{len(missing)} expected columns missing, "
                f"{len(unexpected)} unexpected columns present"
                + (f"; missing: {', '.join(missing[:5])}" if missing else "")
            ),
            missing_columns=missing,
            unexpected_columns=unexpected,
        )


# ---------------------------------------------------------------------------
# distribution drift
# ---------------------------------------------------------------------------


class DistributionCheck(DataQualityCheck):
    """Population Stability Index against a stored baseline.

    PSI is the standard measure for "has this distribution moved?" in credit and
    fraud modelling. Conventional reading: < 0.1 stable, 0.1-0.25 moderate
    shift, > 0.25 significant shift. A significant shift in transaction amounts
    is exactly the signal that an anomaly model trained last month is no longer
    calibrated - which makes this a model-monitoring check as much as a data one.
    """

    dimension = Dimension.DISTRIBUTION
    severity = CheckSeverity.INFO

    def __init__(
        self,
        column: str,
        psi_warn: float = 0.10,
        psi_fail: float = 0.25,
        bins: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.column = column
        self.psi_warn = psi_warn
        self.psi_fail = psi_fail
        self.bins = bins
        self.name = kwargs.get("name") or f"distribution_psi_{column}"
        self.threshold = psi_fail

    def evaluate(self, frame: pd.DataFrame, context: CheckContext) -> CheckResult:
        if self.column not in frame.columns:
            return self._skip(f"column '{self.column}' absent", self.column)

        current = pd.to_numeric(frame[self.column], errors="coerce").dropna()
        if len(current) < 100:
            return self._skip("fewer than 100 values; PSI is not meaningful", self.column)

        baseline = context.baseline.get(self.column)
        if not baseline:
            # First run: record the profile so future runs have something to
            # compare against, and report the check as informational.
            return CheckResult(
                check_name=self.name,
                dimension=self.dimension,
                severity=CheckSeverity.INFO,
                passed=True,
                score=1.0,
                records_evaluated=len(current),
                records_failed=0,
                message="baseline captured; PSI available from the next run",
                column_name=self.column,
                details={"baseline_captured": True, "profile": profile_numeric(current)},
            )

        psi = population_stability_index(
            np.asarray(baseline.get("quantiles", [])),
            np.asarray(baseline.get("proportions", [])),
            current.to_numpy(),
        )
        verdict = (
            "stable"
            if psi < self.psi_warn
            else ("moderate shift" if psi < self.psi_fail else "significant shift")
        )

        return self._result(
            passed=psi < self.psi_fail,
            score=max(0.0, 1.0 - psi / max(self.psi_fail, 1e-9)),
            records_evaluated=len(current),
            records_failed=0,
            column_name=self.column,
            observed_value=round(float(psi), 6),
            message=f"PSI for '{self.column}' is {psi:.4f} ({verdict})",
            psi=round(float(psi), 6),
            verdict=verdict,
        )


def profile_numeric(values: pd.Series, bins: int = 10) -> dict[str, Any]:
    """Capture a distribution profile suitable for later PSI comparison."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {}
    quantiles = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)))
    counts, _ = np.histogram(clean, bins=quantiles)
    total = counts.sum()
    proportions = (counts / total) if total else counts
    return {
        "quantiles": quantiles.tolist(),
        "proportions": proportions.tolist(),
        "count": len(clean),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def population_stability_index(
    quantiles: np.ndarray, baseline_proportions: np.ndarray, current: np.ndarray
) -> float:
    """PSI = sum((current% - baseline%) * ln(current% / baseline%)).

    Zero counts are floored at a small epsilon; without it a single empty bucket
    makes PSI infinite and the metric useless.
    """
    if quantiles.size < 2 or baseline_proportions.size == 0:
        return 0.0
    counts, _ = np.histogram(current, bins=quantiles)
    total = counts.sum()
    if total == 0:
        return 0.0
    current_proportions = counts / total

    epsilon = 1e-6
    baseline = np.clip(baseline_proportions[: len(current_proportions)], epsilon, None)
    observed = np.clip(current_proportions, epsilon, None)
    return float(np.sum((observed - baseline) * np.log(observed / baseline)))
