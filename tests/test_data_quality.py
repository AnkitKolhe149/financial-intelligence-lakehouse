"""Data quality framework: checks, scoring and the validator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_quality.base_check import CheckContext, CheckSeverity, Dimension
from src.data_quality.checks import (
    AllowedValuesCheck,
    CompletenessCheck,
    DistributionCheck,
    FreshnessCheck,
    RangeCheck,
    ReferentialIntegrityCheck,
    SchemaCheck,
    TimestampSanityCheck,
    UniquenessCheck,
    VolumeCheck,
    population_stability_index,
    profile_numeric,
)
from src.data_quality.scoring import aggregate_platform_score, score_dataset
from src.data_quality.suites import build_suite
from src.data_quality.validator import DataQualityValidator


@pytest.fixture
def context() -> CheckContext:
    return CheckContext(dataset="transaction", layer="silver")


class TestCompleteness:
    def test_full_population_scores_one(self, clean_transactions, context):
        check = CompletenessCheck(columns=["transaction_id", "account_id"])
        result = check.evaluate(clean_transactions, context)
        assert result.passed
        assert result.score == 1.0

    def test_score_is_graded_not_binary(self, clean_transactions, context):
        frame = clean_transactions.copy()
        frame.loc[0:1, "account_id"] = None
        result = CompletenessCheck(columns=["account_id"]).evaluate(frame, context)
        assert result.score == pytest.approx(0.8)
        assert result.records_failed == 2
        assert not result.passed

    def test_absent_column_skips_rather_than_failing(self, clean_transactions, context):
        result = CompletenessCheck(columns=["not_a_column"]).evaluate(clean_transactions, context)
        assert result.details.get("skipped") is True
        assert result.score == 1.0

    def test_weakest_column_is_identified(self, clean_transactions, context):
        frame = clean_transactions.copy()
        frame.loc[0:4, "contract_id"] = None
        frame.loc[0, "agency_id"] = None
        result = CompletenessCheck(columns=["contract_id", "agency_id"]).evaluate(frame, context)
        assert result.column_name == "contract_id"


class TestUniqueness:
    def test_unique_keys_pass(self, clean_transactions, context):
        assert UniquenessCheck(column="transaction_id").evaluate(clean_transactions, context).passed

    def test_duplicates_are_counted_and_reported(self, clean_transactions, context):
        frame = pd.concat([clean_transactions, clean_transactions.head(2)], ignore_index=True)
        result = UniquenessCheck(column="transaction_id").evaluate(frame, context)
        assert not result.passed
        assert result.records_failed == 2
        assert result.details["top_offenders"]


class TestValidity:
    def test_range_check_flags_negative_amounts(self, clean_transactions, context):
        frame = clean_transactions.copy()
        frame.loc[0, "amount"] = -50
        result = RangeCheck(column="amount", minimum=0, exclusive_min=True).evaluate(frame, context)
        assert not result.passed
        assert result.records_failed == 1

    def test_allowed_values(self, clean_transactions, context):
        frame = clean_transactions.copy()
        frame.loc[0, "currency"] = "XXX"
        result = AllowedValuesCheck(
            column="currency", allowed=["USD", "EUR", "GBP"], threshold=1.0
        ).evaluate(frame, context)
        assert not result.passed
        assert "XXX" in result.details["unexpected_values"]

    def test_timestamp_sanity_detects_the_future(self, clean_transactions, context):
        frame = clean_transactions.copy()
        frame["transaction_timestamp"] = pd.to_datetime(frame["transaction_timestamp"], utc=True)
        frame.loc[0, "transaction_timestamp"] = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)
        result = TimestampSanityCheck(column="transaction_timestamp").evaluate(frame, context)
        assert not result.passed
        assert result.observed_value == 1.0


class TestIntegrity:
    def test_orphans_are_detected(self, clean_transactions):
        context = CheckContext(
            dataset="transaction",
            references={"silver_accounts": pd.Series(["ACC-0000001"])},
        )
        result = ReferentialIntegrityCheck(
            column="account_id", reference_table="silver_accounts"
        ).evaluate(clean_transactions, context)
        assert not result.passed
        assert result.records_failed == 5  # the ACC-0000002 rows
        assert "ACC-0000002" in result.details["examples"]

    def test_missing_reference_skips(self, clean_transactions, context):
        result = ReferentialIntegrityCheck(
            column="account_id", reference_table="silver_accounts"
        ).evaluate(clean_transactions, context)
        assert result.details.get("skipped") is True


class TestFreshness:
    def test_recent_data_is_fresh(self, context):
        frame = pd.DataFrame(
            {"transaction_timestamp": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)]}
        )
        result = FreshnessCheck(column="transaction_timestamp", max_age_hours=48).evaluate(
            frame, context
        )
        assert result.passed
        assert result.score == 1.0

    def test_staleness_decays_rather_than_cliff_edging(self, context):
        """Data one hour late is not as broken as data a week late."""
        slightly_stale = pd.DataFrame(
            {"transaction_timestamp": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=60)]}
        )
        very_stale = pd.DataFrame(
            {"transaction_timestamp": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)]}
        )
        check = FreshnessCheck(column="transaction_timestamp", max_age_hours=48)
        mild = check.evaluate(slightly_stale, context)
        severe = check.evaluate(very_stale, context)

        assert not mild.passed and not severe.passed
        assert mild.score > severe.score
        assert severe.score == 0.0


class TestVolume:
    def test_low_volume_uses_a_poisson_band(self, context):
        """At ~10 records a day, ±40% is ordinary noise, not an anomaly.

        This is the calibration bug the framework had originally: a flat
        percentage threshold fired on a third of all days purely because the
        counts were small.
        """
        rng = np.random.default_rng(42)
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        counts = rng.poisson(10, size=60)
        frame = pd.DataFrame({"transaction_date": np.repeat(dates.astype(str), counts)})
        result = VolumeCheck(date_column="transaction_date", max_deviation_pct=40).evaluate(
            frame, context
        )
        assert "Poisson" in result.details["regime"]
        assert result.passed, "ordinary Poisson noise must not be reported as a volume anomaly"

    def test_high_volume_uses_the_percentage_band(self, context):
        rng = np.random.default_rng(7)
        dates = pd.date_range("2026-01-01", periods=40, freq="D")
        counts = rng.poisson(500, size=40)
        frame = pd.DataFrame({"transaction_date": np.repeat(dates.astype(str), counts)})
        result = VolumeCheck(date_column="transaction_date").evaluate(frame, context)
        assert "%" in result.details["regime"]
        assert result.passed

    def test_a_genuine_collapse_is_caught(self, context):
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        counts = [500] * 29 + [5]  # the load broke on the last day
        frame = pd.DataFrame({"transaction_date": np.repeat(dates.astype(str), counts)})
        result = VolumeCheck(date_column="transaction_date").evaluate(frame, context)
        assert result.details["anomalous_days"]

    def test_too_few_days_skips(self, context):
        frame = pd.DataFrame({"transaction_date": ["2026-01-01", "2026-01-02"]})
        result = VolumeCheck(date_column="transaction_date").evaluate(frame, context)
        assert result.details.get("skipped") is True


class TestSchemaCheck:
    def test_missing_columns_fail_but_extra_columns_do_not(self, clean_transactions, context):
        result = SchemaCheck(
            expected_columns=[*clean_transactions.columns, "absent_column"]
        ).evaluate(clean_transactions, context)
        assert not result.passed
        assert "absent_column" in result.details["missing_columns"]

        frame = clean_transactions.copy()
        frame["surprise_from_upstream"] = 1
        tolerant = SchemaCheck(expected_columns=list(clean_transactions.columns)).evaluate(
            frame, context
        )
        assert tolerant.passed
        assert "surprise_from_upstream" in tolerant.details["unexpected_columns"]


class TestDistributionDrift:
    def test_psi_is_zero_for_an_identical_distribution(self):
        rng = np.random.default_rng(1)
        values = pd.Series(rng.lognormal(6, 1, 5000))
        profile = profile_numeric(values)
        psi = population_stability_index(
            np.asarray(profile["quantiles"]), np.asarray(profile["proportions"]), values.to_numpy()
        )
        assert psi < 0.01

    def test_psi_detects_a_shifted_distribution(self):
        rng = np.random.default_rng(1)
        baseline = pd.Series(rng.lognormal(6, 1, 5000))
        shifted = pd.Series(rng.lognormal(8, 1, 5000))
        profile = profile_numeric(baseline)
        psi = population_stability_index(
            np.asarray(profile["quantiles"]), np.asarray(profile["proportions"]), shifted.to_numpy()
        )
        assert psi > 0.25

    def test_first_run_captures_a_baseline_instead_of_failing(self, context):
        frame = pd.DataFrame({"amount_usd": np.random.default_rng(3).lognormal(6, 1, 500)})
        result = DistributionCheck(column="amount_usd").evaluate(frame, context)
        assert result.passed
        assert result.details.get("baseline_captured") is True


class TestScoring:
    def _result(self, dimension, score, records=100, passed=True, severity=CheckSeverity.WARNING):
        from src.data_quality.base_check import CheckResult

        return CheckResult(
            check_name=f"{dimension}_check",
            dimension=dimension,
            severity=severity,
            passed=passed,
            score=score,
            records_evaluated=records,
            records_failed=0,
            message="",
        )

    def test_dimensions_are_weighted_by_records(self, settings):
        checks = [
            self._result(Dimension.COMPLETENESS, 1.0, records=10_000),
            self._result(Dimension.COMPLETENESS, 0.0, records=10),
        ]
        card = score_dataset("transaction", checks, "run-1", settings=settings)
        # The big check dominates - a 10-row perfect score must not cancel a
        # 10,000-row failure.
        assert card.dimension_scores["completeness"] > 99

    def test_unmeasured_dimensions_are_none_not_perfect(self, settings):
        checks = [self._result(Dimension.COMPLETENESS, 1.0)]
        card = score_dataset("transaction", checks, "run-1", settings=settings)
        assert card.dimension_scores["freshness"] is None
        assert card.overall_score == 100.0

    def test_blocking_failure_forces_fail_status(self, settings):
        checks = [
            self._result(Dimension.COMPLETENESS, 0.999, records=10_000),
            self._result(Dimension.UNIQUENESS, 0.99, passed=False, severity=CheckSeverity.BLOCKING),
        ]
        card = score_dataset("transaction", checks, "run-1", settings=settings)
        assert card.status == "FAIL"
        assert card.blocking_failures

    def test_platform_score_is_record_weighted(self, settings):
        from src.data_quality.scoring import QualityScorecard

        cards = [
            QualityScorecard("transaction", "silver", "r", 90.0, {}, records_evaluated=100_000),
            QualityScorecard("agency", "silver", "r", 100.0, {}, records_evaluated=20),
        ]
        assert aggregate_platform_score(cards) == pytest.approx(90.0, abs=0.05)


class TestSuites:
    def test_suite_is_derived_from_the_schema(self, settings):
        checks = build_suite("transaction", settings)
        names = {c.name for c in checks}
        assert any("completeness" in n for n in names)
        assert "uniqueness_transaction_id" in names
        assert any("referential" in n for n in names)
        assert any("freshness" in n for n in names)
        assert any("volume" in n for n in names)

    def test_evolved_columns_are_excluded_from_optional_completeness(self, settings):
        """merchant_category is null for every v1.0 record by design; including
        it would guarantee a permanent, meaningless warning."""
        checks = build_suite("transaction", settings)
        optional = next((c for c in checks if c.name == "completeness_optional_transaction"), None)
        assert optional is not None
        assert "merchant_category" not in optional.columns


class TestValidatorIntegration:
    def test_scores_a_dataset_and_persists(self, settings, store, clean_transactions):
        validator = DataQualityValidator("test-run", settings, store)
        frame = clean_transactions.copy()
        frame["transaction_timestamp"] = pd.to_datetime(frame["transaction_timestamp"], utc=True)
        frame["amount_usd"] = frame["amount"]
        frame["transaction_date"] = frame["transaction_timestamp"].dt.date.astype("string")

        card = validator.evaluate_dataset("transaction", frame)
        assert card.overall_score > 0
        assert card.checks

        history = store.quality_history()
        assert not history.empty
        assert history.iloc[0]["dataset"] == "transaction"

    def test_a_raising_check_is_recorded_not_propagated(self, settings, store, clean_transactions):
        """One broken check must not lose the whole scorecard."""

        class ExplodingCheck(CompletenessCheck):
            def evaluate(self, frame, context):
                raise RuntimeError("boom")

        validator = DataQualityValidator("test-run", settings, store)
        result = validator._run_check(
            ExplodingCheck(columns=["account_id"]),
            clean_transactions,
            CheckContext(dataset="transaction"),
        )
        assert not result.passed
        assert result.details.get("errored") is True
        assert "RuntimeError" in result.message
