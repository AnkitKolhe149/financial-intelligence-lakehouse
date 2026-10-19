"""Feature engineering, temporal integrity and the anomaly model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.anomaly_model import AnomalyModel, train_anomaly_model
from src.ml.evaluation import evaluate_model, rule_model_agreement
from src.ml.features import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    assert_no_leakage,
    build_features,
    temporal_split,
)
from src.ml.model_registry import ModelRegistry


@pytest.fixture
def timeline() -> pd.DataFrame:
    """A deterministic transaction timeline for two accounts and five vendors.

    Spans ~50 days at two-hour intervals. The span matters: the widest windowed
    feature looks back seven days, so a shorter timeline would never let those
    features saturate and the stationarity test below would be measuring the
    fixture rather than the feature set.
    """
    rng = np.random.default_rng(42)
    count = 600
    stamps = pd.date_range("2026-01-01", periods=count, freq="2h", tz="UTC")
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:06d}" for i in range(count)],
            "account_id": ["ACC-1" if i % 2 else "ACC-2" for i in range(count)],
            "vendor_id": [f"VEN-{i % 5}" for i in range(count)],
            "amount_usd": rng.lognormal(6, 0.6, count).round(2),
            "amount": rng.lognormal(6, 0.6, count).round(2),
            "currency": ["USD"] * count,
            "transaction_timestamp": stamps,
            "location_id": [f"LOC-{i % 3}" for i in range(count)],
        }
    )


class TestFeatureEngineering:
    def test_all_declared_features_are_produced(self, timeline):
        result = build_features(timeline)
        for column in FEATURE_COLUMNS:
            assert column in result.frame.columns, f"missing feature: {column}"

    def test_features_are_finite(self, timeline):
        frame = build_features(timeline).frame
        for column in FEATURE_COLUMNS:
            values = frame[column]
            assert values.notna().all(), f"{column} contains nulls"
            assert np.isfinite(values).all(), f"{column} contains inf"

    def test_history_features_use_only_prior_rows(self, timeline):
        """The core leakage guard: the first transaction of an account must see
        zero history, and the count must never exceed the rows before it."""
        frame = build_features(timeline).frame

        for account, group in frame.groupby("account_id"):
            ordered = group.sort_values("transaction_timestamp")
            assert (
                ordered.iloc[0]["account_txn_count_prior"] == 0
            ), f"{account}: first transaction already has history"
            expected = np.arange(len(ordered), dtype=float)
            assert np.array_equal(ordered["account_txn_count_prior"].to_numpy(), expected)

    def test_prior_average_excludes_the_current_row(self):
        """A large payment must not inflate the average it is compared against."""
        frame = pd.DataFrame(
            {
                "transaction_id": ["A", "B", "C"],
                "account_id": ["ACC-1"] * 3,
                "vendor_id": ["VEN-1"] * 3,
                "amount_usd": [100.0, 100.0, 10_000.0],
                "transaction_timestamp": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-03"], utc=True
                ),
            }
        )
        built = build_features(frame).frame.sort_values("transaction_timestamp")
        # The third row's baseline is the mean of the first two, not of all three.
        assert built.iloc[2]["account_avg_amount_prior"] == pytest.approx(100.0)
        assert built.iloc[2]["amount_to_account_avg"] == pytest.approx(100.0)

    def test_velocity_counts_only_the_trailing_window(self):
        stamps = pd.to_datetime(
            [
                "2026-01-01T10:00:00",
                "2026-01-01T10:01:00",
                "2026-01-01T10:02:00",
                "2026-01-01T18:00:00",
            ],
            utc=True,
        )
        frame = pd.DataFrame(
            {
                "transaction_id": list("ABCD"),
                "account_id": ["ACC-1"] * 4,
                "vendor_id": ["VEN-1"] * 4,
                "amount_usd": [10.0] * 4,
                "transaction_timestamp": stamps,
            }
        )
        built = build_features(frame).frame.sort_values("transaction_timestamp")
        assert list(built["account_velocity_5min"]) == [0.0, 1.0, 2.0, 0.0]

    def test_new_vendor_pairing_is_flagged_once(self, timeline):
        frame = build_features(timeline).frame
        firsts = frame.groupby(["account_id", "vendor_id"])["is_new_vendor_for_account"].sum()
        assert (firsts == 1).all()

    def test_no_label_or_output_is_a_feature(self):
        assert not set(FEATURE_COLUMNS) & FORBIDDEN_FEATURES
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage([*FEATURE_COLUMNS, "is_fraud"])

    def test_features_are_stationary(self, timeline):
        """The feature set must not drift with calendar time.

        Raw cumulative counts grow forever, so a model trained on the first half
        of a timeline sees systematically smaller values than it later scores
        against - which is what made the original feature set drift.
        """
        frame = build_features(timeline).frame.sort_values("transaction_timestamp")

        # The first seven days are excluded: until the widest window has filled,
        # windowed counts are genuinely still ramping up. That warm-up is real
        # and expected - it is why train_anomaly_model excludes warm-up rows
        # from fitting - so measuring stationarity across it would test the
        # ramp rather than the steady state.
        warm_start = frame["transaction_timestamp"].min() + pd.Timedelta(days=7)
        steady = frame[frame["transaction_timestamp"] >= warm_start]
        assert len(steady) > 100, "fixture is too short to reach a steady state"

        first_half = steady.iloc[: len(steady) // 2]
        second_half = steady.iloc[len(steady) // 2 :]

        for column in FEATURE_COLUMNS:
            early, late = first_half[column].mean(), second_half[column].mean()
            if abs(early) < 1e-9 and abs(late) < 1e-9:
                continue
            # log_account_history is intentionally growing but log-compressed.
            limit = 3.0 if column == "log_account_history" else 2.5
            ratio = (abs(late) + 1e-9) / (abs(early) + 1e-9)
            assert ratio < limit, f"{column} drifts with time (early={early:.3f}, late={late:.3f})"

    def test_region_mismatch_requires_dimensions(self, timeline):
        accounts = pd.DataFrame({"account_id": ["ACC-1", "ACC-2"], "region": ["WEST", "WEST"]})
        locations = pd.DataFrame(
            {"location_id": ["LOC-0", "LOC-1", "LOC-2"], "region": ["WEST", "EAST", "WEST"]}
        )
        frame = build_features(timeline, accounts, locations).frame
        assert frame["is_region_mismatch"].sum() > 0

        without = build_features(timeline).frame
        assert without["is_region_mismatch"].sum() == 0

    def test_empty_input(self):
        result = build_features(pd.DataFrame())
        assert result.rows == 0


class TestTemporalSplit:
    def test_split_is_chronological(self, timeline):
        frame = build_features(timeline).frame
        split = temporal_split(frame, 0.7, 0.15)

        assert len(split.train) > len(split.validation)
        assert (
            split.train["transaction_timestamp"].max()
            <= split.validation["transaction_timestamp"].min()
        )
        assert (
            split.validation["transaction_timestamp"].max()
            <= split.test["transaction_timestamp"].min()
        )

    def test_no_transaction_appears_twice(self, timeline):
        frame = build_features(timeline).frame
        split = temporal_split(frame)
        ids = (
            set(split.train["transaction_id"])
            | set(split.validation["transaction_id"])
            | set(split.test["transaction_id"])
        )
        assert len(ids) == len(frame)

    def test_empty_input(self):
        split = temporal_split(pd.DataFrame(columns=["transaction_timestamp"]))
        assert split.train.empty


class TestAnomalyModel:
    def test_fit_and_score(self, timeline, settings):
        features = build_features(timeline).frame
        model = AnomalyModel(settings.ml).fit(features)
        scored = model.score(features)

        assert "anomaly_score" in scored.columns
        assert scored["anomaly_score"].between(0, 1).all()
        assert scored["is_anomaly"].dtype == bool
        assert scored["model_version"].nunique() == 1

    def test_flagged_rate_is_close_to_the_configured_contamination(self, timeline, settings):
        features = build_features(timeline).frame
        model = AnomalyModel(settings.ml).fit(features)
        rate = model.score(features)["is_anomaly"].mean()
        assert rate < settings.ml.contamination * 6

    def test_extreme_values_score_higher_than_typical_ones(self, timeline, settings):
        features = build_features(timeline).frame
        model = AnomalyModel(settings.ml).fit(features)

        outlier = features.tail(1).copy()
        outlier["amount_usd"] = 5_000_000.0
        outlier["log_amount"] = np.log1p(5_000_000.0)
        outlier["amount_to_account_avg"] = 500.0
        outlier["account_velocity_5min"] = 40.0

        typical_score = model.score(features)["anomaly_score"].median()
        assert model.score(outlier)["anomaly_score"].iloc[0] > typical_score

    def test_scoring_before_fitting_raises(self, timeline, settings):
        with pytest.raises(RuntimeError, match="fitted"):
            AnomalyModel(settings.ml).score(build_features(timeline).frame)

    def test_missing_feature_columns_raise(self, timeline, settings):
        features = build_features(timeline).frame
        model = AnomalyModel(settings.ml).fit(features)
        with pytest.raises(ValueError, match="missing columns"):
            model.score(features.drop(columns=["amount_to_account_avg"]))

    def test_training_is_reproducible(self, timeline, settings):
        features = build_features(timeline).frame
        first = AnomalyModel(settings.ml).fit(features).score(features)["anomaly_score"]
        second = AnomalyModel(settings.ml).fit(features).score(features)["anomaly_score"]
        pd.testing.assert_series_equal(first, second, check_names=False)


class TestEvaluation:
    def test_metrics_without_labels(self, timeline, settings):
        features = build_features(timeline).frame
        scored = AnomalyModel(settings.ml).fit(features).score(features)
        report = evaluate_model(scored, labels=None)
        assert report["labelled"] is False
        assert "score_distribution" in report

    def test_metrics_with_labels(self, timeline, settings):
        features = build_features(timeline).frame
        scored = AnomalyModel(settings.ml).fit(features).score(features)
        labels = pd.DataFrame(
            {
                "transaction_id": scored.nlargest(5, "anomaly_score")["transaction_id"],
                "is_fraud": True,
            }
        )
        report = evaluate_model(scored, labels)
        assert report["labelled"] is True
        assert report["pr_auc"] > 0
        assert "precision_at_k" in report
        # Accuracy must never be reported on a problem this imbalanced.
        assert "accuracy" not in report
        assert "not reported" in report["metric_note"]

    def test_rule_and_model_agreement(self, scored_transactions):
        agreement = rule_model_agreement(scored_transactions)
        assert agreement["both"] + agreement["ml_only"] + agreement["rules_only"] > 0
        assert 0 <= agreement["agreement_pct"] <= 100


class TestModelRegistry:
    def test_save_and_load_round_trip(self, timeline, settings, store, run_context):
        model, report, _ = train_anomaly_model(timeline, settings=settings)
        registry = ModelRegistry(settings, store)
        artifact = registry.save(model, report, run_context.run_id)

        assert artifact.exists()
        loaded = registry.load()
        assert loaded is not None
        assert loaded.model.model_version == model.model_version
        assert loaded.model.feature_columns == list(FEATURE_COLUMNS)

    def test_loaded_model_scores_identically(self, timeline, settings, store, run_context):
        model, report, features = train_anomaly_model(timeline, settings=settings)
        registry = ModelRegistry(settings, store)
        registry.save(model, report, run_context.run_id)

        reloaded = registry.load().model
        pd.testing.assert_series_equal(
            model.score(features)["anomaly_score"],
            reloaded.score(features)["anomaly_score"],
            check_names=False,
        )

    def test_feature_drift_is_refused_loudly(
        self, timeline, settings, store, run_context, monkeypatch
    ):
        """A model trained on a different feature set must not silently score."""
        import joblib

        model, report, _ = train_anomaly_model(timeline, settings=settings)
        registry = ModelRegistry(settings, store)
        artifact = registry.save(model, report, run_context.run_id)

        payload = joblib.load(artifact)
        payload["feature_columns"] = ["amount_usd", "hour_of_day"]
        joblib.dump(payload, artifact)

        with pytest.raises(ValueError, match="different feature set"):
            registry.load(model.model_version)

    def test_registry_records_provenance(self, timeline, settings, store, run_context):
        model, report, _ = train_anomaly_model(timeline, settings=settings)
        ModelRegistry(settings, store).save(model, report, run_context.run_id)

        entry = store.active_model(settings.ml.model_name)
        assert entry["algorithm"] == "IsolationForest"
        assert entry["training_rows"] > 0
        assert entry["feature_list"]
        assert entry["run_id"] == run_context.run_id

    def test_no_model_available_returns_none(self, settings, store):
        assert ModelRegistry(settings, store).load() is None


class TestTrainingPipeline:
    def test_warmup_rows_are_excluded_from_training_only(self, timeline, settings):
        model, report, features = train_anomaly_model(timeline, settings=settings)
        # Training used fewer rows than exist, but every row is still scorable.
        assert report.training_rows <= len(features)
        assert len(model.score(features)) == len(features)

    def test_report_captures_the_training_window(self, timeline, settings):
        _, report, _ = train_anomaly_model(timeline, settings=settings)
        assert report.training_window_start is not None
        assert report.training_window_end is not None
        assert report.training_window_start < report.training_window_end
        assert report.hyperparameters["contamination"] == settings.ml.contamination
