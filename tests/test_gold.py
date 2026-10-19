"""Gold layer: aggregates, risk profiles, alerts and the star schema."""

from __future__ import annotations

import json

import pandas as pd

from src.core.io import read_parquet
from src.gold.account_risk import build_account_risk
from src.gold.anomaly_summary import build_anomaly_transactions, build_risk_alerts
from src.gold.base import percentage, safe_divide
from src.gold.financial_summary import build_daily_financial_summary
from src.gold.spending import (
    build_agency_spending,
    build_department_spending,
    build_transaction_velocity,
)
from src.gold.star_schema import build_dim_vendor, build_fact_transactions, surrogate_key
from src.gold.vendor_risk import build_vendor_risk


class TestHelpers:
    def test_safe_divide_never_yields_infinity(self):
        result = safe_divide(pd.Series([10.0, 5.0]), pd.Series([0.0, 5.0]))
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 1.0

    def test_percentage(self):
        assert percentage(pd.Series([25.0]), pd.Series([100.0])).iloc[0] == 25.0


class TestDailySummary:
    def test_aggregates_per_day(self, scored_transactions, settings, store, run_context):
        result = build_daily_financial_summary(
            scored_transactions, settings, store=store, run_id=run_context.run_id
        )
        assert result.records > 0

        frame = read_parquet(settings.dataset_path("gold", "gold_daily_financial_summary"))
        assert frame["transaction_count"].sum() == len(scored_transactions)
        assert "median_transaction_usd" in frame.columns
        assert "rolling_7d_avg_volume" in frame.columns

    def test_median_and_mean_are_both_reported(
        self, scored_transactions, settings, store, run_context
    ):
        """Heavy-tailed amounts make the two diverge; showing only one hides it."""
        build_daily_financial_summary(
            scored_transactions, settings, store=store, run_id=run_context.run_id
        )
        frame = read_parquet(settings.dataset_path("gold", "gold_daily_financial_summary"))
        assert "average_transaction_usd" in frame.columns
        assert "median_transaction_usd" in frame.columns

    def test_empty_input_still_produces_a_readable_dataset(self, settings, store, run_context):
        result = build_daily_financial_summary(
            pd.DataFrame(), settings, store=store, run_id=run_context.run_id
        )
        assert result.records == 0
        # A zero-row day must be queryable, not "table not found".
        from src.core.io import dataset_exists

        assert dataset_exists(settings.dataset_path("gold", "gold_daily_financial_summary"))


class TestAccountRisk:
    def test_profiles_every_account(
        self, scored_transactions, reference_data, settings, store, run_context
    ):
        build_account_risk(
            scored_transactions,
            reference_data.accounts,
            settings,
            store=store,
            run_id=run_context.run_id,
        )
        frame = read_parquet(settings.dataset_path("gold", "gold_account_risk"))
        assert set(frame["account_id"]) == set(scored_transactions["account_id"])
        assert frame["risk_score"].between(0, 100).all()
        assert set(frame["risk_level"]) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_a_rare_critical_is_not_averaged_away(self, settings, store, run_context):
        """One critical transaction among many routine ones must still register.

        Averaging transaction scores would bury exactly the event the platform
        exists to surface.
        """
        rows = 200
        frame = pd.DataFrame(
            {
                "transaction_id": [f"TXN-{i}" for i in range(rows)],
                "account_id": ["ACC-1"] * rows,
                "vendor_id": ["VEN-1"] * rows,
                "amount_usd": [100.0] * rows,
                "transaction_timestamp": pd.date_range(
                    "2026-01-01", periods=rows, freq="h", tz="UTC"
                ),
                "transaction_date": "2026-01-01",
                "location_id": ["LOC-1"] * rows,
                "risk_score": [0] * (rows - 1) + [100],
                "risk_level": ["LOW"] * (rows - 1) + ["CRITICAL"],
                "is_anomaly": [False] * (rows - 1) + [True],
            }
        )
        build_account_risk(frame, pd.DataFrame(), settings, store=store, run_id=run_context.run_id)
        profile = read_parquet(settings.dataset_path("gold", "gold_account_risk")).iloc[0]
        assert profile["risk_score"] > 0
        assert profile["max_transaction_risk_score"] == 100


class TestVendorRisk:
    def test_profile_includes_contract_coverage(
        self, scored_transactions, reference_data, settings, store, run_context
    ):
        build_vendor_risk(
            scored_transactions,
            reference_data.vendors,
            reference_data.contracts,
            settings,
            store=store,
            run_id=run_context.run_id,
        )
        frame = read_parquet(settings.dataset_path("gold", "gold_vendor_risk"))
        assert "uncontracted_payment_pct" in frame.columns
        assert "growth_ratio" in frame.columns
        assert "contract_count" in frame.columns
        assert frame["risk_score"].between(0, 100).all()

    def test_duplicate_registrations_are_counted(
        self, scored_transactions, reference_data, settings, store, run_context
    ):
        vendors = reference_data.vendors.copy()
        vendors.loc[vendors.index[:2], "tax_id"] = "TX-SHARED"
        build_vendor_risk(
            scored_transactions,
            vendors,
            reference_data.contracts,
            settings,
            store=store,
            run_id=run_context.run_id,
        )
        frame = read_parquet(settings.dataset_path("gold", "gold_vendor_risk"))
        assert "shares_tax_id_with" in frame.columns


class TestAnomaliesAndAlerts:
    def test_flagged_transactions_are_published_with_explanations(
        self, scored_transactions, settings, store, run_context
    ):
        build_anomaly_transactions(
            scored_transactions, settings, store=store, run_id=run_context.run_id
        )
        frame = read_parquet(settings.dataset_path("gold", "gold_anomaly_transactions"))
        assert len(frame) > 0
        # Reasons are JSON strings in Gold so every consumer reads them the same way.
        assert isinstance(json.loads(frame.iloc[0]["risk_reasons"]), list)

    def test_selection_is_a_union_of_model_and_rules(self, settings, store, run_context):
        """Requiring both would discard the novel patterns the model exists for."""
        frame = pd.DataFrame(
            {
                "transaction_id": ["A", "B", "C"],
                "account_id": ["ACC-1"] * 3,
                "vendor_id": ["VEN-1"] * 3,
                "amount_usd": [10.0, 20.0, 30.0],
                "transaction_timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
                "transaction_date": ["2026-01-01"] * 3,
                "is_anomaly": [True, False, False],  # model only
                "risk_level": ["LOW", "HIGH", "LOW"],  # rules only
                "risk_score": [0, 60, 0],
                "risk_reasons": [[], ["rule"], []],
                "risk_rule_hits": [[], ["new_vendor"], []],
            }
        )
        build_anomaly_transactions(frame, settings, store=store, run_id=run_context.run_id)
        published = read_parquet(settings.dataset_path("gold", "gold_anomaly_transactions"))
        assert set(published["transaction_id"]) == {"A", "B"}

    def test_alert_ids_are_deterministic_across_runs(
        self, scored_transactions, settings, store, run_context
    ):
        """A re-run must update the case, not create a duplicate one."""
        build_risk_alerts(
            scored_transactions, settings, store=store, run_id=run_context.run_id, minimum_score=50
        )
        first = read_parquet(settings.dataset_path("gold", "gold_risk_alerts"))["alert_id"].tolist()

        build_risk_alerts(
            scored_transactions, settings, store=store, run_id="a-different-run", minimum_score=50
        )
        second = read_parquet(settings.dataset_path("gold", "gold_risk_alerts"))[
            "alert_id"
        ].tolist()

        assert sorted(first) == sorted(second)

    def test_alerts_start_open(self, scored_transactions, settings, store, run_context):
        build_risk_alerts(
            scored_transactions, settings, store=store, run_id=run_context.run_id, minimum_score=50
        )
        frame = read_parquet(settings.dataset_path("gold", "gold_risk_alerts"))
        assert (frame["status"] == "OPEN").all()


class TestSpending:
    def test_agency_spending_joins_budgets(
        self, scored_transactions, reference_data, settings, store, run_context
    ):
        build_agency_spending(
            scored_transactions,
            reference_data.agencies,
            reference_data.budgets,
            settings,
            store=store,
            run_id=run_context.run_id,
        )
        frame = read_parquet(settings.dataset_path("gold", "gold_agency_spending"))
        assert "budget_utilisation_pct" in frame.columns
        assert "top_vendor_share_pct" in frame.columns

    def test_department_overruns_are_flagged(
        self, scored_transactions, reference_data, settings, store, run_context
    ):
        build_department_spending(
            scored_transactions,
            reference_data.departments,
            reference_data.budgets,
            settings,
            store=store,
            run_id=run_context.run_id,
        )
        frame = read_parquet(settings.dataset_path("gold", "gold_department_spending"))
        assert "is_over_budget" in frame.columns

    def test_velocity_flags_bursts(self, scored_transactions, settings, store, run_context):
        frame = scored_transactions.copy()
        frame["account_velocity_5min"] = [1.0] * 9 + [25.0]
        build_transaction_velocity(frame, settings, store=store, run_id=run_context.run_id)

        velocity = read_parquet(settings.dataset_path("gold", "gold_transaction_velocity"))
        assert velocity["is_burst"].any()


class TestStarSchema:
    def test_surrogate_keys_are_deterministic(self):
        assert surrogate_key("VEN", "VEN-1") == surrogate_key("VEN", "VEN-1")
        assert surrogate_key("VEN", "VEN-1") != surrogate_key("VEN", "VEN-2")

    def test_fact_table_is_narrow(self, scored_transactions, settings, store, run_context):
        build_fact_transactions(
            scored_transactions, settings, store=store, run_id=run_context.run_id
        )
        fact = read_parquet(settings.dataset_path("gold", "fact_transactions"))
        assert "transaction_key" in fact.columns
        assert "vendor_key" in fact.columns
        # Descriptive attributes belong in dimensions, not the fact.
        assert "vendor_name" not in fact.columns
        assert "description" not in fact.columns

    def test_scd2_opens_one_version_per_vendor_initially(
        self, reference_data, settings, store, run_context
    ):
        build_dim_vendor(reference_data.vendors, settings, store=store, run_id=run_context.run_id)
        dimension = read_parquet(settings.dataset_path("gold", "dim_vendor"))
        assert len(dimension) == len(reference_data.vendors)
        assert dimension["is_current"].all()
        assert (dimension["version"] == 1).all()

    def test_scd2_versions_a_changed_attribute(self, reference_data, settings, store, run_context):
        """History must be preserved, not overwritten."""
        build_dim_vendor(reference_data.vendors, settings, store=store, run_id=run_context.run_id)
        existing = read_parquet(settings.dataset_path("gold", "dim_vendor"))

        changed = reference_data.vendors.copy()
        changed.loc[changed.index[0], "risk_flag"] = "SANCTIONED"

        build_dim_vendor(
            changed, settings, existing=existing, store=store, run_id=run_context.run_id
        )
        dimension = read_parquet(settings.dataset_path("gold", "dim_vendor"))

        vendor_id = changed.iloc[0]["vendor_id"]
        versions = dimension[dimension["vendor_id"] == vendor_id]
        assert len(versions) == 2, "the previous version must be retained"
        assert versions["is_current"].sum() == 1
        current = versions[versions["is_current"]].iloc[0]
        assert current["risk_flag"] == "SANCTIONED"
        assert current["version"] == 2

    def test_scd2_leaves_unchanged_vendors_alone(
        self, reference_data, settings, store, run_context
    ):
        build_dim_vendor(reference_data.vendors, settings, store=store, run_id=run_context.run_id)
        existing = read_parquet(settings.dataset_path("gold", "dim_vendor"))

        build_dim_vendor(
            reference_data.vendors,
            settings,
            existing=existing,
            store=store,
            run_id=run_context.run_id,
        )
        dimension = read_parquet(settings.dataset_path("gold", "dim_vendor"))
        assert len(dimension) == len(reference_data.vendors)
