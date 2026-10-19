"""Silver transformations: cleaning, casting, validation and quarantine."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.config import Settings
from src.core.contracts import ErrorCode
from src.core.io import read_parquet
from src.silver.spec import get_silver_spec
from src.silver.transformations import build_silver
from src.silver.validation import (
    add_transaction_derivations,
    cast_types,
    normalise_strings,
    validate_frame,
)


@pytest.fixture
def spec():
    return get_silver_spec("transaction")


class TestNormalisation:
    def test_whitespace_and_case_are_normalised(self, spec):
        frame = pd.DataFrame(
            {
                "transaction_id": ["  TXN-1  "],
                "currency": [" usd "],
                "description": ["Monthly   services    invoice"],
            }
        )
        cleaned = normalise_strings(frame, spec.normalisation)
        assert cleaned.loc[0, "transaction_id"] == "TXN-1"
        assert cleaned.loc[0, "currency"] == "USD"
        assert cleaned.loc[0, "description"] == "Monthly services invoice"

    @pytest.mark.parametrize("token", ["", "  ", "NULL", "none", "N/A", "nan", "-", "unknown"])
    def test_nullish_tokens_become_real_nulls(self, spec, token):
        frame = pd.DataFrame({"transaction_id": ["TXN-1"], "contract_id": [token]})
        cleaned = normalise_strings(frame, spec.normalisation)
        assert pd.isna(cleaned.loc[0, "contract_id"])

    def test_audit_columns_are_left_alone(self, spec):
        frame = pd.DataFrame({"transaction_id": ["TXN-1"], "_record_hash": ["  AbCdEf  "]})
        cleaned = normalise_strings(frame, spec.normalisation)
        assert cleaned.loc[0, "_record_hash"] == "  AbCdEf  "


class TestCasting:
    def test_types_follow_the_canonical_schema(self, spec, clean_transactions):
        typed = cast_types(clean_transactions, spec)
        assert pd.api.types.is_numeric_dtype(typed["amount"])
        assert pd.api.types.is_datetime64_any_dtype(typed["transaction_timestamp"])
        assert str(typed["transaction_id"].dtype) in {"string", "str"}

    def test_unparseable_values_become_null_rather_than_raising(self, spec):
        frame = pd.DataFrame(
            {
                "transaction_id": ["TXN-1", "TXN-2"],
                "amount": ["100.50", "not-a-number"],
                "transaction_timestamp": ["2026-01-01T10:00:00", "garbage"],
            }
        )
        typed = cast_types(frame, spec)
        assert typed.loc[0, "amount"] == 100.50
        assert pd.isna(typed.loc[1, "amount"])
        assert pd.isna(typed.loc[1, "transaction_timestamp"])

    def test_mixed_timestamp_formats_all_parse(self, spec):
        frame = pd.DataFrame(
            {
                "transaction_id": ["A", "B", "C"],
                "transaction_timestamp": [
                    "2026-01-01T10:00:00",
                    "2026-01-02 11:30:00",
                    "2026-01-03",
                ],
            }
        )
        typed = cast_types(frame, spec)
        assert typed["transaction_timestamp"].notna().all()


class TestDerivations:
    def test_usd_conversion_and_partition_columns(self):
        frame = pd.DataFrame(
            {
                "transaction_id": ["A", "B"],
                "amount": [100.0, 100.0],
                "currency": ["USD", "EUR"],
                "transaction_timestamp": pd.to_datetime(
                    ["2026-01-15T10:00:00", "2025-10-05T10:00:00"], utc=True
                ),
                "posted_timestamp": pd.to_datetime(
                    ["2026-01-15T12:00:00", "2025-10-09T10:00:00"], utc=True
                ),
            }
        )
        enriched = add_transaction_derivations(frame)
        assert enriched.loc[0, "amount_usd"] == 100.0
        assert enriched.loc[1, "amount_usd"] == 109.0
        assert enriched.loc[0, "transaction_date"] == "2026-01-15"
        # October starts the next US federal fiscal year.
        assert enriched.loc[1, "fiscal_year"] == 2026
        assert enriched.loc[0, "fiscal_year"] == 2026
        # Four days between occurrence and posting is late-arriving.
        assert bool(enriched.loc[1, "_is_late_arriving"]) is True
        assert bool(enriched.loc[0, "_is_late_arriving"]) is False


class TestValidation:
    def _validate(self, frame, spec, settings, references=None):
        normalised = normalise_strings(frame, spec.normalisation)
        typed = add_transaction_derivations(cast_types(normalised, spec))
        return validate_frame(typed, spec, settings, references, raw_frame=normalised)

    def test_clean_data_survives_intact(self, clean_transactions, spec, settings: Settings):
        outcome = self._validate(clean_transactions, spec, settings)
        assert outcome.valid_count == 10
        assert outcome.rejected_count == 0

    @pytest.mark.parametrize(
        ("column", "value", "expected"),
        [
            ("account_id", None, ErrorCode.NULL_REQUIRED_FIELD),
            ("vendor_id", None, ErrorCode.NULL_REQUIRED_FIELD),
            ("amount", -500.0, ErrorCode.INVALID_AMOUNT),
            ("amount", 0.0, ErrorCode.INVALID_AMOUNT),
            ("currency", "XXX", ErrorCode.INVALID_CURRENCY),
            ("transaction_timestamp", "not-a-date", ErrorCode.INVALID_TIMESTAMP),
            ("transaction_timestamp", "2099-01-01T00:00:00", ErrorCode.FUTURE_TIMESTAMP),
        ],
    )
    def test_each_defect_produces_its_error_code(
        self, clean_transactions, spec, settings, column, value, expected
    ):
        frame = clean_transactions.copy()
        frame.loc[0, column] = value
        outcome = self._validate(frame, spec, settings)
        assert outcome.rejected_count == 1
        assert outcome.rejected.iloc[0]["error_code"] == expected
        assert outcome.valid_count == 9

    def test_amount_above_the_platform_limit_is_rejected(self, clean_transactions, spec, settings):
        frame = clean_transactions.copy()
        frame.loc[0, "amount"] = settings.quality.max_transaction_amount * 2
        outcome = self._validate(frame, spec, settings)
        assert outcome.rejected.iloc[0]["error_code"] == ErrorCode.AMOUNT_EXCEEDS_LIMIT

    def test_first_failure_wins_so_counts_do_not_double(self, clean_transactions, spec, settings):
        """A record wrong in three ways is one rejection, not three."""
        frame = clean_transactions.copy()
        frame.loc[0, "account_id"] = None
        frame.loc[0, "amount"] = -1
        frame.loc[0, "currency"] = "XXX"
        outcome = self._validate(frame, spec, settings)
        assert outcome.rejected_count == 1
        assert sum(outcome.counts_by_code.values()) == 1

    def test_blocking_foreign_key_quarantines(self, clean_transactions, spec, settings):
        frame = clean_transactions.copy()
        frame.loc[0, "account_id"] = "ACC-9999999"
        references = {"silver_accounts": pd.Series(["ACC-0000001", "ACC-0000002"])}
        outcome = self._validate(frame, spec, settings, references)
        assert outcome.rejected.iloc[0]["error_code"] == ErrorCode.UNKNOWN_ACCOUNT

    def test_non_blocking_foreign_key_nulls_the_reference(self, clean_transactions, spec, settings):
        """An unknown contract is not a reason to discard a real payment."""
        frame = clean_transactions.copy()
        frame["contract_id"] = "CON-0000000"
        references = {"silver_contracts": pd.Series(["CON-0000001"])}
        outcome = self._validate(frame, spec, settings, references)
        assert outcome.rejected_count == 0
        assert outcome.valid["contract_id"].isna().all()

    def test_missing_reference_dataset_skips_the_check(self, clean_transactions, spec, settings):
        frame = clean_transactions.copy()
        frame.loc[0, "account_id"] = "ACC-UNKNOWN"
        outcome = self._validate(frame, spec, settings, references={})
        assert outcome.rejected_count == 0

    def test_empty_frame_is_handled(self, clean_transactions, spec, settings):
        outcome = self._validate(clean_transactions.iloc[0:0], spec, settings)
        assert outcome.valid_count == 0
        assert outcome.rejected_count == 0


class TestSilverBuild:
    @pytest.fixture
    def bronze_loaded(self, settings, store, run_context, transactions, reference_data):
        """Load generated (defect-carrying) data through Bronze."""
        from src.bronze.loader import BronzeLoader
        from src.ingestion.base import IngestionResult, stamp_audit_columns

        loader = BronzeLoader(run_context.run_id, settings, store)
        datasets = {
            "transaction": transactions,
            "account": reference_data.accounts,
            "vendor": reference_data.vendors,
            "agency": reference_data.agencies,
            "department": reference_data.departments,
            "contract": reference_data.contracts,
            "location": reference_data.locations,
        }
        for dataset, frame in datasets.items():
            stamped = stamp_audit_columns(
                frame,
                dataset=dataset,
                source_system="TEST",
                source_format="csv",
                source_file=f"{dataset}.csv",
                run_id=run_context.run_id,
            )
            loader.load(
                dataset,
                [
                    IngestionResult(
                        dataset=dataset,
                        source_system="TEST",
                        source_format="csv",
                        source_file=f"{dataset}.csv",
                        frame=stamped,
                    )
                ],
            )
        return settings

    def test_end_to_end_silver_build(self, bronze_loaded, run_context, store):
        settings = bronze_loaded
        for dataset in ("agency", "department", "location", "vendor", "contract", "account"):
            build_silver(dataset, run_context, settings, store=store)

        result = build_silver("transaction", run_context, settings, store=store)

        assert result.records_in > 0
        assert result.records_out > 0
        assert result.records_quarantined > 0, "generated data must contain defects"
        assert result.records_out < result.records_in

        silver = read_parquet(settings.dataset_path("silver", "silver_transactions"))
        assert silver["transaction_id"].duplicated().sum() == 0
        assert silver["account_id"].isna().sum() == 0
        assert (pd.to_numeric(silver["amount"]) > 0).all()
        assert set(silver["currency"].dropna()) <= set(settings.quality.allowed_currencies)

    def test_quarantine_preserves_the_raw_record(self, bronze_loaded, run_context, store):
        from src.silver.quarantine import read_quarantine

        settings = bronze_loaded
        build_silver("transaction", run_context, settings, store=store)

        quarantined = read_quarantine("transaction", settings)
        assert not quarantined.empty
        assert quarantined["raw_record"].str.len().gt(10).all()
        assert quarantined["error_code"].notna().all()
        assert quarantined["pipeline_run_id"].eq(run_context.run_id).all()

    def test_rebuild_is_deterministic(self, bronze_loaded, run_context, store):
        """Same input, same output - the basis of reproducibility."""
        settings = bronze_loaded
        first = build_silver("transaction", run_context, settings, store=store)
        second = build_silver("transaction", run_context, settings, store=store)
        assert first.records_out == second.records_out
        assert first.records_in == second.records_in

    def test_missing_bronze_returns_empty_result(self, settings, run_context, store):
        result = build_silver("transaction", run_context, settings, store=store)
        assert result.records_in == 0
        assert result.records_out == 0
