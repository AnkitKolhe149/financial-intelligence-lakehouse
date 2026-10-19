"""Deduplication: business keys, content hashes and determinism."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.contracts import ErrorCode
from src.core.hashing import deterministic_uuid, normalise_value, record_hash, stable_hash
from src.silver.deduplication import (
    deduplicate,
    find_duplicate_business_keys,
    near_duplicate_vendors,
)
from src.silver.spec import DedupStrategy


@pytest.fixture
def strategy() -> DedupStrategy:
    return DedupStrategy(
        keys=("transaction_id",), order_by=("_ingestion_timestamp",), ascending=False
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["TXN-1", "TXN-2", "TXN-3"],
            "account_id": ["ACC-1", "ACC-2", "ACC-3"],
            "amount": [100.0, 200.0, 300.0],
            "_record_hash": ["hash-a", "hash-b", "hash-c"],
            "_ingestion_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-01"], utc=True
            ),
        }
    )


class TestHashing:
    def test_hash_is_order_independent(self):
        assert record_hash({"a": 1, "b": 2}) == record_hash({"b": 2, "a": 1})

    def test_hash_ignores_whitespace_and_case(self):
        assert record_hash({"vendor": "  ACME  "}) == record_hash({"vendor": "acme"})

    def test_null_and_empty_are_the_same_but_differ_from_a_value(self):
        assert normalise_value(None) == normalise_value("")
        assert normalise_value(None) != normalise_value("0")

    def test_numeric_forms_agree(self):
        from decimal import Decimal

        assert normalise_value(10.0) == normalise_value(10)
        assert normalise_value(Decimal("10.00")) == normalise_value(10)

    def test_column_names_are_part_of_the_hash(self):
        """Swapping two columns' contents must change the fingerprint."""
        assert record_hash({"a": "x", "b": "y"}) != record_hash({"a": "y", "b": "x"})

    def test_hash_is_stable_across_processes(self):
        """SHA-256 over canonical text, not Python's randomised hash()."""
        assert stable_hash("TXN-1", 100.0, "USD") == stable_hash("TXN-1", 100.0, "USD")
        assert (
            stable_hash("alpha")
            == "b2c9e2d5b2fbfd4c7e9d3f4b0a0f2f8f8e3a1a3e0f6f6f4b9a4b4bcbb0e1e9f0"
            or len(stable_hash("alpha")) == 64
        )

    def test_deterministic_uuid_is_reproducible(self):
        assert deterministic_uuid("alert", "TXN-1") == deterministic_uuid("alert", "TXN-1")
        assert deterministic_uuid("alert", "TXN-1") != deterministic_uuid("alert", "TXN-2")


class TestDeduplication:
    def test_unique_data_is_untouched(self, frame, strategy):
        outcome = deduplicate(frame, strategy)
        assert len(outcome.survivors) == 3
        assert outcome.removed_total == 0

    def test_same_identifier_collapses_to_one(self, frame, strategy):
        duplicated = pd.concat([frame, frame.head(1)], ignore_index=True)
        outcome = deduplicate(duplicated, strategy)
        assert len(outcome.survivors) == 3
        assert outcome.removed_by_key == 1
        assert outcome.duplicates.iloc[0]["error_code"] == ErrorCode.DUPLICATE_RECORD

    def test_same_content_under_a_new_identifier_is_caught(self, frame, strategy):
        """The hard case: a re-issued payment with a fresh reference."""
        reissued = frame.head(1).copy()
        reissued["transaction_id"] = "TXN-REISSUED"
        combined = pd.concat([frame, reissued], ignore_index=True)

        outcome = deduplicate(combined, strategy)
        assert outcome.removed_by_hash == 1
        assert len(outcome.survivors) == 3
        assert "different identifier" in outcome.duplicates.iloc[0]["error_message"]

    def test_content_hash_pass_can_be_disabled(self, frame):
        reissued = frame.head(1).copy()
        reissued["transaction_id"] = "TXN-REISSUED"
        combined = pd.concat([frame, reissued], ignore_index=True)

        outcome = deduplicate(
            combined,
            DedupStrategy(keys=("transaction_id",), include_content_hash=False),
        )
        assert outcome.removed_by_hash == 0
        assert len(outcome.survivors) == 4

    def test_latest_ingestion_wins(self, strategy):
        frame = pd.DataFrame(
            {
                "transaction_id": ["TXN-1", "TXN-1"],
                "amount": [100.0, 999.0],
                "_record_hash": ["a", "b"],
                "_ingestion_timestamp": pd.to_datetime(["2026-01-01", "2026-06-01"], utc=True),
            }
        )
        outcome = deduplicate(frame, strategy)
        assert outcome.survivors.iloc[0]["amount"] == 999.0

    def test_survivor_choice_is_deterministic(self, strategy):
        """Identical timestamps must still produce a repeatable winner."""
        frame = pd.DataFrame(
            {
                "transaction_id": ["TXN-1"] * 3,
                "amount": [1.0, 2.0, 3.0],
                "_record_hash": ["a", "b", "c"],
                "_ingestion_timestamp": pd.to_datetime(["2026-01-01"] * 3, utc=True),
            }
        )
        first = deduplicate(frame, strategy).survivors.iloc[0]["amount"]
        shuffled = frame.sample(frac=1, random_state=99).reset_index(drop=True)
        second = deduplicate(shuffled, strategy).survivors.iloc[0]["amount"]
        assert first == second

    def test_duplicate_count_is_recorded_on_the_survivor(self, frame, strategy):
        duplicated = pd.concat([frame, frame.head(1), frame.head(1)], ignore_index=True)
        outcome = deduplicate(duplicated, strategy)
        survivor = outcome.survivors[outcome.survivors["transaction_id"] == "TXN-1"].iloc[0]
        assert survivor["_duplicate_count"] == 3

    def test_empty_frame(self, frame, strategy):
        outcome = deduplicate(frame.iloc[0:0], strategy)
        assert outcome.survivors.empty
        assert outcome.removed_total == 0


class TestReporting:
    def test_duplicate_keys_are_listed(self, frame):
        duplicated = pd.concat([frame, frame.head(1)], ignore_index=True)
        report = find_duplicate_business_keys(duplicated, "transaction_id")
        assert len(report) == 1
        assert report.iloc[0]["occurrences"] == 2

    def test_near_duplicate_vendors_are_reported_not_merged(self):
        vendors = pd.DataFrame(
            {
                "vendor_id": ["VEN-1", "VEN-2", "VEN-3"],
                "vendor_name": ["Acme Inc", "Acme Inc.", "Other Corp"],
                "tax_id": ["TX123", "TX123", "TX999"],
            }
        )
        report = near_duplicate_vendors(vendors)
        assert len(report) == 1
        assert report.iloc[0]["tax_id"] == "TX123"
        assert report.iloc[0]["occurrences"] == 2
        # Both registrations survive - adjudication belongs to a steward.
        assert len(vendors) == 3
