"""Data contracts and canonical schema registry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.core.contracts import (
    ALLOWED_CURRENCIES,
    ErrorCode,
    TransactionContract,
    normalise_currency,
    parse_amount,
    parse_timestamp,
    to_usd,
    validate_records,
)
from src.core.schema import SCHEMAS, get_schema, pluralise, table_name


class TestCanonicalSchemas:
    def test_every_schema_has_a_business_key(self):
        for name, schema in SCHEMAS.items():
            assert schema.business_key, f"{name} has no business key"
            for key in schema.business_key:
                assert key in schema.column_names, f"{name}: key {key} is not a declared field"

    def test_required_columns_are_not_nullable(self):
        transaction = get_schema("transaction")
        assert "transaction_id" in transaction.required_columns
        assert "amount" in transaction.required_columns
        assert "contract_id" not in transaction.required_columns

    def test_foreign_keys_reference_known_datasets(self):
        for name, schema in SCHEMAS.items():
            for foreign_key in schema.foreign_keys:
                assert foreign_key.references_dataset in SCHEMAS, (
                    f"{name}.{foreign_key.column} references unknown "
                    f"{foreign_key.references_dataset}"
                )

    def test_hash_columns_exclude_the_identifier(self):
        """Two records with identical content under different ids must collide.

        This is what lets deduplication catch a re-issued payment.
        """
        transaction = get_schema("transaction")
        assert "transaction_id" not in transaction.hash_columns

    def test_contract_export_is_valid_json_schema(self):
        contract = get_schema("transaction").to_contract()
        assert contract["type"] == "object"
        assert "transaction_id" in contract["properties"]
        assert "transaction_id" in contract["required"]
        # Bronze must tolerate fields we have not declared yet.
        assert contract["additionalProperties"] is True

    @pytest.mark.parametrize(
        ("singular", "expected"),
        [("agency", "agencies"), ("vendor", "vendors"), ("transaction", "transactions")],
    )
    def test_pluralisation(self, singular: str, expected: str):
        assert pluralise(singular) == expected

    def test_table_naming(self):
        assert table_name("bronze", "agency") == "bronze_agencies"
        assert table_name("silver", "transaction") == "silver_transactions"


class TestParsers:
    @pytest.mark.parametrize(
        "value",
        ["2026-01-05T10:11:12", "2026-01-05 10:11:12", "01/05/2026", "2026-01-05", 1767607872],
    )
    def test_timestamp_formats(self, value):
        assert isinstance(parse_timestamp(value), datetime)

    def test_unparseable_timestamp_raises_with_our_code(self):
        from src.core.contracts import ContractViolation

        with pytest.raises(ContractViolation) as info:
            parse_timestamp("not-a-date")
        assert info.value.code == ErrorCode.INVALID_TIMESTAMP

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("$1,234.50", "1234.50"), ("1234.5", "1234.50"), ("(500)", "-500.00"), (99, "99.00")],
    )
    def test_amount_parsing(self, value, expected):
        assert parse_amount(value) == Decimal(expected)

    @pytest.mark.parametrize(
        ("value", "expected"), [("usd", "USD"), (" eur ", "EUR"), ("US", "USD"), ("EURO", "EUR")]
    )
    def test_currency_normalisation(self, value, expected):
        assert normalise_currency(value) == expected

    def test_unknown_currency_is_rejected(self):
        from src.core.contracts import ContractViolation

        with pytest.raises(ContractViolation) as info:
            normalise_currency("XXX")
        assert info.value.code == ErrorCode.INVALID_CURRENCY

    def test_usd_conversion(self):
        assert to_usd(Decimal("100"), "USD") == Decimal("100.00")
        assert to_usd(Decimal("100"), "EUR") == Decimal("109.00")

    def test_every_allowed_currency_has_a_rate(self):
        from src.core.contracts import USD_REFERENCE_RATES

        assert set(USD_REFERENCE_RATES) >= ALLOWED_CURRENCIES


class TestTransactionContract:
    def _record(self, **overrides):
        base = {
            "transaction_id": "TXN-1",
            "account_id": "ACC-1",
            "vendor_id": "VEN-1",
            "amount": "100.00",
            "currency": "USD",
            "transaction_timestamp": "2026-01-05T10:00:00",
        }
        base.update(overrides)
        return base

    def test_valid_record_is_accepted_and_enriched(self):
        accepted, rejected = validate_records("transaction", [self._record()])
        assert not rejected
        assert accepted[0]["amount_usd"] == Decimal("100.00")
        assert accepted[0]["fiscal_year"] == 2026

    def test_fiscal_year_rolls_over_in_october(self):
        model = TransactionContract.model_validate(
            self._record(transaction_timestamp="2025-10-01T00:00:00")
        )
        assert model.enriched()["fiscal_year"] == 2026

    def test_identifiers_are_trimmed(self):
        accepted, _ = validate_records("transaction", [self._record(transaction_id="  TXN-9  ")])
        assert accepted[0]["transaction_id"] == "TXN-9"

    @pytest.mark.parametrize(
        ("overrides", "expected_code"),
        [
            ({"amount": "-500"}, ErrorCode.INVALID_AMOUNT),
            ({"amount": "0"}, ErrorCode.INVALID_AMOUNT),
            ({"amount": "not-a-number"}, ErrorCode.INVALID_AMOUNT),
            ({"amount": "999999999"}, ErrorCode.AMOUNT_EXCEEDS_LIMIT),
            ({"currency": "XXX"}, ErrorCode.INVALID_CURRENCY),
            ({"account_id": None}, ErrorCode.NULL_REQUIRED_FIELD),
            ({"vendor_id": ""}, ErrorCode.NULL_REQUIRED_FIELD),
            ({"transaction_timestamp": "garbage"}, ErrorCode.INVALID_TIMESTAMP),
            ({"transaction_timestamp": "2099-01-01"}, ErrorCode.FUTURE_TIMESTAMP),
        ],
    )
    def test_invalid_records_get_the_right_error_code(self, overrides, expected_code):
        accepted, rejected = validate_records("transaction", [self._record(**overrides)])
        assert not accepted
        assert rejected[0].error_code == expected_code

    def test_small_clock_skew_is_tolerated(self):
        """A source machine two minutes fast must not quarantine good records."""
        near_future = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
        accepted, rejected = validate_records(
            "transaction", [self._record(transaction_timestamp=near_future)]
        )
        assert accepted and not rejected

    def test_unknown_fields_are_preserved_not_rejected(self):
        """Additive source changes must not break ingestion."""
        accepted, rejected = validate_records(
            "transaction", [self._record(brand_new_field="surprise")]
        )
        assert not rejected
        assert accepted[0]["brand_new_field"] == "surprise"

    def test_batch_partitions_valid_from_invalid(self):
        records = [self._record(transaction_id=f"TXN-{i}") for i in range(5)]
        records.append(self._record(transaction_id="TXN-BAD", amount="-1"))
        accepted, rejected = validate_records("transaction", records)
        assert len(accepted) == 5
        assert len(rejected) == 1
        assert rejected[0].row_number == 5


class TestVendorContract:
    def test_malformed_email_is_nulled_not_rejected(self):
        """A bad contact address is a completeness problem, not a reason to
        discard a vendor."""
        accepted, rejected = validate_records(
            "vendor",
            [{"vendor_id": "VEN-1", "vendor_name": "Acme", "contact_email": "not-an-email"}],
        )
        assert not rejected
        assert accepted[0]["contact_email"] is None

    def test_valid_email_survives(self):
        accepted, _ = validate_records(
            "vendor",
            [{"vendor_id": "VEN-1", "vendor_name": "Acme", "contact_email": "A@Example.COM"}],
        )
        assert accepted[0]["contact_email"] == "a@example.com"
