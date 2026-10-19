"""Pydantic data contracts for record-level validation.

These models are the *enforcement* side of the schemas declared in
:mod:`src.core.schema`. They are applied where records arrive one at a time and
must be judged individually - principally REST API ingestion and streaming
events - as opposed to the vectorised, set-based checks that
:mod:`src.data_quality` runs over a whole dataframe.

Two rules shape the design:

1. **Validation never raises out of the pipeline.** :func:`validate_records`
   returns ``(valid, rejected)``; rejected records carry a structured error code
   and continue on to quarantine. One malformed payload must not abort a load.
2. **Error codes are stable strings**, not Pydantic prose, so quarantine rows
   stay queryable (``WHERE error_code = 'INVALID_CURRENCY'``) as the library's
   messages change between versions.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# error codes - the closed vocabulary written to the quarantine table
# ---------------------------------------------------------------------------


class ErrorCode:
    NULL_REQUIRED_FIELD = "NULL_REQUIRED_FIELD"
    INVALID_AMOUNT = "INVALID_AMOUNT"
    INVALID_CURRENCY = "INVALID_CURRENCY"
    DUPLICATE_RECORD = "DUPLICATE_RECORD"
    UNKNOWN_ACCOUNT = "UNKNOWN_ACCOUNT"
    UNKNOWN_VENDOR = "UNKNOWN_VENDOR"
    UNKNOWN_CONTRACT = "UNKNOWN_CONTRACT"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    INVALID_ENUM_VALUE = "INVALID_ENUM_VALUE"
    AMOUNT_EXCEEDS_LIMIT = "AMOUNT_EXCEEDS_LIMIT"
    REFERENTIAL_INTEGRITY = "REFERENTIAL_INTEGRITY"

    ALL: ClassVar[tuple[str, ...]] = (
        NULL_REQUIRED_FIELD,
        INVALID_AMOUNT,
        INVALID_CURRENCY,
        DUPLICATE_RECORD,
        UNKNOWN_ACCOUNT,
        UNKNOWN_VENDOR,
        UNKNOWN_CONTRACT,
        INVALID_TIMESTAMP,
        FUTURE_TIMESTAMP,
        SCHEMA_MISMATCH,
        INVALID_ENUM_VALUE,
        AMOUNT_EXCEEDS_LIMIT,
        REFERENTIAL_INTEGRITY,
    )


#: Currencies the platform will settle. Anything else is quarantined rather
#: than silently coerced - an unknown currency means an unknown amount.
ALLOWED_CURRENCIES: frozenset[str] = frozenset({"USD", "EUR", "GBP", "CAD", "JPY", "AUD", "CHF"})

#: Fixed reference rates to USD. A real platform would join a rates dimension
#: with effective dating; fixed rates keep the demo deterministic and are
#: documented as such in docs/data_dictionary.md.
USD_REFERENCE_RATES: dict[str, Decimal] = {
    "USD": Decimal("1.00"),
    "EUR": Decimal("1.09"),
    "GBP": Decimal("1.27"),
    "CAD": Decimal("0.74"),
    "JPY": Decimal("0.0067"),
    "AUD": Decimal("0.66"),
    "CHF": Decimal("1.12"),
}

#: Tolerance for clock skew between source systems before a timestamp counts as
#: "in the future". Without it, a source machine running two minutes fast would
#: quarantine perfectly good records.
FUTURE_TOLERANCE = timedelta(minutes=5)

MAX_TRANSACTION_AMOUNT = Decimal("50000000")


class ContractViolation(Exception):
    """Raised inside validators; converted to a structured rejection by
    :func:`validate_records`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# shared normalisers
# ---------------------------------------------------------------------------


def _clean_string(value: Any) -> str | None:
    """Trim and collapse whitespace; empty becomes ``None``.

    The generator injects padded and mixed-case values, and every source system
    in the wild has some of this. Normalising at the contract boundary means
    downstream joins do not have to.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "nan", "n/a", "-"}:
        return None
    return " ".join(text.split())


def parse_timestamp(value: Any) -> datetime:
    """Parse the timestamp formats seen across the source systems.

    Sources disagree on format (ISO-8601 with and without ``T``, US
    ``MM/DD/YYYY``, epoch seconds), so the parser tries each in turn instead of
    assuming one. An unparseable value raises ``INVALID_TIMESTAMP``.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, int | float) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise ContractViolation(ErrorCode.INVALID_TIMESTAMP, f"Bad epoch {value!r}") from exc

    text = _clean_string(value)
    if text is None:
        raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "timestamp is required")

    normalised = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalised)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ContractViolation(ErrorCode.INVALID_TIMESTAMP, f"Unparseable timestamp {text!r}")


def parse_amount(value: Any) -> Decimal:
    """Parse a monetary amount, tolerating currency symbols and thousands separators."""
    if value is None:
        raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "amount is required")
    if isinstance(value, Decimal):
        candidate = value
    else:
        text = str(value).strip().replace(",", "").replace("$", "").replace(" ", "")
        if text.startswith("(") and text.endswith(")"):  # accounting negative
            text = "-" + text[1:-1]
        if not text:
            raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "amount is required")
        try:
            candidate = Decimal(text)
        except (InvalidOperation, ValueError) as exc:
            raise ContractViolation(
                ErrorCode.INVALID_AMOUNT, f"Unparseable amount {value!r}"
            ) from exc
    if not candidate.is_finite():
        raise ContractViolation(ErrorCode.INVALID_AMOUNT, f"Non-finite amount {value!r}")
    return candidate.quantize(Decimal("0.01"))


def normalise_currency(value: Any) -> str:
    code = _clean_string(value)
    if code is None:
        raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "currency is required")
    code = code.upper().replace("$", "").strip()
    aliases = {"US": "USD", "USDOLLAR": "USD", "EURO": "EUR", "POUND": "GBP", "GBR": "GBP"}
    code = aliases.get(code, code)
    if code not in ALLOWED_CURRENCIES:
        raise ContractViolation(ErrorCode.INVALID_CURRENCY, f"Unsupported currency {code!r}")
    return code


def to_usd(amount: Decimal, currency: str) -> Decimal:
    """Convert to USD using the fixed reference rates."""
    rate = USD_REFERENCE_RATES.get(currency.upper())
    if rate is None:
        raise ContractViolation(ErrorCode.INVALID_CURRENCY, f"No USD rate for {currency!r}")
    return (amount * rate).quantize(Decimal("0.01"))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class CanonicalModel(BaseModel):
    """Base for every contract model.

    ``extra="allow"`` is intentional: an upstream system adding a field must not
    break ingestion. New fields land in Bronze untouched and are promoted to
    Silver only once declared in :mod:`src.core.schema` - additive schema
    evolution without a deployment race.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True, populate_by_name=True)


class TransactionContract(CanonicalModel):
    """Record-level contract for a financial transaction."""

    transaction_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)
    contract_id: str | None = None
    department_id: str | None = None
    agency_id: str | None = None
    amount: Decimal
    currency: str
    amount_usd: Decimal | None = None
    transaction_timestamp: datetime
    posted_timestamp: datetime | None = None
    transaction_type: str | None = None
    payment_method: str | None = None
    merchant_category: str | None = None
    description: str | None = None
    location_id: str | None = None
    channel: str | None = None
    status: str | None = None

    @field_validator("transaction_id", "account_id", "vendor_id", mode="before")
    @classmethod
    def _require_identifier(cls, value: Any) -> str:
        cleaned = _clean_string(value)
        if cleaned is None:
            raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "identifier is required")
        return cleaned

    @field_validator(
        "contract_id",
        "department_id",
        "agency_id",
        "location_id",
        "description",
        "merchant_category",
        mode="before",
    )
    @classmethod
    def _optional_string(cls, value: Any) -> str | None:
        return _clean_string(value)

    @field_validator("transaction_type", "payment_method", "channel", "status", mode="before")
    @classmethod
    def _upper_enum(cls, value: Any) -> str | None:
        cleaned = _clean_string(value)
        return cleaned.upper().replace(" ", "_") if cleaned else None

    @field_validator("amount", mode="before")
    @classmethod
    def _amount(cls, value: Any) -> Decimal:
        parsed = parse_amount(value)
        if parsed <= 0:
            raise ContractViolation(
                ErrorCode.INVALID_AMOUNT, f"Amount must be positive, got {parsed}"
            )
        if parsed > MAX_TRANSACTION_AMOUNT:
            raise ContractViolation(
                ErrorCode.AMOUNT_EXCEEDS_LIMIT,
                f"Amount {parsed} exceeds the {MAX_TRANSACTION_AMOUNT} platform limit",
            )
        return parsed

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, value: Any) -> str:
        return normalise_currency(value)

    @field_validator("transaction_timestamp", mode="before")
    @classmethod
    def _timestamp(cls, value: Any) -> datetime:
        parsed = parse_timestamp(value)
        if parsed > datetime.now(UTC) + FUTURE_TOLERANCE:
            raise ContractViolation(
                ErrorCode.FUTURE_TIMESTAMP, f"Timestamp {parsed.isoformat()} is in the future"
            )
        return parsed

    @field_validator("posted_timestamp", mode="before")
    @classmethod
    def _posted(cls, value: Any) -> datetime | None:
        if value is None or _clean_string(value) is None:
            return None
        return parse_timestamp(value)

    def enriched(self) -> dict[str, Any]:
        """Model payload plus the derived columns Silver expects."""
        payload = self.model_dump()
        payload["amount_usd"] = to_usd(self.amount, self.currency)
        payload["transaction_date"] = self.transaction_timestamp.date()
        # US federal fiscal year starts on 1 October.
        ts = self.transaction_timestamp
        payload["fiscal_year"] = ts.year + 1 if ts.month >= 10 else ts.year
        return payload


class AccountContract(CanonicalModel):
    account_id: str = Field(min_length=1)
    account_number: str | None = None
    department_id: str = Field(min_length=1)
    account_type: str | None = None
    currency: str | None = None
    opened_date: date | None = None
    status: str | None = None
    credit_limit_usd: Decimal | None = None
    owner_name: str | None = None
    region: str | None = None

    @field_validator("account_id", "department_id", mode="before")
    @classmethod
    def _require(cls, value: Any) -> str:
        cleaned = _clean_string(value)
        if cleaned is None:
            raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "identifier is required")
        return cleaned

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, value: Any) -> str | None:
        return normalise_currency(value) if _clean_string(value) else None

    @field_validator("opened_date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> date | None:
        if _clean_string(value) is None:
            return None
        return parse_timestamp(value).date()


class VendorContract(CanonicalModel):
    vendor_id: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)
    vendor_legal_name: str | None = None
    tax_id: str | None = None
    vendor_type: str | None = None
    industry_code: str | None = None
    contact_email: str | None = None
    phone: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str | None = None
    registration_date: date | None = None
    is_active: bool | None = None
    risk_flag: str | None = None

    @field_validator("vendor_id", "vendor_name", mode="before")
    @classmethod
    def _require(cls, value: Any) -> str:
        cleaned = _clean_string(value)
        if cleaned is None:
            raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "field is required")
        return cleaned

    @field_validator("contact_email", mode="before")
    @classmethod
    def _email(cls, value: Any) -> str | None:
        """Malformed addresses are nulled, not rejected.

        A bad contact email is a completeness problem, not a reason to discard a
        vendor record - the null is visible to the completeness check and the
        vendor still joins correctly.
        """
        cleaned = _clean_string(value)
        if cleaned is None:
            return None
        candidate = cleaned.lower()
        if candidate.count("@") != 1:
            return None
        local, _, domain = candidate.partition("@")
        if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
            return None
        return candidate

    @field_validator("country", "state", mode="before")
    @classmethod
    def _upper_code(cls, value: Any) -> str | None:
        cleaned = _clean_string(value)
        return cleaned.upper() if cleaned else None

    @field_validator("registration_date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> date | None:
        if _clean_string(value) is None:
            return None
        return parse_timestamp(value).date()

    @field_validator("is_active", mode="before")
    @classmethod
    def _bool(cls, value: Any) -> bool | None:
        if value is None or _clean_string(value) is None:
            return None
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "active"}


class AgencyContract(CanonicalModel):
    agency_id: str = Field(min_length=1)
    agency_name: str = Field(min_length=1)
    agency_abbreviation: str | None = None
    agency_type: str | None = None
    toptier_code: str | None = None
    established_year: int | None = None
    headquarters_state: str | None = None
    annual_budget_usd: Decimal | None = None

    @field_validator("agency_id", "agency_name", mode="before")
    @classmethod
    def _require(cls, value: Any) -> str:
        cleaned = _clean_string(value)
        if cleaned is None:
            raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "field is required")
        return cleaned


class ContractContract(CanonicalModel):
    """Contract (award) records. Named awkwardly because 'contract' is both the
    business entity and the data-contract concept."""

    contract_id: str = Field(min_length=1)
    award_id: str | None = None
    vendor_id: str = Field(min_length=1)
    agency_id: str = Field(min_length=1)
    contract_type: str | None = None
    contract_value_usd: Decimal | None = None
    start_date: date | None = None
    end_date: date | None = None
    status: str | None = None
    description: str | None = None
    competition_type: str | None = None

    @field_validator("contract_id", "vendor_id", "agency_id", mode="before")
    @classmethod
    def _require(cls, value: Any) -> str:
        cleaned = _clean_string(value)
        if cleaned is None:
            raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "identifier is required")
        return cleaned

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> date | None:
        if _clean_string(value) is None:
            return None
        return parse_timestamp(value).date()

    @field_validator("contract_value_usd", mode="before")
    @classmethod
    def _value(cls, value: Any) -> Decimal | None:
        if value is None or _clean_string(value) is None:
            return None
        parsed = parse_amount(value)
        if parsed < 0:
            raise ContractViolation(ErrorCode.INVALID_AMOUNT, "Contract value cannot be negative")
        return parsed


class TransactionEventContract(CanonicalModel):
    """Streaming envelope. Event time is mandatory; ingest time is stamped by us."""

    event_id: str = Field(min_length=1)
    event_type: str = "TRANSACTION_CREATED"
    event_timestamp: datetime
    ingest_timestamp: datetime | None = None
    producer: str | None = None
    schema_version: str = "1.0"
    transaction_id: str = Field(min_length=1)
    account_id: str | None = None
    vendor_id: str | None = None
    amount: Decimal | None = None
    currency: str | None = None
    location_id: str | None = None
    channel: str | None = None

    @field_validator("event_id", "transaction_id", mode="before")
    @classmethod
    def _require(cls, value: Any) -> str:
        cleaned = _clean_string(value)
        if cleaned is None:
            raise ContractViolation(ErrorCode.NULL_REQUIRED_FIELD, "identifier is required")
        return cleaned

    @field_validator("event_timestamp", mode="before")
    @classmethod
    def _event_time(cls, value: Any) -> datetime:
        return parse_timestamp(value)

    @field_validator("amount", mode="before")
    @classmethod
    def _amount(cls, value: Any) -> Decimal | None:
        if value is None or _clean_string(value) is None:
            return None
        parsed = parse_amount(value)
        if parsed <= 0:
            raise ContractViolation(ErrorCode.INVALID_AMOUNT, "Event amount must be positive")
        return parsed

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, value: Any) -> str | None:
        return normalise_currency(value) if _clean_string(value) else None


#: Canonical dataset name -> contract model.
CONTRACT_MODELS: dict[str, type[CanonicalModel]] = {
    "transaction": TransactionContract,
    "account": AccountContract,
    "vendor": VendorContract,
    "agency": AgencyContract,
    "contract": ContractContract,
    "transaction_event": TransactionEventContract,
}


# ---------------------------------------------------------------------------
# batch validation
# ---------------------------------------------------------------------------


class RejectedRecord(BaseModel):
    """A record that failed its contract, in the shape the quarantine expects."""

    model_config = ConfigDict(extra="allow")

    raw_record: dict[str, Any]
    error_code: str
    error_message: str
    field_name: str | None = None
    row_number: int | None = None


def _first_violation(exc: ValidationError) -> tuple[str, str, str | None]:
    """Extract a stable error code, message and field from a Pydantic error.

    Pydantic wraps exceptions raised inside validators, so a
    :class:`ContractViolation` surfaces as a ``value_error`` whose context holds
    the original. We dig it out to keep our own error codes.
    """
    for error in exc.errors():
        original = error.get("ctx", {}).get("error")
        field_name = ".".join(str(part) for part in error.get("loc", ())) or None
        if isinstance(original, ContractViolation):
            return original.code, original.message, field_name
        error_type = error.get("type", "")
        if error_type in {"missing", "string_too_short"}:
            return (
                ErrorCode.NULL_REQUIRED_FIELD,
                f"{field_name or 'field'}: {error.get('msg', 'is required')}",
                field_name,
            )
        return (
            ErrorCode.SCHEMA_MISMATCH,
            f"{field_name or 'record'}: {error.get('msg', 'failed validation')}",
            field_name,
        )
    return ErrorCode.SCHEMA_MISMATCH, str(exc), None


def validate_record(
    model: type[CanonicalModel], record: dict[str, Any]
) -> tuple[CanonicalModel | None, RejectedRecord | None]:
    """Validate one record. Returns ``(model, None)`` or ``(None, rejection)``."""
    try:
        return model.model_validate(record), None
    except ValidationError as exc:
        code, message, field_name = _first_violation(exc)
        return None, RejectedRecord(
            raw_record=record, error_code=code, error_message=message, field_name=field_name
        )
    except ContractViolation as exc:  # raised outside a field validator
        return None, RejectedRecord(
            raw_record=record, error_code=exc.code, error_message=exc.message
        )


def validate_records(
    dataset: str, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[RejectedRecord]]:
    """Validate a batch, partitioning it into accepted payloads and rejections.

    Never raises for record-level problems - that is the whole point. A caller
    writes ``accepted`` onward and ``rejected`` to quarantine.
    """
    model = CONTRACT_MODELS.get(dataset)
    if model is None:
        raise KeyError(f"No contract model registered for dataset {dataset!r}")

    accepted: list[dict[str, Any]] = []
    rejected: list[RejectedRecord] = []
    for row_number, record in enumerate(records):
        validated, rejection = validate_record(model, record)
        if validated is not None:
            payload = (
                validated.enriched()
                if isinstance(validated, TransactionContract)
                else validated.model_dump()
            )
            accepted.append(payload)
        elif rejection is not None:
            rejection.row_number = row_number
            rejected.append(rejection)
    return accepted, rejected
