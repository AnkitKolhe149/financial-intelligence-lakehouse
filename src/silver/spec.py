"""Declarative Silver transformation specifications.

Each spec states *what* the Silver form of a dataset is - which columns are
trimmed, which are upper-cased, how duplicates are resolved, which foreign keys
are enforced - without stating *how* to compute it. That separation is what lets
the pandas engine and the Spark engine produce identical results: they are two
interpreters of the same specification, and the parity test in
``tests/test_engine_parity.py`` holds them to it.

It also makes the transformation rules reviewable. A data steward can read this
file and see exactly what normalisation the platform applies, without reading
any pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.schema import DatasetSchema, ForeignKey, get_schema


@dataclass(frozen=True)
class DedupStrategy:
    """How duplicate records are collapsed.

    ``keys``
        Columns that identify "the same record". Two levels are used in
        practice: the business key (same id) and the content hash (same content
        under a different id) - the second is the one that catches a source
        system re-issuing a payment with a fresh reference.
    ``order_by`` / ``ascending``
        Which row survives. Deterministic ordering is essential: a
        non-deterministic survivor makes the whole pipeline non-reproducible.
    """

    keys: tuple[str, ...]
    order_by: tuple[str, ...] = ()
    ascending: bool = False
    #: Also collapse records that share ``_record_hash`` even when their
    #: business keys differ.
    include_content_hash: bool = True


@dataclass(frozen=True)
class NormalisationRules:
    """Column-level cleaning applied before validation."""

    #: Strip surrounding whitespace and collapse internal runs.
    trim: tuple[str, ...] = ()
    #: Upper-case (codes, identifiers, currencies, enums).
    upper: tuple[str, ...] = ()
    #: Lower-case (emails, urls).
    lower: tuple[str, ...] = ()
    #: Title-case for display names, applied after trimming.
    title: tuple[str, ...] = ()
    #: Values treated as null wherever they appear, in addition to empty strings.
    null_tokens: tuple[str, ...] = ("null", "none", "nan", "n/a", "na", "-", "unknown", "tbd")


@dataclass(frozen=True)
class SilverSpec:
    """Everything needed to turn one Bronze table into its Silver form."""

    dataset: str
    source_table: str
    target_table: str
    normalisation: NormalisationRules
    dedup: DedupStrategy
    #: Columns whose null-ness sends the record to quarantine. Defaults to the
    #: schema's NOT NULL columns; stated explicitly so a steward can widen it.
    required_columns: tuple[str, ...] = ()
    #: Numeric columns that must be strictly positive.
    positive_columns: tuple[str, ...] = ()
    #: Timestamp columns to parse; the first is the event-time column.
    timestamp_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    #: Reference datasets needed to enforce foreign keys, as
    #: ``{silver_table: key_column}``.
    reference_tables: dict[str, str] = field(default_factory=dict)
    partition_column: str | None = None
    #: Derived columns computed after cleaning (documented in the data dictionary).
    derived_columns: tuple[str, ...] = ()
    description: str = ""

    @property
    def schema(self) -> DatasetSchema:
        return get_schema(self.dataset)

    @property
    def foreign_keys(self) -> tuple[ForeignKey, ...]:
        return self.schema.foreign_keys

    def effective_required(self) -> list[str]:
        return list(self.required_columns or tuple(self.schema.required_columns))

    def transformation_names(self) -> list[str]:
        """The transformation list recorded in lineage for this dataset."""
        names = ["schema_alignment", "type_casting", "string_normalisation", "null_policy"]
        if self.timestamp_columns:
            names.append("timestamp_parsing")
        if "currency" in self.normalisation.upper:
            names.append("currency_normalisation")
        names.append("deduplication")
        if self.foreign_keys:
            names.append("referential_integrity")
        if self.derived_columns:
            names.append("derived_columns")
        names.append("quarantine_invalid_records")
        return names


# ---------------------------------------------------------------------------
# specifications
# ---------------------------------------------------------------------------

TRANSACTION_SPEC = SilverSpec(
    dataset="transaction",
    source_table="bronze_transactions",
    target_table="silver_transactions",
    description=(
        "The validated transaction fact. Amounts are converted to USD at fixed "
        "reference rates, timestamps are parsed to UTC, and records missing an "
        "account, vendor, amount, currency or timestamp are quarantined."
    ),
    normalisation=NormalisationRules(
        trim=(
            "transaction_id",
            "account_id",
            "vendor_id",
            "contract_id",
            "department_id",
            "agency_id",
            "description",
            "location_id",
            "merchant_category",
        ),
        upper=(
            "currency",
            "transaction_type",
            "payment_method",
            "channel",
            "status",
            "merchant_category",
        ),
    ),
    dedup=DedupStrategy(
        keys=("transaction_id",),
        # Latest ingestion wins, then the most recently posted record. Both
        # columns are always populated, so the ordering is total.
        order_by=("_ingestion_timestamp", "posted_timestamp"),
        ascending=False,
        include_content_hash=True,
    ),
    required_columns=(
        "transaction_id",
        "account_id",
        "vendor_id",
        "amount",
        "currency",
        "transaction_timestamp",
    ),
    positive_columns=("amount",),
    timestamp_columns=("transaction_timestamp", "posted_timestamp"),
    reference_tables={
        "silver_accounts": "account_id",
        "silver_vendors": "vendor_id",
        "silver_contracts": "contract_id",
    },
    # Monthly, not daily. See add_transaction_derivations for the measurement
    # that drove this: daily partitions produced ~17-row files and made metadata
    # operations slower than full scans.
    partition_column="transaction_month",
    derived_columns=(
        "amount_usd",
        "transaction_date",
        "transaction_month",
        "fiscal_year",
        "_is_late_arriving",
    ),
)

ACCOUNT_SPEC = SilverSpec(
    dataset="account",
    source_table="bronze_accounts",
    target_table="silver_accounts",
    description="Validated disbursement accounts, one row per account_id.",
    normalisation=NormalisationRules(
        trim=("account_id", "account_number", "department_id", "owner_name", "region"),
        upper=("account_type", "status", "currency", "region"),
    ),
    dedup=DedupStrategy(keys=("account_id",), order_by=("_ingestion_timestamp",)),
    required_columns=("account_id", "department_id"),
    date_columns=("opened_date",),
    reference_tables={"silver_departments": "department_id"},
)

VENDOR_SPEC = SilverSpec(
    dataset="vendor",
    source_table="bronze_vendors",
    target_table="silver_vendors",
    description=(
        "Validated vendor master. Names are trimmed and title-cased so that "
        "'  ACME  ' and 'acme' resolve to one entity for reporting, while the "
        "near-duplicate registrations remain visible as separate vendor_ids "
        "sharing a tax_id - a genuine entity-resolution problem, not one the "
        "pipeline pretends to have solved."
    ),
    normalisation=NormalisationRules(
        trim=("vendor_id", "vendor_name", "vendor_legal_name", "tax_id", "city", "postal_code"),
        upper=("country", "state", "vendor_type", "risk_flag"),
        lower=("contact_email",),
        title=("vendor_name",),
    ),
    dedup=DedupStrategy(keys=("vendor_id",), order_by=("_ingestion_timestamp",)),
    required_columns=("vendor_id", "vendor_name"),
    date_columns=("registration_date",),
)

AGENCY_SPEC = SilverSpec(
    dataset="agency",
    source_table="bronze_agencies",
    target_table="silver_agencies",
    description="Validated agency dimension.",
    normalisation=NormalisationRules(
        trim=("agency_id", "agency_name", "agency_abbreviation", "toptier_code"),
        upper=("agency_abbreviation", "agency_type", "headquarters_state"),
    ),
    dedup=DedupStrategy(keys=("agency_id",), order_by=("_ingestion_timestamp",)),
    required_columns=("agency_id", "agency_name"),
)

DEPARTMENT_SPEC = SilverSpec(
    dataset="department",
    source_table="bronze_departments",
    target_table="silver_departments",
    description="Validated department dimension, rolls up to agency.",
    normalisation=NormalisationRules(
        trim=("department_id", "department_name", "agency_id", "cost_center", "manager_name"),
        upper=("cost_center",),
    ),
    dedup=DedupStrategy(keys=("department_id",), order_by=("_ingestion_timestamp",)),
    required_columns=("department_id", "department_name", "agency_id"),
    reference_tables={"silver_agencies": "agency_id"},
)

CONTRACT_SPEC = SilverSpec(
    dataset="contract",
    source_table="bronze_contracts",
    target_table="silver_contracts",
    description="Validated award dimension linking vendors to agencies.",
    normalisation=NormalisationRules(
        trim=("contract_id", "award_id", "vendor_id", "agency_id", "description"),
        upper=("contract_type", "status", "competition_type"),
    ),
    dedup=DedupStrategy(keys=("contract_id",), order_by=("_ingestion_timestamp",)),
    required_columns=("contract_id", "vendor_id", "agency_id"),
    date_columns=("start_date", "end_date"),
    reference_tables={"silver_vendors": "vendor_id", "silver_agencies": "agency_id"},
)

BUDGET_SPEC = SilverSpec(
    dataset="budget",
    source_table="bronze_budgets",
    target_table="silver_budgets",
    description="Validated budget lines by fiscal year, agency, department and category.",
    normalisation=NormalisationRules(
        trim=("budget_id", "agency_id", "department_id"),
        upper=("budget_category", "quarter"),
    ),
    dedup=DedupStrategy(keys=("budget_id",), order_by=("_ingestion_timestamp",)),
    required_columns=("budget_id", "fiscal_year", "agency_id"),
    reference_tables={"silver_agencies": "agency_id"},
)

LOCATION_SPEC = SilverSpec(
    dataset="location",
    source_table="bronze_locations",
    target_table="silver_locations",
    description="Validated geography reference data.",
    normalisation=NormalisationRules(
        trim=("location_id", "city", "region"),
        upper=("state", "country", "region"),
    ),
    dedup=DedupStrategy(keys=("location_id",), order_by=("_ingestion_timestamp",)),
    required_columns=("location_id",),
)

TRANSACTION_EVENT_SPEC = SilverSpec(
    dataset="transaction_event",
    source_table="bronze_transaction_events",
    target_table="silver_transaction_events",
    description=(
        "Validated streaming events. Event time is the ordering key; the gap "
        "between event and ingest time is retained so lateness is measurable "
        "rather than assumed."
    ),
    normalisation=NormalisationRules(
        trim=("event_id", "transaction_id", "account_id", "vendor_id", "location_id"),
        upper=("event_type", "currency", "channel"),
    ),
    dedup=DedupStrategy(keys=("event_id",), order_by=("event_timestamp",)),
    required_columns=("event_id", "event_timestamp", "transaction_id"),
    timestamp_columns=("event_timestamp", "ingest_timestamp"),
    derived_columns=("lateness_seconds", "_is_late_arriving"),
)

#: Every Silver specification, in dependency order. Dimensions are built before
#: the fact table because the fact's referential-integrity checks read them.
SILVER_SPECS: dict[str, SilverSpec] = {
    "agency": AGENCY_SPEC,
    "department": DEPARTMENT_SPEC,
    "location": LOCATION_SPEC,
    "vendor": VENDOR_SPEC,
    "contract": CONTRACT_SPEC,
    "account": ACCOUNT_SPEC,
    "budget": BUDGET_SPEC,
    "transaction": TRANSACTION_SPEC,
    "transaction_event": TRANSACTION_EVENT_SPEC,
}

#: Build order. The fact table must come last.
SILVER_BUILD_ORDER: tuple[str, ...] = (
    "agency",
    "department",
    "location",
    "vendor",
    "contract",
    "account",
    "budget",
    "transaction",
)


def get_silver_spec(dataset: str) -> SilverSpec:
    try:
        return SILVER_SPECS[dataset]
    except KeyError as exc:
        raise KeyError(
            f"No Silver specification for {dataset!r}; known: {', '.join(sorted(SILVER_SPECS))}"
        ) from exc
