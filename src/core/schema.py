"""Canonical schema registry - one definition, many consumers.

The heterogeneous sources (CSV, JSON, XML, Excel, REST) all describe the same
business domain with different field names and types. Rather than scatter that
knowledge across five ingestion modules, each canonical dataset is declared
**once** here, and everything else derives from it:

* Silver type casting reads the declared types.
* Data-quality completeness/validity checks read ``nullable`` and ``domain``.
* Deduplication reads ``business_key`` and ``hash_columns``.
* Referential-integrity checks read ``foreign_keys``.
* ``docs/data_dictionary.md`` is *generated* from these descriptions, so the
  documentation cannot drift away from the code.
* ``schemas/*.json`` data contracts are exported from the same objects.

Adding a column is therefore a one-line change that automatically propagates to
validation, documentation and the published contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# audit columns
# ---------------------------------------------------------------------------

# Prefixed with an underscore so they never collide with a business column and
# are trivially strippable when publishing Gold datasets.
BRONZE_AUDIT_COLUMNS: dict[str, str] = {
    "_source_system": "string",
    "_source_file": "string",
    "_source_format": "string",
    "_ingestion_timestamp": "timestamp",
    "_ingestion_run_id": "string",
    "_record_hash": "string",
    "_schema_version": "string",
    "_source_row_number": "int",
}

SILVER_AUDIT_COLUMNS: dict[str, str] = {
    "_source_system": "string",
    "_record_hash": "string",
    "_ingestion_run_id": "string",
    "_silver_run_id": "string",
    "_processed_timestamp": "timestamp",
    "_schema_version": "string",
    "_is_late_arriving": "bool",
}

CANONICAL_TYPES = {
    "string",
    "int",
    "long",
    "float",
    "decimal",
    "bool",
    "date",
    "timestamp",
    "json",
}


@dataclass(frozen=True)
class FieldSpec:
    """One column of a canonical dataset."""

    name: str
    dtype: str
    description: str
    nullable: bool = True
    source: str = "synthetic"
    business_meaning: str = ""
    domain: tuple[str, ...] | None = None  # closed value set, if any
    minimum: float | None = None
    maximum: float | None = None
    example: Any = None

    def __post_init__(self) -> None:
        if self.dtype not in CANONICAL_TYPES:
            raise ValueError(
                f"{self.name}: unknown canonical type {self.dtype!r}; "
                f"expected one of {sorted(CANONICAL_TYPES)}"
            )

    def to_contract(self) -> dict[str, Any]:
        """JSON-Schema-ish fragment used in ``schemas/*.json``."""
        json_types = {
            "string": "string",
            "int": "integer",
            "long": "integer",
            "float": "number",
            "decimal": "number",
            "bool": "boolean",
            "date": "string",
            "timestamp": "string",
            "json": "object",
        }
        fragment: dict[str, Any] = {
            "type": [json_types[self.dtype], "null"] if self.nullable else json_types[self.dtype],
            "description": self.description,
        }
        if self.dtype == "date":
            fragment["format"] = "date"
        if self.dtype == "timestamp":
            fragment["format"] = "date-time"
        if self.domain:
            fragment["enum"] = list(self.domain) + ([None] if self.nullable else [])
        if self.minimum is not None:
            fragment["minimum"] = self.minimum
        if self.maximum is not None:
            fragment["maximum"] = self.maximum
        if self.example is not None:
            fragment["examples"] = [self.example]
        return fragment


@dataclass(frozen=True)
class ForeignKey:
    """A referential-integrity edge between canonical datasets."""

    column: str
    references_dataset: str
    references_column: str
    #: Whether a violation quarantines the record or is merely reported.
    blocking: bool = True
    error_code: str = "REFERENTIAL_INTEGRITY"


@dataclass(frozen=True)
class DatasetSchema:
    """A canonical dataset: its fields, keys and integrity rules."""

    name: str
    description: str
    fields: tuple[FieldSpec, ...]
    business_key: tuple[str, ...]
    #: Columns whose content defines record identity for dedup/idempotency.
    hash_columns: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    partition_column: str | None = None
    schema_version: str = "1.0"
    owner: str = "data-engineering"
    classification: str = "internal"

    def field(self, name: str) -> FieldSpec:
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise KeyError(f"{self.name} has no field {name!r}")

    @property
    def column_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def required_columns(self) -> list[str]:
        return [f.name for f in self.fields if not f.nullable]

    @property
    def dtypes(self) -> dict[str, str]:
        return {f.name: f.dtype for f in self.fields}

    def effective_hash_columns(self) -> list[str]:
        return list(self.hash_columns or self.column_names)

    def to_contract(self) -> dict[str, Any]:
        """Export a JSON Schema document (draft 2020-12)."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://financial-intelligence-lakehouse/schemas/{self.name}.json",
            "title": self.name,
            "description": self.description,
            "type": "object",
            "x-schema-version": self.schema_version,
            "x-business-key": list(self.business_key),
            "x-owner": self.owner,
            "x-classification": self.classification,
            "properties": {f.name: f.to_contract() for f in self.fields},
            "required": self.required_columns,
            "additionalProperties": True,  # Bronze must tolerate new source fields
        }


# ---------------------------------------------------------------------------
# canonical datasets
# ---------------------------------------------------------------------------

AGENCY = DatasetSchema(
    name="agency",
    description="A federal agency that owns budget authority and awards contracts.",
    business_key=("agency_id",),
    hash_columns=("agency_id", "agency_name", "agency_abbreviation"),
    classification="public",
    fields=(
        FieldSpec(
            "agency_id",
            "string",
            "Surrogate agency identifier.",
            nullable=False,
            business_meaning="Primary key for the agency dimension.",
            example="AGY-0007",
        ),
        FieldSpec(
            "agency_name",
            "string",
            "Official agency name.",
            nullable=False,
            business_meaning="Human-readable agency label used in reporting.",
        ),
        FieldSpec("agency_abbreviation", "string", "Short agency code, e.g. DOT."),
        FieldSpec(
            "agency_type",
            "string",
            "Branch of government the agency belongs to.",
            domain=("EXECUTIVE", "LEGISLATIVE", "JUDICIAL", "INDEPENDENT"),
        ),
        FieldSpec(
            "toptier_code",
            "string",
            "USAspending top-tier agency code where known.",
            source="usaspending",
        ),
        FieldSpec(
            "established_year",
            "int",
            "Year the agency was established.",
            minimum=1780,
            maximum=2100,
        ),
        FieldSpec("headquarters_state", "string", "Two-letter US state code of the HQ."),
        FieldSpec(
            "annual_budget_usd",
            "decimal",
            "Published annual budget authority.",
            minimum=0,
            business_meaning="Denominator for budget-utilisation metrics.",
        ),
    ),
)

DEPARTMENT = DatasetSchema(
    name="department",
    description="An organisational unit inside an agency that incurs spending.",
    business_key=("department_id",),
    hash_columns=("department_id", "department_name", "agency_id"),
    foreign_keys=(ForeignKey("agency_id", "agency", "agency_id"),),
    fields=(
        FieldSpec(
            "department_id",
            "string",
            "Surrogate department identifier.",
            nullable=False,
            example="DEP-00042",
        ),
        FieldSpec("department_name", "string", "Department name.", nullable=False),
        FieldSpec(
            "agency_id",
            "string",
            "Owning agency.",
            nullable=False,
            business_meaning="Roll-up key from department spending to agency spending.",
        ),
        FieldSpec("cost_center", "string", "Finance cost-centre code."),
        FieldSpec(
            "budget_allocation_usd", "decimal", "Budget allocated to the department.", minimum=0
        ),
        FieldSpec(
            "manager_name",
            "string",
            "Fictional responsible manager.",
            business_meaning="Ownership contact for spending oversight.",
        ),
    ),
)

VENDOR = DatasetSchema(
    name="vendor",
    description="A supplier receiving payments; the 'recipient' in USAspending terms.",
    business_key=("vendor_id",),
    hash_columns=("vendor_id", "vendor_name", "tax_id", "country"),
    fields=(
        FieldSpec(
            "vendor_id",
            "string",
            "Surrogate vendor identifier.",
            nullable=False,
            example="VEN-000123",
        ),
        FieldSpec(
            "vendor_name",
            "string",
            "Legal vendor name.",
            nullable=False,
            business_meaning="Subject of vendor-concentration and risk analysis.",
        ),
        FieldSpec("vendor_legal_name", "string", "Registered legal entity name."),
        FieldSpec(
            "tax_id",
            "string",
            "Fictional tax identifier (never a real EIN/SSN).",
            business_meaning="Entity-resolution key for near-duplicate vendors.",
        ),
        FieldSpec(
            "vendor_type",
            "string",
            "Business classification.",
            domain=(
                "SMALL_BUSINESS",
                "LARGE_BUSINESS",
                "NON_PROFIT",
                "GOVERNMENT",
                "FOREIGN_ENTITY",
                "SOLE_PROPRIETOR",
            ),
        ),
        FieldSpec("industry_code", "string", "NAICS-style industry code."),
        FieldSpec("contact_email", "string", "Fictional contact address."),
        FieldSpec("phone", "string", "Fictional contact phone number."),
        FieldSpec("address_line1", "string", "Street address."),
        FieldSpec("city", "string", "City."),
        FieldSpec("state", "string", "Two-letter state code where applicable."),
        FieldSpec("postal_code", "string", "Postal code."),
        FieldSpec("country", "string", "ISO-3166 alpha-2 country code.", example="US"),
        FieldSpec(
            "registration_date",
            "date",
            "Date the vendor was registered.",
            business_meaning="Drives the 'new vendor' risk rule.",
        ),
        FieldSpec("is_active", "bool", "Whether the vendor is currently active."),
        FieldSpec(
            "risk_flag",
            "string",
            "Externally supplied watch-list flag.",
            domain=("NONE", "WATCH", "SANCTIONED"),
        ),
    ),
)

CONTRACT = DatasetSchema(
    name="contract",
    description="An award linking a vendor to an agency with a ceiling value.",
    business_key=("contract_id",),
    hash_columns=("contract_id", "vendor_id", "agency_id", "contract_value_usd"),
    foreign_keys=(
        ForeignKey("vendor_id", "vendor", "vendor_id"),
        ForeignKey("agency_id", "agency", "agency_id"),
    ),
    fields=(
        FieldSpec(
            "contract_id",
            "string",
            "Surrogate contract/award identifier.",
            nullable=False,
            example="CON-0001234",
        ),
        FieldSpec(
            "award_id", "string", "External award identifier (PIID-style).", source="usaspending"
        ),
        FieldSpec("vendor_id", "string", "Awarded vendor.", nullable=False),
        FieldSpec("agency_id", "string", "Awarding agency.", nullable=False),
        FieldSpec(
            "contract_type",
            "string",
            "Award instrument type.",
            domain=(
                "FIXED_PRICE",
                "COST_PLUS",
                "TIME_AND_MATERIALS",
                "IDIQ",
                "GRANT",
                "PURCHASE_ORDER",
            ),
        ),
        FieldSpec(
            "contract_value_usd",
            "decimal",
            "Total obligated ceiling value.",
            minimum=0,
            business_meaning="Compared against cumulative payments to detect overruns.",
        ),
        FieldSpec("start_date", "date", "Period-of-performance start."),
        FieldSpec("end_date", "date", "Period-of-performance end."),
        FieldSpec(
            "status",
            "string",
            "Lifecycle status.",
            domain=("ACTIVE", "COMPLETED", "TERMINATED", "PENDING"),
        ),
        FieldSpec("description", "string", "Free-text description of the award."),
        FieldSpec(
            "competition_type",
            "string",
            "How the award was competed.",
            domain=("FULL_AND_OPEN", "LIMITED", "SOLE_SOURCE", "SET_ASIDE"),
            business_meaning="Sole-source concentration is an oversight red flag.",
        ),
    ),
)

ACCOUNT = DatasetSchema(
    name="account",
    description="A departmental disbursement account that transactions are made from.",
    business_key=("account_id",),
    hash_columns=("account_id", "department_id", "account_type"),
    foreign_keys=(ForeignKey("department_id", "department", "department_id"),),
    fields=(
        FieldSpec(
            "account_id",
            "string",
            "Surrogate account identifier.",
            nullable=False,
            example="ACC-0004821",
        ),
        FieldSpec("account_number", "string", "Masked account number (fictional)."),
        FieldSpec("department_id", "string", "Owning department.", nullable=False),
        FieldSpec(
            "account_type",
            "string",
            "Kind of disbursement account.",
            domain=("OPERATING", "PAYROLL", "GRANT", "CAPITAL", "TRAVEL", "PROCUREMENT"),
        ),
        FieldSpec("currency", "string", "Account base currency.", example="USD"),
        FieldSpec("opened_date", "date", "Date the account was opened."),
        FieldSpec(
            "status", "string", "Account status.", domain=("ACTIVE", "DORMANT", "CLOSED", "FROZEN")
        ),
        FieldSpec("credit_limit_usd", "decimal", "Spending limit for the account.", minimum=0),
        FieldSpec("owner_name", "string", "Fictional account owner."),
        FieldSpec("region", "string", "Operating region of the account."),
    ),
)

BUDGET = DatasetSchema(
    name="budget",
    description="Fiscal-year budget lines by agency and department.",
    business_key=("budget_id",),
    hash_columns=("budget_id", "agency_id", "department_id", "fiscal_year"),
    foreign_keys=(
        ForeignKey("agency_id", "agency", "agency_id"),
        ForeignKey("department_id", "department", "department_id", blocking=False),
    ),
    fields=(
        FieldSpec("budget_id", "string", "Surrogate budget-line identifier.", nullable=False),
        FieldSpec(
            "fiscal_year",
            "int",
            "US federal fiscal year.",
            nullable=False,
            minimum=2000,
            maximum=2100,
        ),
        FieldSpec("agency_id", "string", "Agency the budget belongs to.", nullable=False),
        FieldSpec("department_id", "string", "Department the budget belongs to."),
        FieldSpec(
            "budget_category",
            "string",
            "Spending category.",
            domain=("PERSONNEL", "OPERATIONS", "CAPITAL", "GRANTS", "IT", "TRAVEL", "MAINTENANCE"),
        ),
        FieldSpec("allocated_amount_usd", "decimal", "Amount allocated.", minimum=0),
        FieldSpec(
            "spent_amount_usd",
            "decimal",
            "Amount spent to date.",
            minimum=0,
            business_meaning="Numerator of budget utilisation.",
        ),
        FieldSpec("quarter", "string", "Fiscal quarter.", domain=("Q1", "Q2", "Q3", "Q4")),
    ),
)

LOCATION = DatasetSchema(
    name="location",
    description="Geography where a transaction was initiated.",
    business_key=("location_id",),
    hash_columns=("location_id",),
    fields=(
        FieldSpec("location_id", "string", "Surrogate location identifier.", nullable=False),
        FieldSpec("city", "string", "City name."),
        FieldSpec("state", "string", "Two-letter state code."),
        FieldSpec("country", "string", "ISO-3166 alpha-2 country code."),
        FieldSpec("latitude", "float", "Latitude.", minimum=-90, maximum=90),
        FieldSpec("longitude", "float", "Longitude.", minimum=-180, maximum=180),
        FieldSpec("region", "string", "Reporting region."),
    ),
)

TRANSACTION = DatasetSchema(
    name="transaction",
    description="A single financial disbursement - the fact table of the platform.",
    business_key=("transaction_id",),
    # The hash deliberately excludes transaction_id: two records carrying the
    # same business content under different ids are still duplicates, which is
    # exactly the defect the generator injects.
    hash_columns=("account_id", "vendor_id", "amount", "currency", "transaction_timestamp"),
    partition_column="transaction_date",
    foreign_keys=(
        ForeignKey("account_id", "account", "account_id", error_code="UNKNOWN_ACCOUNT"),
        ForeignKey("vendor_id", "vendor", "vendor_id", error_code="UNKNOWN_VENDOR"),
        ForeignKey(
            "contract_id", "contract", "contract_id", blocking=False, error_code="UNKNOWN_CONTRACT"
        ),
    ),
    fields=(
        FieldSpec(
            "transaction_id",
            "string",
            "Unique transaction identifier.",
            nullable=False,
            source="source transaction systems",
            business_meaning="Primary key; uniqueness is a blocking quality check.",
            example="TXN-000000123456",
        ),
        FieldSpec(
            "account_id",
            "string",
            "Account the funds were disbursed from.",
            nullable=False,
            business_meaning="Join key to the account dimension.",
        ),
        FieldSpec(
            "vendor_id",
            "string",
            "Vendor that received the funds.",
            nullable=False,
            business_meaning="Join key to the vendor dimension.",
        ),
        FieldSpec(
            "contract_id",
            "string",
            "Contract the payment was made under, if any.",
            business_meaning="Nullable: petty-cash and travel spend has no contract.",
        ),
        FieldSpec("department_id", "string", "Department that incurred the spend."),
        FieldSpec("agency_id", "string", "Agency the spend rolls up to."),
        FieldSpec(
            "amount",
            "decimal",
            "Transaction amount in the stated currency.",
            nullable=False,
            minimum=0,
            business_meaning="Measure for every spending aggregation.",
        ),
        FieldSpec(
            "currency",
            "string",
            "ISO-4217 currency code.",
            nullable=False,
            example="USD",
            business_meaning="Normalised to upper case; unknown codes are quarantined.",
        ),
        FieldSpec(
            "amount_usd",
            "decimal",
            "Amount converted to USD at a fixed reference rate.",
            minimum=0,
            business_meaning="Currency-neutral measure for cross-agency totals.",
        ),
        FieldSpec(
            "transaction_timestamp",
            "timestamp",
            "When the transaction occurred.",
            nullable=False,
            business_meaning="Drives temporal analysis, velocity and ML train/test splits.",
        ),
        FieldSpec("transaction_date", "date", "Date component, used as the partition key."),
        FieldSpec(
            "posted_timestamp",
            "timestamp",
            "When the source system posted the record.",
            business_meaning="Difference from transaction_timestamp reveals late arrivals.",
        ),
        FieldSpec(
            "transaction_type",
            "string",
            "Payment instrument.",
            domain=("PAYMENT", "REFUND", "ADJUSTMENT", "TRANSFER", "PAYROLL", "GRANT_DISBURSEMENT"),
        ),
        FieldSpec(
            "payment_method",
            "string",
            "Method of payment.",
            domain=("ACH", "WIRE", "CHECK", "CARD", "EFT"),
        ),
        FieldSpec(
            "merchant_category",
            "string",
            "Merchant category code. Added in schema version 1.1 - see docs/schema_evolution.",
            business_meaning="Demonstrates additive schema evolution.",
        ),
        FieldSpec("description", "string", "Free-text payment description."),
        FieldSpec("location_id", "string", "Where the transaction was initiated."),
        FieldSpec(
            "channel",
            "string",
            "Origination channel.",
            domain=("ONLINE", "BRANCH", "BATCH", "MOBILE", "API"),
        ),
        FieldSpec(
            "status",
            "string",
            "Settlement status.",
            domain=("SETTLED", "PENDING", "REVERSED", "FAILED"),
        ),
        FieldSpec("fiscal_year", "int", "US federal fiscal year of the transaction."),
    ),
)

TRANSACTION_EVENT = DatasetSchema(
    name="transaction_event",
    description="Streaming envelope carrying a transaction plus event-time metadata.",
    business_key=("event_id",),
    hash_columns=("transaction_id", "account_id", "vendor_id", "amount", "event_timestamp"),
    fields=(
        FieldSpec("event_id", "string", "Unique event identifier.", nullable=False),
        FieldSpec(
            "event_type",
            "string",
            "Event kind.",
            domain=("TRANSACTION_CREATED", "TRANSACTION_UPDATED", "TRANSACTION_REVERSED"),
        ),
        FieldSpec(
            "event_timestamp",
            "timestamp",
            "Event time as stamped by the producer.",
            nullable=False,
            business_meaning="Event time, not processing time - watermarking uses this.",
        ),
        FieldSpec(
            "ingest_timestamp",
            "timestamp",
            "When the platform received the event.",
            business_meaning="Gap vs event_timestamp is the observed lateness.",
        ),
        FieldSpec("producer", "string", "Source system that emitted the event."),
        FieldSpec("schema_version", "string", "Producer's schema version."),
        FieldSpec("transaction_id", "string", "Transaction the event describes.", nullable=False),
        FieldSpec("account_id", "string", "Account involved."),
        FieldSpec("vendor_id", "string", "Vendor involved."),
        FieldSpec("amount", "decimal", "Transaction amount.", minimum=0),
        FieldSpec("currency", "string", "ISO-4217 currency code."),
        FieldSpec("location_id", "string", "Origination location."),
        FieldSpec("channel", "string", "Origination channel."),
    ),
)

#: Every canonical dataset, keyed by name.
SCHEMAS: dict[str, DatasetSchema] = {
    schema.name: schema
    for schema in (
        AGENCY,
        DEPARTMENT,
        VENDOR,
        CONTRACT,
        ACCOUNT,
        BUDGET,
        LOCATION,
        TRANSACTION,
        TRANSACTION_EVENT,
    )
}

#: English plurals that are not formed by appending "s". Table names appear in
#: SQL, dashboards and documentation, so "bronze_agencys" is not acceptable.
_IRREGULAR_PLURALS = {"agency": "agencies", "company": "companies"}


def pluralise(name: str) -> str:
    if name in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[name]
    if name.endswith("y") and not name.endswith(("ay", "ey", "oy", "uy")):
        return f"{name[:-1]}ies"
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return f"{name}es"
    return f"{name}s"


def table_name(layer: str, dataset: str) -> str:
    """Physical table name for a canonical dataset in a lake layer."""
    return f"{layer}_{pluralise(dataset)}"


#: Mapping from canonical dataset to its Bronze / Silver table names.
LAYER_TABLES: dict[str, dict[str, str]] = {
    name: {"bronze": table_name("bronze", name), "silver": table_name("silver", name)}
    for name in SCHEMAS
}


def get_schema(name: str) -> DatasetSchema:
    """Look up a canonical schema by dataset name."""
    try:
        return SCHEMAS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown canonical dataset {name!r}; known: {', '.join(sorted(SCHEMAS))}"
        ) from exc


def bronze_columns(name: str) -> dict[str, str]:
    """Declared columns of a Bronze table (business columns + audit columns)."""
    return {**get_schema(name).dtypes, **BRONZE_AUDIT_COLUMNS}


def silver_columns(name: str) -> dict[str, str]:
    return {**get_schema(name).dtypes, **SILVER_AUDIT_COLUMNS}


# ---------------------------------------------------------------------------
# schema evolution
# ---------------------------------------------------------------------------

#: Columns introduced after v1.0. Bronze accepts them unconditionally (it is
#: schema-on-read); Silver back-fills nulls for records written before the
#: column existed. See docs/architecture_decisions.md.
SCHEMA_EVOLUTION: dict[str, dict[str, list[str]]] = {
    "transaction": {
        "1.0": [],
        "1.1": ["merchant_category"],
    }
}


def columns_added_in(dataset: str, version: str) -> list[str]:
    return SCHEMA_EVOLUTION.get(dataset, {}).get(version, [])


def current_schema_version(dataset: str) -> str:
    versions = SCHEMA_EVOLUTION.get(dataset)
    if not versions:
        return get_schema(dataset).schema_version
    return max(versions, key=lambda v: tuple(int(p) for p in v.split(".")))
