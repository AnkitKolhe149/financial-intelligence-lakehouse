"""Check suites, derived from the canonical schemas.

Most checks are *inferred* rather than hand-written: a NOT NULL column implies a
completeness check, a business key implies a uniqueness check, a declared
``domain`` implies an allowed-values check, a declared foreign key implies a
referential-integrity check. That inference means the quality suite cannot drift
away from the schema, and adding a column with a domain automatically gets it
validated.

Dataset-specific checks that cannot be inferred - freshness on the right
timestamp column, volume on the right date column, distribution drift on the
columns the ML model consumes - are declared explicitly below.
"""

from __future__ import annotations

from src.core.config import Settings, get_settings
from src.core.schema import SCHEMA_EVOLUTION, DatasetSchema, get_schema
from src.data_quality.base_check import CheckSeverity, DataQualityCheck
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
)
from src.silver.spec import get_silver_spec


def build_suite(dataset: str, settings: Settings | None = None) -> list[DataQualityCheck]:
    """Assemble the full check suite for a canonical dataset."""
    settings = settings or get_settings()
    schema = get_schema(dataset)
    checks: list[DataQualityCheck] = []

    checks.extend(_inferred_checks(schema, settings))
    checks.extend(_declared_checks(dataset, settings))
    return checks


def _inferred_checks(schema: DatasetSchema, settings: Settings) -> list[DataQualityCheck]:
    checks: list[DataQualityCheck] = []

    # Schema conformance - are the declared columns present at all?
    checks.append(SchemaCheck(expected_columns=schema.column_names))

    # Completeness of NOT NULL columns.
    required = schema.required_columns
    if required:
        checks.append(
            CompletenessCheck(
                columns=required,
                threshold=0.995,
                name=f"completeness_required_{schema.name}",
                severity=CheckSeverity.BLOCKING,
            )
        )

    # Completeness of important optional columns, as a warning only. These
    # nulls are legitimate but a sudden jump in them is worth seeing.
    #
    # Columns introduced by a later schema version are excluded: they are null
    # for every record written before the column existed, so including them
    # would guarantee a permanent warning that says nothing about data health.
    # Their population rate is tracked instead by the schema-evolution report.
    evolved_columns = {
        column
        for versions in [SCHEMA_EVOLUTION.get(schema.name, {})]
        for version, columns in versions.items()
        if version != "1.0"
        for column in columns
    }
    optional = [
        f.name
        for f in schema.fields
        if f.nullable
        and f.business_meaning
        and f.name not in required
        and f.name not in evolved_columns
    ]
    if optional:
        checks.append(
            CompletenessCheck(
                columns=optional,
                threshold=0.80,
                name=f"completeness_optional_{schema.name}",
                severity=CheckSeverity.WARNING,
            )
        )

    # Uniqueness of the business key.
    for key in schema.business_key:
        checks.append(UniquenessCheck(column=key))

    # Ranges and closed value sets declared on the fields themselves.
    for field_spec in schema.fields:
        if field_spec.minimum is not None or field_spec.maximum is not None:
            checks.append(
                RangeCheck(
                    column=field_spec.name,
                    minimum=field_spec.minimum,
                    maximum=field_spec.maximum,
                    threshold=0.999,
                    severity=(
                        CheckSeverity.BLOCKING
                        if field_spec.name in ("amount", "amount_usd")
                        else CheckSeverity.WARNING
                    ),
                )
            )
        if field_spec.domain:
            checks.append(
                AllowedValuesCheck(
                    column=field_spec.name,
                    allowed=list(field_spec.domain),
                    threshold=0.98,
                )
            )
        if field_spec.dtype == "timestamp":
            checks.append(TimestampSanityCheck(column=field_spec.name))

    # Referential integrity from the declared foreign keys.
    try:
        spec = get_silver_spec(schema.name)
    except KeyError:
        spec = None
    if spec is not None:
        for foreign_key in schema.foreign_keys:
            reference_table = next(
                (t for t, key in spec.reference_tables.items() if key == foreign_key.column),
                None,
            )
            if reference_table is None:
                continue
            checks.append(
                ReferentialIntegrityCheck(
                    column=foreign_key.column,
                    reference_table=reference_table,
                    threshold=1.0 if foreign_key.blocking else 0.95,
                    severity=(
                        CheckSeverity.BLOCKING if foreign_key.blocking else CheckSeverity.WARNING
                    ),
                )
            )

    return checks


def _declared_checks(dataset: str, settings: Settings) -> list[DataQualityCheck]:
    """Checks that need dataset-specific knowledge."""
    checks: list[DataQualityCheck] = []

    if dataset == "transaction":
        checks.extend(
            [
                # Currency is validated against the *configured* approved set,
                # which may be narrower than the schema's declared domain.
                AllowedValuesCheck(
                    column="currency",
                    allowed=settings.quality.allowed_currencies,
                    threshold=1.0,
                    name="validity_currency_approved",
                    severity=CheckSeverity.BLOCKING,
                ),
                RangeCheck(
                    column="amount",
                    minimum=0,
                    maximum=settings.quality.max_transaction_amount,
                    exclusive_min=True,
                    name="validity_amount_positive",
                    severity=CheckSeverity.BLOCKING,
                ),
                FreshnessCheck(
                    column="transaction_timestamp",
                    max_age_hours=settings.quality.freshness_max_age_hours,
                ),
                VolumeCheck(
                    date_column="transaction_date",
                    max_deviation_pct=settings.quality.volume.max_deviation_pct,
                ),
                DistributionCheck(
                    column="amount_usd",
                    psi_warn=settings.quality.distribution.psi_warn,
                    psi_fail=settings.quality.distribution.psi_fail,
                ),
            ]
        )

    elif dataset == "transaction_event":
        checks.append(
            FreshnessCheck(
                column="event_timestamp",
                max_age_hours=2,
                name="freshness_event_stream",
                severity=CheckSeverity.WARNING,
            )
        )

    elif dataset == "vendor":
        checks.append(
            CompletenessCheck(
                columns=["contact_email"],
                threshold=0.90,
                name="completeness_vendor_contact",
                severity=CheckSeverity.WARNING,
            )
        )

    elif dataset == "budget":
        checks.append(
            RangeCheck(
                column="spent_amount_usd",
                minimum=0,
                name="validity_budget_spend_non_negative",
                severity=CheckSeverity.WARNING,
            )
        )

    return checks


#: Datasets the batch pipeline scores on every run.
DEFAULT_SCORED_DATASETS: tuple[str, ...] = (
    "transaction",
    "account",
    "vendor",
    "contract",
    "agency",
    "department",
    "budget",
)
