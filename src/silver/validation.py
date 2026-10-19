"""Record-level validation and cleaning for the Silver layer.

Everything is vectorised. A per-row ``apply`` over 500,000 transactions with a
Pydantic model per record would take minutes; the same rules expressed as column
operations take under a second, and correctness is identical because the rules
are the same rules. (The Pydantic contracts in :mod:`src.core.contracts` remain
the enforcement path for record-at-a-time sources like the REST feed and the
event stream, where a batch does not exist.)

The rejection model is **first failure wins**: each record is assigned at most
one error code, in a fixed priority order, so quarantine counts by error code
sum to the number of rejected records instead of double-counting a row that is
wrong in three ways.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import Settings
from src.core.contracts import USD_REFERENCE_RATES, ErrorCode
from src.core.logging import get_logger
from src.silver.spec import NormalisationRules, SilverSpec

log = get_logger(__name__)

#: Clock-skew tolerance before a timestamp counts as being in the future.
FUTURE_TOLERANCE = timedelta(minutes=5)


@dataclass
class ValidationOutcome:
    """Records that passed, records that failed, and why."""

    valid: pd.DataFrame
    rejected: pd.DataFrame
    counts_by_code: dict[str, int] = field(default_factory=dict)

    @property
    def valid_count(self) -> int:
        return len(self.valid)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------


def normalise_strings(frame: pd.DataFrame, rules: NormalisationRules) -> pd.DataFrame:
    """Apply trim / case rules and convert null-ish tokens to real nulls.

    The generator injects ``"  ACME  "``, ``"acme"`` and ``"N/A"``; without this
    step they would become three distinct vendors and one fake one.
    """
    cleaned = frame.copy()
    null_tokens = {token.casefold() for token in rules.null_tokens}

    text_columns = {
        column
        for column in cleaned.columns
        if not column.startswith("_")
        and (
            cleaned[column].dtype == object
            or str(cleaned[column].dtype).startswith(("str", "string"))
        )
    }

    for column in text_columns:
        series = cleaned[column].astype("string")
        # Collapse internal whitespace runs and strip the ends everywhere, not
        # only on the columns listed for trimming - trailing spaces are never
        # meaningful in this domain and they break every join.
        series = series.str.strip().str.replace(r"\s+", " ", regex=True)
        series = series.mask(series.str.casefold().isin(null_tokens) | (series == ""), pd.NA)
        cleaned[column] = series

    for column in rules.upper:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.upper()
    for column in rules.lower:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.lower()
    for column in rules.title:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.title()

    return cleaned


def cast_types(frame: pd.DataFrame, spec: SilverSpec) -> pd.DataFrame:
    """Cast columns to their declared canonical types.

    Failures become nulls rather than exceptions; the validation pass then
    decides whether that null is fatal. Casting and judging are separate steps
    on purpose - a cast that raises tells you a batch failed, not which record.
    """
    typed = frame.copy()
    dtypes = spec.schema.dtypes

    for column, dtype in dtypes.items():
        if column not in typed.columns:
            continue
        if dtype in {"int", "long"}:
            typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("Int64")
        elif dtype in {"float", "decimal"}:
            typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("Float64")
        elif dtype == "bool":
            typed[column] = _to_boolean(typed[column])
        elif dtype == "timestamp":
            typed[column] = _to_datetime(typed[column])
        elif dtype == "date":
            typed[column] = _to_datetime(typed[column]).dt.date
        else:
            typed[column] = typed[column].astype("string")

    for column in spec.timestamp_columns:
        if column in typed.columns and not pd.api.types.is_datetime64_any_dtype(typed[column]):
            typed[column] = _to_datetime(typed[column])

    return typed


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse mixed-format timestamps to UTC, coercing failures to NaT."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return parsed


def _to_boolean(series: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "t", "active"}
    falsy = {"false", "0", "no", "n", "f", "inactive"}
    text = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result = result.mask(text.isin(truthy), True)
    result = result.mask(text.isin(falsy), False)
    return result


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate_frame(
    frame: pd.DataFrame,
    spec: SilverSpec,
    settings: Settings,
    references: dict[str, pd.Series] | None = None,
    raw_frame: pd.DataFrame | None = None,
) -> ValidationOutcome:
    """Apply every record-level rule, partitioning the frame into valid/rejected.

    ``references`` maps a reference table name to the Series of valid key values,
    used for foreign-key enforcement. When a reference is absent (for example
    the first run, before dimensions exist) the corresponding check is skipped
    and logged rather than failing every record.

    ``raw_frame`` is the **pre-cast** frame. It matters more than it looks:
    casting turns an unparseable timestamp into ``NaT``, which is
    indistinguishable from a genuinely absent value once the cast has happened.
    Without the raw frame, ``"not-a-date"`` gets reported as
    ``NULL_REQUIRED_FIELD`` - technically true of the cast value, and actively
    misleading to whoever has to go and fix the source system. Comparing against
    the raw values lets the record be reported as ``INVALID_TIMESTAMP``, which
    is the defect that actually happened.
    """
    if frame.empty:
        return ValidationOutcome(frame.copy(), _empty_rejections(frame), {})

    working = frame.copy()
    source = raw_frame if raw_frame is not None else frame
    error_code = pd.Series(pd.NA, index=working.index, dtype="string")
    error_message = pd.Series(pd.NA, index=working.index, dtype="string")
    error_field = pd.Series(pd.NA, index=working.index, dtype="string")

    def reject(
        mask: pd.Series | np.ndarray, code: str, message: str, column: str | None = None
    ) -> None:
        """Assign an error to records that do not already carry one."""
        mask = pd.Series(mask, index=working.index).fillna(False).astype(bool)
        target = mask & error_code.isna()
        if not target.any():
            return
        error_code.loc[target] = code
        error_message.loc[target] = message
        if column:
            error_field.loc[target] = column

    def was_present(column: str) -> pd.Series:
        """True where the source supplied *something* for this column.

        Evaluated against the raw frame, before casting destroyed the evidence.
        """
        if column not in source.columns:
            return pd.Series(False, index=working.index)
        present = source[column].notna()
        if present.dtype != bool:
            present = present.astype(bool)
        # Empty strings are "absent", not "unparseable".
        if source[column].dtype == object or str(source[column].dtype).startswith(
            ("str", "string")
        ):
            present = present & source[column].astype("string").str.strip().ne("")
        return present.reindex(working.index, fill_value=False)

    # -- 1. unparseable values ----------------------------------------------
    # Deliberately first: a value that arrived and could not be parsed is a
    # different defect from a value that never arrived, and the error code an
    # analyst reads should say which one happened.
    now = pd.Timestamp(datetime.now(UTC))
    for column in spec.timestamp_columns:
        if column not in working.columns:
            continue
        reject(
            was_present(column) & working[column].isna(),
            ErrorCode.INVALID_TIMESTAMP,
            f"Field '{column}' could not be parsed as a timestamp",
            column,
        )

    for column in spec.positive_columns:
        if column not in working.columns:
            continue
        reject(
            was_present(column) & pd.to_numeric(working[column], errors="coerce").isna(),
            ErrorCode.INVALID_AMOUNT,
            f"Field '{column}' is not a number",
            column,
        )

    # -- 2. required fields --------------------------------------------------
    for column in spec.effective_required():
        if column not in working.columns:
            reject(
                pd.Series(True, index=working.index),
                ErrorCode.SCHEMA_MISMATCH,
                f"Required column '{column}' is absent from the source",
                column,
            )
            continue
        reject(
            working[column].isna(),
            ErrorCode.NULL_REQUIRED_FIELD,
            f"Required field '{column}' is null",
            column,
        )

    # -- 3. future timestamps ------------------------------------------------
    for column in spec.timestamp_columns:
        if column not in working.columns:
            continue
        reject(
            working[column].notna() & (working[column] > now + FUTURE_TOLERANCE),
            ErrorCode.FUTURE_TIMESTAMP,
            f"Field '{column}' is dated in the future",
            column,
        )

    # -- 4. amounts ----------------------------------------------------------
    for column in spec.positive_columns:
        if column not in working.columns:
            continue
        values = pd.to_numeric(working[column], errors="coerce")
        reject(
            values.notna() & (values <= 0),
            ErrorCode.INVALID_AMOUNT,
            f"Field '{column}' must be greater than zero",
            column,
        )
        reject(
            values.notna() & (values > settings.quality.max_transaction_amount),
            ErrorCode.AMOUNT_EXCEEDS_LIMIT,
            f"Field '{column}' exceeds the platform limit of "
            f"{settings.quality.max_transaction_amount:,.0f}",
            column,
        )

    # -- 5. currency ---------------------------------------------------------
    if "currency" in working.columns:
        allowed = set(settings.quality.allowed_currencies)
        currency = working["currency"].astype("string").str.strip().str.upper()
        working["currency"] = currency
        reject(
            currency.notna() & ~currency.isin(allowed),
            ErrorCode.INVALID_CURRENCY,
            "Currency is not in the approved set",
            "currency",
        )

    # -- 6. closed value sets ------------------------------------------------
    # Enum violations are reported, not rejected: an unrecognised payment method
    # does not make the money movement untrue. It surfaces in the validity score.
    enum_violations: dict[str, int] = {}
    for field_spec in spec.schema.fields:
        if not field_spec.domain or field_spec.name not in working.columns:
            continue
        column_values = working[field_spec.name].astype("string").str.upper()
        offending = column_values.notna() & ~column_values.isin(set(field_spec.domain))
        if offending.any():
            enum_violations[field_spec.name] = int(offending.sum())

    # -- 7. referential integrity -------------------------------------------
    references = references or {}
    for foreign_key in spec.foreign_keys:
        if foreign_key.column not in working.columns:
            continue
        reference_table = next(
            (table for table, key in spec.reference_tables.items() if key == foreign_key.column),
            None,
        )
        valid_keys = references.get(reference_table) if reference_table else None
        if valid_keys is None:
            log.debug(
                "referential_check_skipped",
                dataset=spec.dataset,
                column=foreign_key.column,
                reason="reference dataset unavailable",
            )
            continue

        known = set(valid_keys.dropna().astype(str))
        column_values = working[foreign_key.column].astype("string")
        orphan = column_values.notna() & ~column_values.isin(known)

        if foreign_key.blocking:
            reject(
                orphan,
                foreign_key.error_code,
                f"'{foreign_key.column}' does not exist in {foreign_key.references_dataset}",
                foreign_key.column,
            )
        elif orphan.any():
            # Non-blocking: null the broken reference and keep the record. A
            # payment with an unknown contract is still a real payment.
            log.info(
                "non_blocking_orphans_nulled",
                dataset=spec.dataset,
                column=foreign_key.column,
                records=int(orphan.sum()),
            )
            working.loc[orphan, foreign_key.column] = pd.NA

    # -- partition ----------------------------------------------------------
    rejected_mask = error_code.notna()
    valid = working.loc[~rejected_mask].copy()

    rejected = working.loc[rejected_mask].copy()
    if not rejected.empty:
        rejected["error_code"] = error_code.loc[rejected_mask]
        rejected["error_message"] = error_message.loc[rejected_mask]
        rejected["error_field"] = error_field.loc[rejected_mask]
    else:
        rejected = _empty_rejections(working)

    counts = error_code.loc[rejected_mask].value_counts().to_dict() if rejected_mask.any() else {}
    if enum_violations:
        log.info("enum_violations_reported", dataset=spec.dataset, **enum_violations)

    return ValidationOutcome(
        valid=valid, rejected=rejected, counts_by_code={str(k): int(v) for k, v in counts.items()}
    )


def _empty_rejections(frame: pd.DataFrame) -> pd.DataFrame:
    empty = frame.iloc[0:0].copy()
    for column in ("error_code", "error_message", "error_field"):
        empty[column] = pd.Series(dtype="string")
    return empty


# ---------------------------------------------------------------------------
# derived columns
# ---------------------------------------------------------------------------


def add_transaction_derivations(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the columns Gold and the ML layer depend on.

    ``amount_usd`` uses fixed reference rates (documented as a simplification in
    the data dictionary). A production platform would join a rates dimension
    with effective dating; the important part - that cross-currency totals are
    computed on a common basis rather than by summing mixed currencies - is the
    same either way.
    """
    if frame.empty:
        for column in ("amount_usd", "transaction_date", "fiscal_year", "_is_late_arriving"):
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="Float64" if column == "amount_usd" else "object")
        return frame

    enriched = frame.copy()
    rates = {code: float(rate) for code, rate in USD_REFERENCE_RATES.items()}
    currency = enriched.get("currency")
    amount = pd.to_numeric(enriched.get("amount"), errors="coerce")

    if currency is not None:
        rate_series = currency.astype("string").map(rates).astype("Float64")
        enriched["amount_usd"] = (amount * rate_series).round(2)
    else:
        enriched["amount_usd"] = amount.round(2)

    timestamps = enriched["transaction_timestamp"]
    enriched["transaction_date"] = timestamps.dt.date.astype("string")
    # Month is the *partition* key; date is kept as an ordinary column for
    # filtering and grouping. Partitioning by day looked natural and measured
    # badly: at demo volumes it produced ~570 directories holding a couple of
    # dozen rows each, and per-file overhead then dominated every scan - a
    # metadata-only row count came out slower than reading the whole table.
    # Monthly partitions keep files large enough to be worth opening while
    # still pruning the great majority of a multi-year table.
    enriched["transaction_month"] = timestamps.dt.strftime("%Y-%m")
    # US federal fiscal year starts on 1 October.
    enriched["fiscal_year"] = np.where(
        timestamps.dt.month >= 10, timestamps.dt.year + 1, timestamps.dt.year
    ).astype("float")
    enriched["fiscal_year"] = enriched["fiscal_year"].astype("Int64")

    if "posted_timestamp" in enriched.columns:
        lag = (enriched["posted_timestamp"] - timestamps).dt.total_seconds()
        # More than 24h between occurrence and posting is late-arriving data.
        enriched["_is_late_arriving"] = (lag > 86_400).fillna(False)
    else:
        enriched["_is_late_arriving"] = False

    return enriched


def decimalise(values: pd.Series) -> pd.Series:
    """Convert a float column to ``Decimal`` for exact monetary arithmetic.

    Used when writing monetary aggregates that must not drift; kept opt-in
    because Decimal columns are object-typed and slow to aggregate at scale.
    """
    return values.map(lambda v: Decimal(str(v)).quantize(Decimal("0.01")) if pd.notna(v) else None)


def summarise_validation(outcome: ValidationOutcome, dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "valid": outcome.valid_count,
        "rejected": outcome.rejected_count,
        **{f"error_{code.lower()}": count for code, count in outcome.counts_by_code.items()},
    }
