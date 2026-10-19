"""Deduplication.

Two distinct kinds of duplicate exist in this data, and conflating them is a
classic source of silent data loss:

**Same identifier, re-sent.** The source system delivered ``TXN-000123`` twice -
a replayed file, an at-least-once queue. One survivor, chosen deterministically.

**Different identifier, same content.** The source re-issued the same payment
under a new reference. The business key cannot see this; the content hash can,
because it is computed over account, vendor, amount, currency and timestamp
rather than over the identifier.

Both are collapsed here, and both are *recorded* - survivors keep a
``_duplicate_count`` so an analyst can see that a record was deduplicated, and
the losers are written to quarantine with ``DUPLICATE_RECORD`` rather than
vanishing.

Determinism is enforced by sorting on explicit columns and falling back to the
business key. Without a total ordering, two runs over identical input could keep
different survivors and every downstream number would move.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.contracts import ErrorCode
from src.core.logging import get_logger
from src.silver.spec import DedupStrategy

log = get_logger(__name__)


@dataclass
class DeduplicationOutcome:
    survivors: pd.DataFrame
    duplicates: pd.DataFrame
    removed_by_key: int = 0
    removed_by_hash: int = 0

    @property
    def removed_total(self) -> int:
        return self.removed_by_key + self.removed_by_hash


def deduplicate(
    frame: pd.DataFrame, strategy: DedupStrategy, dataset: str = ""
) -> DeduplicationOutcome:
    """Collapse duplicates according to ``strategy``."""
    if frame.empty:
        return DeduplicationOutcome(frame.copy(), _empty_duplicates(frame))

    working = frame.copy()
    ordered = _order_deterministically(working, strategy)

    duplicate_frames: list[pd.DataFrame] = []
    removed_by_key = 0
    removed_by_hash = 0

    # -- pass 1: business key ------------------------------------------------
    keys = [column for column in strategy.keys if column in ordered.columns]
    if keys:
        occurrence = ordered.groupby(keys, dropna=False).cumcount()
        losers = ordered.loc[occurrence > 0].copy()
        if not losers.empty:
            losers["error_code"] = ErrorCode.DUPLICATE_RECORD
            losers["error_message"] = (
                f"Duplicate {'+'.join(keys)}; a record with the same key was retained"
            )
            losers["error_field"] = "+".join(keys)
            duplicate_frames.append(losers)
            removed_by_key = len(losers)
        ordered = ordered.loc[occurrence == 0].copy()
        # Record how many source rows collapsed into each survivor.
        counts = frame.groupby(keys, dropna=False).size().rename("_duplicate_count")
        ordered = ordered.merge(counts, how="left", left_on=keys, right_index=True)
    else:
        ordered["_duplicate_count"] = 1

    # -- pass 2: content hash ------------------------------------------------
    if strategy.include_content_hash and "_record_hash" in ordered.columns:
        hash_values = ordered["_record_hash"]
        occurrence = ordered.groupby(hash_values.fillna("__null__"), dropna=False).cumcount()
        losers = ordered.loc[occurrence > 0].copy()
        if not losers.empty:
            losers["error_code"] = ErrorCode.DUPLICATE_RECORD
            losers["error_message"] = (
                "Duplicate record content under a different identifier " "(matched on _record_hash)"
            )
            losers["error_field"] = "_record_hash"
            duplicate_frames.append(losers)
            removed_by_hash = len(losers)
        ordered = ordered.loc[occurrence == 0].copy()

    duplicates = (
        pd.concat(duplicate_frames, ignore_index=True)
        if duplicate_frames
        else _empty_duplicates(frame)
    )

    if removed_by_key or removed_by_hash:
        log.info(
            "deduplicated",
            dataset=dataset,
            input_records=len(frame),
            survivors=len(ordered),
            removed_by_key=removed_by_key,
            removed_by_content_hash=removed_by_hash,
        )

    return DeduplicationOutcome(
        survivors=ordered.reset_index(drop=True),
        duplicates=duplicates,
        removed_by_key=removed_by_key,
        removed_by_hash=removed_by_hash,
    )


def _order_deterministically(frame: pd.DataFrame, strategy: DedupStrategy) -> pd.DataFrame:
    """Sort so the surviving row is always the same one.

    Sort columns come from the strategy (usually 'latest ingestion wins'), with
    the business key appended as the final tie-breaker so the order is total
    even when every timestamp is identical.
    """
    sort_columns = [c for c in strategy.order_by if c in frame.columns]
    tie_breakers = [c for c in strategy.keys if c in frame.columns and c not in sort_columns]
    columns = sort_columns + tie_breakers
    if not columns:
        return frame

    ascending = [strategy.ascending] * len(sort_columns) + [True] * len(tie_breakers)
    return frame.sort_values(by=columns, ascending=ascending, kind="stable", na_position="last")


def _empty_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    empty = frame.iloc[0:0].copy()
    for column in ("error_code", "error_message", "error_field"):
        empty[column] = pd.Series(dtype="string")
    return empty


def find_duplicate_business_keys(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Report duplicate keys and their multiplicity - used by the quality check."""
    if frame.empty or key not in frame.columns:
        return pd.DataFrame(columns=[key, "occurrences"])
    counts = frame[key].value_counts()
    offenders = counts[counts > 1]
    return pd.DataFrame({key: offenders.index, "occurrences": offenders.to_numpy()})


def near_duplicate_vendors(vendors: pd.DataFrame) -> pd.DataFrame:
    """Vendors sharing a tax id under different ids.

    Deliberately *reported* rather than merged. Auto-merging master data on a
    fuzzy signal is how a platform loses an audit trail; the platform's job is to
    surface the candidates for a steward to adjudicate.
    """
    if vendors.empty or "tax_id" not in vendors.columns:
        return pd.DataFrame(columns=["tax_id", "vendor_ids", "vendor_names", "occurrences"])

    populated = vendors.dropna(subset=["tax_id"])
    grouped = populated.groupby("tax_id").agg(
        vendor_ids=("vendor_id", lambda s: sorted(map(str, s))),
        vendor_names=("vendor_name", lambda s: sorted({str(v) for v in s})),
        occurrences=("vendor_id", "size"),
    )
    return (
        grouped[grouped["occurrences"] > 1]
        .reset_index()
        .sort_values("occurrences", ascending=False)
    )
