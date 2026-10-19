"""Deterministic hashing - the backbone of idempotency and deduplication.

Two independent needs are served here:

``record_hash``
    A content fingerprint over a chosen set of business columns. Two records
    with the same business content produce the same hash regardless of source
    file, ingestion time or column order. Silver deduplication and Bronze
    idempotency both key off this, which is why the implementation must be
    stable across processes, machines and Python versions - hence SHA-256 over
    a canonical string rather than Python's randomised ``hash()``.

``deterministic_uuid``
    A UUIDv5 derived from a namespace + natural key, so surrogate identifiers
    (alert ids, run-scoped record ids) are reproducible on re-run instead of
    creating a new row every time the pipeline executes.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

# Fixed namespace for the platform. Must never change: it is baked into every
# previously-generated surrogate identifier.
NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_NULL_TOKEN = "\x00NULL\x00"
_FIELD_SEP = "\x1f"  # ASCII unit separator - cannot appear in normalised values


def normalise_value(value: Any) -> str:
    """Render a value into its canonical string form for hashing.

    The normalisation rules matter more than they look:

    * ``None`` becomes an explicit token so ``("a", None)`` and ``("a", "")``
      hash differently - a missing vendor is not an empty vendor.
    * Floats are formatted with ``repr`` and Decimals are normalised, so
      ``10.0`` and ``Decimal("10.00")`` agree.
    * Strings are stripped and case-folded, so ``" ACME "`` and ``"acme"``
      are the same business entity. This is deliberate: the injected defects
      include whitespace and casing noise, and duplicate detection must see
      through it.
    """
    if value is None:
        return _NULL_TOKEN
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.casefold() if stripped else _NULL_TOKEN
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        # normalize() collapses trailing zeros; the +0 re-expands exponent form
        # (Decimal("1E+2") -> Decimal("100")) so 100 and 1E+2 agree.
        return str(value.normalize() + Decimal(0))
    if isinstance(value, float):
        if value != value:  # NaN
            return _NULL_TOKEN
        rounded = round(value, 10)
        # An integral float renders as an integer so that 100 and 100.0 produce
        # the same fingerprint. This is not cosmetic: the CSV feed yields the
        # string "100", the JSON feed yields the number 100.0, and they are the
        # same payment. Without this, the same transaction arriving through two
        # sources would survive deduplication as two records.
        if rounded == int(rounded) and abs(rounded) < 1e15:
            return str(int(rounded))
        return repr(rounded)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        inner = _FIELD_SEP.join(f"{k}={normalise_value(v)}" for k, v in sorted(value.items()))
        return f"{{{inner}}}"
    if isinstance(value, list | tuple | set):
        items = sorted(normalise_value(v) for v in value)
        return f"[{_FIELD_SEP.join(items)}]"
    return str(value).strip().casefold()


def stable_hash(*parts: Any) -> str:
    """SHA-256 over the canonical rendering of ``parts``. Returns 64 hex chars."""
    canonical = _FIELD_SEP.join(normalise_value(part) for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_hash(record: Mapping[str, Any], columns: Sequence[str] | None = None) -> str:
    """Content fingerprint of a record over ``columns`` (or every column).

    Column *names* are included alongside values so that swapping two columns'
    contents changes the hash - important when sources reorder fields between
    releases.
    """
    keys: Iterable[str] = columns if columns is not None else sorted(record.keys())
    payload = _FIELD_SEP.join(f"{key}={normalise_value(record.get(key))}" for key in keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short_hash(*parts: Any, length: int = 12) -> str:
    """A shortened fingerprint for human-facing identifiers (batch ids, files)."""
    return stable_hash(*parts)[:length]


def deterministic_uuid(*parts: Any) -> str:
    """UUIDv5 from a natural key - the same inputs always yield the same id."""
    canonical = _FIELD_SEP.join(normalise_value(part) for part in parts)
    return str(uuid.uuid5(NAMESPACE, canonical))


def file_content_hash(path: Any, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed so large files do not load into RAM.

    Bronze stores this so a re-ingested, byte-identical file can be recognised
    and skipped instead of duplicating its rows.
    """
    from pathlib import Path

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
