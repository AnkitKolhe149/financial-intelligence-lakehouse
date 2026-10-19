"""Source-system REST API (the 'upstream' the platform ingests from).

    uvicorn api.source_system.main:app --port 8001

Endpoints
---------
    GET /health
    GET /transactions   ?page &page_size &start_date &end_date &account_id &vendor_id &min_amount
    GET /accounts       ?page &page_size &department_id &status
    GET /vendors        ?page &page_size &country &vendor_type &active_only
    GET /agencies       ?page &page_size &agency_type
    GET /contracts      ?page &page_size &agency_id &vendor_id &status

Behaviours that exist specifically to make ingestion non-trivial - and which
mirror what real source systems do:

* **Envelope pagination** with ``page``/``page_size``/``total_pages`` rather
  than a bare array, so the client must follow the cursor.
* **A hard page-size ceiling** (``MAX_PAGE_SIZE``), silently clamped, so a
  client that asks for everything at once does not get it.
* **Optional fault injection** (``?fail_rate=``) returning 503s, so retry and
  backoff logic can be exercised deterministically rather than hoped about.
* **Records served as-is**, defects included. A source system does not clean its
  data for you.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from src.core.config import get_settings

MAX_PAGE_SIZE = 500
DEFAULT_PAGE_SIZE = 100

app = FastAPI(
    title="Financial Source System API",
    description=(
        "Simulated upstream operational system. Serves transactions, accounts, "
        "vendors, agencies and contracts with pagination and filters."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load(dataset: str) -> list[dict[str, Any]]:
    """Load a dataset from the generated API payloads.

    Cached because the source system is read-only for the lifetime of a demo;
    re-reading a 50 MB JSON file on every page request would make pagination
    look artificially slow.
    """
    settings = get_settings()
    path: Path = settings.paths.raw / "api" / f"{dataset}.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Source dataset '{dataset}' has not been generated yet. "
                "Run: python -m src.data_generation.generate_all"
            ),
        )
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    records = document.get(dataset)
    if records is None:  # single-key documents may use a different label
        records = next((v for v in document.values() if isinstance(v, list)), [])
    return records


def _maybe_fail(fail_rate: float) -> None:
    """Deterministically-seeded fault injection for retry testing."""
    if fail_rate > 0 and random.random() < fail_rate:
        raise HTTPException(status_code=503, detail="Upstream temporarily unavailable")


def _paginate(
    records: list[dict[str, Any]], page: int, page_size: int, dataset: str
) -> JSONResponse:
    """Wrap a slice in the pagination envelope the client expects."""
    size = max(1, min(page_size, MAX_PAGE_SIZE))
    total = len(records)
    total_pages = max(1, (total + size - 1) // size)
    start = (page - 1) * size
    window = records[start : start + size]

    return JSONResponse(
        {
            "metadata": {
                "source_system": "SOURCE_SYSTEM_API",
                "dataset": dataset,
                "served_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "schema_version": "1.1" if dataset == "transactions" else "1.0",
            },
            "pagination": {
                "page": page,
                "page_size": size,
                "total_records": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
            },
            "results": window,
        }
    )


def _matches_date_range(
    record: dict[str, Any], start_date: str | None, end_date: str | None
) -> bool:
    """Filter on transaction date, tolerating the unparseable values in the feed.

    A record whose timestamp cannot be parsed is *included* rather than dropped:
    the source system has no basis to discard it, and the platform's quarantine
    is the right place for that decision.
    """
    if not start_date and not end_date:
        return True
    raw = record.get("transaction_timestamp")
    if not raw:
        return True
    try:
        stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except ValueError:
        return True
    if start_date and stamp < datetime.fromisoformat(start_date).date():
        return False
    return not (end_date and stamp > datetime.fromisoformat(end_date).date())


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["operations"])
def health() -> dict[str, Any]:
    """Liveness probe. Reports which datasets are actually loadable."""
    settings = get_settings()
    api_dir = settings.paths.raw / "api"
    datasets = {
        name: (api_dir / f"{name}.json").exists()
        for name in ("transactions", "accounts", "vendors", "agencies", "contracts")
    }
    return {
        "status": "healthy" if any(datasets.values()) else "degraded",
        "service": "source-system-api",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "datasets": datasets,
    }


@app.get("/transactions", tags=["data"])
def get_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1),
    start_date: str | None = Query(None, description="Inclusive ISO date, e.g. 2026-01-01"),
    end_date: str | None = Query(None, description="Inclusive ISO date"),
    account_id: str | None = None,
    vendor_id: str | None = None,
    min_amount: float | None = Query(None, ge=0),
    fail_rate: float = Query(0.0, ge=0.0, le=1.0, description="Fault injection for retry tests"),
) -> JSONResponse:
    _maybe_fail(fail_rate)
    records = _load("transactions")

    if any((start_date, end_date, account_id, vendor_id, min_amount is not None)):
        filtered = []
        for record in records:
            if not _matches_date_range(record, start_date, end_date):
                continue
            if account_id and record.get("account_id") != account_id:
                continue
            if vendor_id and record.get("vendor_id") != vendor_id:
                continue
            if min_amount is not None:
                try:
                    if float(record.get("amount") or 0) < min_amount:
                        continue
                except (TypeError, ValueError):
                    continue
            filtered.append(record)
        records = filtered

    return _paginate(records, page, page_size, "transactions")


@app.get("/accounts", tags=["data"])
def get_accounts(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1),
    department_id: str | None = None,
    status: str | None = None,
) -> JSONResponse:
    records = _load("accounts")
    if department_id:
        records = [r for r in records if r.get("department_id") == department_id]
    if status:
        records = [r for r in records if str(r.get("status", "")).upper() == status.upper()]
    return _paginate(records, page, page_size, "accounts")


@app.get("/vendors", tags=["data"])
def get_vendors(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1),
    country: str | None = None,
    vendor_type: str | None = None,
    active_only: bool = False,
    fail_rate: float = Query(0.0, ge=0.0, le=1.0),
) -> JSONResponse:
    _maybe_fail(fail_rate)
    records = _load("vendors")
    if country:
        records = [r for r in records if str(r.get("country", "")).upper() == country.upper()]
    if vendor_type:
        records = [
            r for r in records if str(r.get("vendor_type", "")).upper() == vendor_type.upper()
        ]
    if active_only:
        records = [r for r in records if r.get("is_active") in (True, "true", "True", 1)]
    return _paginate(records, page, page_size, "vendors")


@app.get("/agencies", tags=["data"])
def get_agencies(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1),
    agency_type: str | None = None,
) -> JSONResponse:
    records = _load("agencies")
    if agency_type:
        records = [
            r for r in records if str(r.get("agency_type", "")).upper() == agency_type.upper()
        ]
    return _paginate(records, page, page_size, "agencies")


@app.get("/contracts", tags=["data"])
def get_contracts(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1),
    agency_id: str | None = None,
    vendor_id: str | None = None,
    status: str | None = None,
) -> JSONResponse:
    records = _load("contracts")
    if agency_id:
        records = [r for r in records if r.get("agency_id") == agency_id]
    if vendor_id:
        records = [r for r in records if r.get("vendor_id") == vendor_id]
    if status:
        records = [r for r in records if str(r.get("status", "")).upper() == status.upper()]
    return _paginate(records, page, page_size, "contracts")


def run() -> None:  # pragma: no cover - entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.source_system.main:app",
        host=settings.services.source_api_host,
        port=settings.services.source_api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
