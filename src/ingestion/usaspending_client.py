"""USAspending.gov API client - the real public data source.

USAspending publishes US federal spending under an open licence and requires no
authentication, which makes it the right external source for a financial
oversight platform: real agencies, real recipients, real obligation amounts.

    Base:  https://api.usaspending.gov
    Docs:  https://api.usaspending.gov/docs/endpoints/

Endpoints used
--------------
``GET  /api/v2/references/toptier_agencies/``
    The agency reference list.
``POST /api/v2/search/spending_by_award/``
    Award-level records for a time period and award type - the closest thing
    the API offers to a transaction feed.
``POST /api/v2/search/spending_over_time/``
    Monthly obligation totals, used as an independent control total.
``POST /api/v2/search/spending_by_category/recipient/``
    Recipient (vendor) rankings.

Design notes
------------
* Deliberately **bounded**: the client pulls a configurable slice (a date range,
  a page cap), never "all federal spending". An ingestion job that tries to
  download a national dataset in a demo is a bug, not a feature.
* **Fails soft.** Every method returns an empty frame and logs, rather than
  raising, when the network is unavailable. :func:`fetch_or_fallback` then
  substitutes a synthetic federal-spending source so the pipeline still runs -
  the MVP fallback the specification calls for.
* **Field mapping is explicit.** The API returns display-cased keys such as
  ``"Award Amount"``; mapping them onto canonical snake_case names is done in
  one visible dictionary rather than by a clever transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import Settings, get_settings
from src.core.exceptions import IngestionError
from src.core.logging import get_logger
from src.ingestion.http_client import ResilientHttpClient

log = get_logger(__name__)

#: Contract award type codes (A-D are the definitive contract types).
CONTRACT_AWARD_TYPES = ["A", "B", "C", "D"]
#: Grant award types, used when pulling assistance spending.
GRANT_AWARD_TYPES = ["02", "03", "04", "05"]

AWARD_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Award Amount",
    "Total Outlays",
    "Description",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Start Date",
    "End Date",
    "Award Type",
    "Contract Award Type",
    "recipient_id",
    "Place of Performance State Code",
    "Place of Performance City Code",
]

#: API display names -> canonical column names.
AWARD_FIELD_MAP = {
    "Award ID": "award_id",
    "Recipient Name": "recipient_name",
    "Award Amount": "award_amount_usd",
    "Total Outlays": "total_outlays_usd",
    "Description": "description",
    "Awarding Agency": "awarding_agency_name",
    "Awarding Sub Agency": "awarding_sub_agency_name",
    "Start Date": "period_of_performance_start",
    "End Date": "period_of_performance_end",
    "Award Type": "award_type",
    "Contract Award Type": "contract_award_type",
    "recipient_id": "recipient_id",
    "Place of Performance State Code": "pop_state_code",
    "internal_id": "internal_award_id",
    "generated_internal_id": "generated_internal_id",
}


@dataclass
class USASpendingFetchResult:
    """Outcome of an external fetch, including which source actually supplied it."""

    frame: pd.DataFrame
    source: str  # "usaspending_api" | "synthetic_fallback"
    endpoint: str
    pages: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def is_live(self) -> bool:
        return self.source == "usaspending_api"


class USASpendingClient:
    """Bounded, retrying client for the USAspending public API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: ResilientHttpClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        config = self.settings.usaspending
        self.enabled = config.enabled
        self.page_size = config.page_size
        self._owns_client = client is None
        self.client = client or ResilientHttpClient(
            config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            rate_limit_per_sec=config.rate_limit_per_sec,
        )

    # -- reference data -----------------------------------------------------

    def list_agencies(self) -> pd.DataFrame:
        """``GET /api/v2/references/toptier_agencies/`` - the agency reference list."""
        if not self.enabled:
            return pd.DataFrame()
        try:
            payload = self.client.get("/api/v2/references/toptier_agencies/")
        except IngestionError as exc:
            log.warning("usaspending_unavailable", endpoint="toptier_agencies", error=exc.message)
            return pd.DataFrame()

        results = payload.get("results", [])
        if not results:
            return pd.DataFrame()
        frame = pd.DataFrame(results)
        renames = {
            "agency_name": "agency_name",
            "abbreviation": "agency_abbreviation",
            "toptier_code": "toptier_code",
            "agency_id": "usaspending_agency_id",
            "budget_authority_amount": "annual_budget_usd",
        }
        frame = frame.rename(columns={k: v for k, v in renames.items() if k in frame.columns})
        if "toptier_code" in frame.columns:
            frame["agency_id"] = "USA-" + frame["toptier_code"].astype(str)
        frame["_external_source"] = "usaspending"
        return frame

    # -- award search -------------------------------------------------------

    def spending_by_award(
        self,
        *,
        start_date: date,
        end_date: date,
        award_type_codes: list[str] | None = None,
        agencies: list[dict[str, str]] | None = None,
        max_pages: int = 5,
        page_size: int | None = None,
    ) -> USASpendingFetchResult:
        """``POST /api/v2/search/spending_by_award/`` with cursor pagination.

        ``max_pages`` is the guard rail that keeps a demo ingestion bounded.
        """
        endpoint = "/api/v2/search/spending_by_award/"
        if not self.enabled:
            return USASpendingFetchResult(pd.DataFrame(), "disabled", endpoint)

        limit = page_size or self.page_size
        filters: dict[str, Any] = {
            "time_period": [
                {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
            ],
            "award_type_codes": award_type_codes or CONTRACT_AWARD_TYPES,
        }
        if agencies:
            filters["agencies"] = agencies

        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        pages = 0

        for page in range(1, max_pages + 1):
            body = {
                "filters": filters,
                "fields": AWARD_FIELDS,
                "page": page,
                "limit": limit,
                "sort": "Award Amount",
                "order": "desc",
                "subawards": False,
            }
            try:
                payload = self.client.post(endpoint, body)
            except IngestionError as exc:
                errors.append(f"page {page}: {exc.message}")
                log.warning("usaspending_page_failed", page=page, error=exc.message)
                break

            results = payload.get("results", [])
            rows.extend(results)
            pages += 1
            metadata = payload.get("page_metadata", {})
            if not metadata.get("hasNext") or not results:
                break

        frame = self._normalise_awards(rows)
        source = "usaspending_api" if pages and not frame.empty else "unavailable"
        log.info(
            "usaspending_fetch",
            endpoint=endpoint,
            pages=pages,
            records=len(frame),
            errors=len(errors),
        )
        return USASpendingFetchResult(frame, source, endpoint, pages, errors)

    def spending_over_time(
        self, *, start_date: date, end_date: date, group: str = "month"
    ) -> pd.DataFrame:
        """``POST /api/v2/search/spending_over_time/`` - obligation totals per period.

        Useful as an independent control total: if our ingested award amounts
        diverge wildly from the API's own aggregate, we ingested wrongly.
        """
        if not self.enabled:
            return pd.DataFrame()
        body = {
            "group": group,
            "filters": {
                "time_period": [
                    {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
                ],
                "award_type_codes": CONTRACT_AWARD_TYPES,
            },
        }
        try:
            payload = self.client.post("/api/v2/search/spending_over_time/", body)
        except IngestionError as exc:
            log.warning("usaspending_unavailable", endpoint="spending_over_time", error=exc.message)
            return pd.DataFrame()

        results = payload.get("results", [])
        if not results:
            return pd.DataFrame()
        frame = pd.json_normalize(results)
        frame.columns = [str(c).replace("time_period.", "") for c in frame.columns]
        frame["_external_source"] = "usaspending"
        return frame

    def spending_by_recipient(
        self, *, start_date: date, end_date: date, limit: int = 100
    ) -> pd.DataFrame:
        """``POST /api/v2/search/spending_by_category/recipient/`` - top recipients."""
        if not self.enabled:
            return pd.DataFrame()
        body = {
            "filters": {
                "time_period": [
                    {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
                ],
                "award_type_codes": CONTRACT_AWARD_TYPES,
            },
            "category": "recipient",
            "limit": limit,
            "page": 1,
        }
        try:
            payload = self.client.post("/api/v2/search/spending_by_category/recipient/", body)
        except IngestionError as exc:
            log.warning(
                "usaspending_unavailable", endpoint="spending_by_recipient", error=exc.message
            )
            return pd.DataFrame()

        results = payload.get("results", [])
        if not results:
            return pd.DataFrame()
        frame = pd.DataFrame(results).rename(
            columns={
                "name": "recipient_name",
                "amount": "obligated_amount_usd",
                "code": "recipient_code",
            }
        )
        frame["_external_source"] = "usaspending"
        return frame

    def _normalise_awards(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame = frame.rename(
            columns={k: v for k, v in AWARD_FIELD_MAP.items() if k in frame.columns}
        )
        frame["_external_source"] = "usaspending"
        return frame

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> USASpendingClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# fallback source
# ---------------------------------------------------------------------------


def synthetic_federal_awards(
    *, start_date: date, end_date: date, count: int = 500, seed: int = 42
) -> pd.DataFrame:
    """Structurally-identical stand-in for USAspending awards.

    Used when the API is unreachable, disabled, or running in CI. It produces
    the same columns as :meth:`USASpendingClient.spending_by_award`, so every
    downstream stage is exercised identically - the pipeline cannot tell the
    difference, which is the point of a fallback.
    """
    rng = np.random.default_rng(seed)
    agencies = [
        ("Department of Defense", "DOD"),
        ("Department of Veterans Affairs", "VA"),
        ("Department of Health and Human Services", "HHS"),
        ("Department of Energy", "DOE"),
        ("General Services Administration", "GSA"),
        ("Department of Transportation", "DOT"),
    ]
    span = max((end_date - start_date).days, 1)

    rows = []
    for index in range(count):
        agency_name, agency_abbr = agencies[int(rng.integers(0, len(agencies)))]
        start = start_date + timedelta(days=int(rng.integers(0, span)))
        rows.append(
            {
                "award_id": f"SYN{agency_abbr}{index:06d}",
                "recipient_name": f"Synthetic Contractor {index % 250:03d} LLC",
                "award_amount_usd": float(round(rng.lognormal(13.0, 1.6), 2)),
                "total_outlays_usd": None,
                "description": "Synthetic federal award (USAspending unavailable)",
                "awarding_agency_name": agency_name,
                "awarding_sub_agency_name": f"{agency_abbr} Office of Acquisition",
                "period_of_performance_start": start.isoformat(),
                "period_of_performance_end": (
                    start + timedelta(days=int(rng.integers(90, 1095)))
                ).isoformat(),
                "award_type": "DEFINITIVE CONTRACT",
                "contract_award_type": "D",
                "recipient_id": f"SYNREC-{index % 250:05d}",
                "pop_state_code": str(rng.choice(["VA", "MD", "DC", "CA", "TX", "NY", "WA"])),
                "_external_source": "synthetic_fallback",
            }
        )
    return pd.DataFrame(rows)


def fetch_or_fallback(
    settings: Settings | None = None,
    *,
    days: int = 120,
    max_pages: int = 3,
    fallback_count: int = 500,
) -> USASpendingFetchResult:
    """Fetch federal awards, substituting the synthetic source on any failure.

    This is the function pipelines call. It never raises and always returns a
    populated frame, so an offline laptop produces the same pipeline shape as a
    connected one - with the source recorded honestly in the result and in
    lineage metadata.
    """
    settings = settings or get_settings()
    end = date.today()
    start = end - timedelta(days=days)

    if settings.usaspending.enabled:
        try:
            with USASpendingClient(settings) as client:
                result = client.spending_by_award(
                    start_date=start, end_date=end, max_pages=max_pages
                )
            if result.is_live and not result.frame.empty:
                return result
            log.warning(
                "usaspending_fallback_engaged",
                reason="no live records returned",
                errors=result.errors,
            )
        except Exception as exc:
            log.warning("usaspending_fallback_engaged", reason=str(exc)[:200])
    else:
        log.info("usaspending_disabled", detail="USASPENDING_ENABLED=false; using synthetic source")

    frame = synthetic_federal_awards(
        start_date=start, end_date=end, count=fallback_count, seed=settings.generation.seed
    )
    return USASpendingFetchResult(
        frame, "synthetic_fallback", "local", 0, ["external source unavailable"]
    )
