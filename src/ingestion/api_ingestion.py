"""REST ingestion from the local source system.

Demonstrates the full ``REST API -> Python ingestion -> Bronze`` path, including
the parts that are easy to skip and expensive to omit:

* **Cursor-following pagination** driven by the response envelope, with a hard
  page ceiling so a runaway ``has_next`` cannot loop forever.
* **Record-level contract validation** at the boundary. API records arrive one
  at a time and are judged individually - invalid ones are captured as
  rejections and carried forward to quarantine rather than dropped.
* **Partial-failure tolerance.** If page 7 of 12 fails after its retries, the
  first six pages are still returned and the failure is recorded. A load that
  yields 60% of the data plus an honest error beats a load that yields nothing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.contracts import RejectedRecord, validate_records
from src.core.exceptions import IngestionError
from src.core.logging import get_logger
from src.ingestion.base import IngestionResult, SourceReader
from src.ingestion.http_client import ResilientHttpClient

log = get_logger(__name__)

MAX_PAGES = 10_000


class ApiSourceReader(SourceReader):
    """Read a paginated REST endpoint into a canonical Bronze frame."""

    source_format = "rest_api"

    def __init__(
        self,
        dataset: str,
        source_system: str,
        *,
        run_id: str,
        endpoint: str,
        settings: Settings | None = None,
        client: ResilientHttpClient | None = None,
        params: dict[str, Any] | None = None,
        page_size: int = 200,
        max_records: int | None = None,
        validate: bool = True,
        column_map: dict[str, str] | None = None,
    ) -> None:
        super().__init__(dataset, source_system, run_id=run_id, column_map=column_map)
        self.settings = settings or get_settings()
        self.endpoint = endpoint
        self.params = params or {}
        self.page_size = page_size
        self.max_records = max_records
        self.validate = validate
        self._owns_client = client is None
        self.client = client or ResilientHttpClient(
            self.settings.services.source_api_base_url,
            timeout=30,
            max_retries=4,
            rate_limit_per_sec=0,  # a local source system needs no throttling
        )
        #: Records rejected by their contract, handed to the quarantine writer.
        self.rejected: list[RejectedRecord] = []

    def read(self) -> list[IngestionResult]:
        records: list[dict[str, Any]] = []
        errors: list[str] = []
        pages_read = 0
        page = 1

        while page <= MAX_PAGES:
            params = {**self.params, "page": page, "page_size": self.page_size}
            try:
                payload = self.client.get(self.endpoint, params=params)
            except IngestionError as exc:
                # Partial success: keep what we have, report what we lost.
                message = f"page {page} failed: {exc.message}"
                errors.append(message)
                log.error("api_page_failed", endpoint=self.endpoint, page=page, error=exc.message)
                break

            batch = _results_from(payload)
            records.extend(batch)
            pages_read += 1

            pagination = payload.get("pagination", {}) if isinstance(payload, dict) else {}
            has_next = bool(pagination.get("has_next"))
            if self.max_records is not None and len(records) >= self.max_records:
                records = records[: self.max_records]
                break
            if not has_next or not batch:
                break
            page += 1

        frame, rejected = self._to_frame(records)
        self.rejected = rejected
        if rejected:
            errors.append(f"{len(rejected)} records failed contract validation")

        metrics = {
            "pages": pages_read,
            "rejected_records": len(rejected),
            **self.client.stats.as_dict(),
        }
        result = self.finalise(
            frame,
            source_file=f"{self.settings.services.source_api_base_url}{self.endpoint}",
            errors=errors,
            metrics=metrics,
        )
        if self._owns_client:
            self.client.close()
        return [result]

    def _to_frame(self, records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[RejectedRecord]]:
        """Validate (when a contract exists) and build the frame."""
        if not records:
            return pd.DataFrame(), []
        if not self.validate:
            return pd.DataFrame(records), []

        try:
            accepted, rejected = validate_records(self.dataset, records)
        except KeyError:
            # No contract model for this dataset - Bronze takes it verbatim.
            return pd.DataFrame(records), []

        # Bronze stores what the source sent. The validated payload is used for
        # the accept/reject decision, but the *raw* record is what is persisted,
        # preserving source fidelity exactly as the file readers do.
        accepted_ids = {
            record.get(f"{self.dataset}_id") or record.get("transaction_id") for record in accepted
        }
        key_column = (
            f"{self.dataset}_id" if f"{self.dataset}_id" in records[0] else "transaction_id"
        )
        kept = [r for r in records if r.get(key_column) in accepted_ids] if accepted_ids else []
        return pd.DataFrame(kept or accepted), rejected


def _results_from(payload: Any) -> list[dict[str, Any]]:
    """Locate the record array in a response envelope."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("results", "data", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return next((v for v in payload.values() if isinstance(v, list)), [])


def source_api_available(settings: Settings | None = None, timeout: float = 2.0) -> bool:
    """Cheap health probe used to decide whether to attempt REST ingestion.

    The demo must not fail because a developer did not start the source API, so
    callers use this to skip the REST path and log the reason.
    """
    import requests

    settings = settings or get_settings()
    try:
        response = requests.get(
            f"{settings.services.source_api_base_url.rstrip('/')}/health", timeout=timeout
        )
        return response.status_code == 200
    except requests.RequestException:
        return False
