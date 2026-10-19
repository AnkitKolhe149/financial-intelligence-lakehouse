"""The source catalogue: which reader handles which feed.

Declaring the source landscape as data rather than as branching code means
adding a feed is a new entry in a list, and the batch pipeline never grows an
``if source == "csv"`` chain. It also documents, in one screen, exactly what the
platform ingests and from where - which is the first question anyone asks about
a data platform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core.config import Settings
from src.core.logging import get_logger
from src.ingestion.api_ingestion import ApiSourceReader, source_api_available
from src.ingestion.base import IngestionResult, SourceReader
from src.ingestion.csv_ingestion import CsvSourceReader
from src.ingestion.excel_ingestion import ExcelSourceReader
from src.ingestion.json_ingestion import JsonSourceReader
from src.ingestion.xml_ingestion import XmlSourceReader

log = get_logger(__name__)


@dataclass
class SourceDefinition:
    """One feed: a canonical dataset arriving in a particular format."""

    name: str
    dataset: str
    source_system: str
    source_format: str
    #: Builds the reader. Deferred so that constructing the catalogue does not
    #: touch the filesystem or the network.
    factory: Callable[[Settings, str], SourceReader]
    #: Feeds that need an external service running (the REST source system).
    requires_service: bool = False
    description: str = ""
    tags: list[str] = field(default_factory=list)


def build_catalogue() -> list[SourceDefinition]:
    """Every feed the batch pipeline knows how to ingest."""
    return [
        SourceDefinition(
            name="transactions_csv",
            dataset="transaction",
            source_system="PAYMENTS_CORE",
            source_format="csv",
            description=(
                "Daily transaction extracts. Two files with different schema "
                "versions (v1.0 without merchant_category, v1.1 with it)."
            ),
            tags=["fact", "schema-evolution"],
            factory=lambda settings, run_id: CsvSourceReader(
                "transaction",
                "PAYMENTS_CORE",
                run_id=run_id,
                directory=settings.paths.raw / "csv",
                pattern="transactions_*.csv",
            ),
        ),
        SourceDefinition(
            name="accounts_json",
            dataset="account",
            source_system="TREASURY_CORE",
            source_format="json",
            description="Account master export with a metadata envelope.",
            tags=["dimension"],
            factory=lambda settings, run_id: JsonSourceReader(
                "account",
                "TREASURY_CORE",
                run_id=run_id,
                files=[settings.paths.raw / "json" / "accounts.json"],
                record_key="accounts",
            ),
        ),
        SourceDefinition(
            name="locations_json",
            dataset="location",
            source_system="REFERENCE_DATA_SERVICE",
            source_format="json",
            description="Geography reference data.",
            tags=["dimension", "reference"],
            factory=lambda settings, run_id: JsonSourceReader(
                "location",
                "REFERENCE_DATA_SERVICE",
                run_id=run_id,
                files=[settings.paths.raw / "json" / "locations.json"],
                record_key="locations",
            ),
        ),
        SourceDefinition(
            name="contracts_xml",
            dataset="contract",
            source_system="PROCUREMENT_SYSTEM",
            source_format="xml",
            description="Award records as XML, id held as an element attribute.",
            tags=["dimension"],
            factory=lambda settings, run_id: XmlSourceReader(
                "contract",
                "PROCUREMENT_SYSTEM",
                run_id=run_id,
                files=[settings.paths.raw / "xml" / "contracts.xml"],
                record_tag="award",
                id_column="contract_id",
            ),
        ),
        SourceDefinition(
            name="budgets_excel",
            dataset="budget",
            source_system="FINANCE_PLANNING",
            source_format="xlsx",
            description="Budget workbook, 'budgets' sheet, title row above header.",
            tags=["dimension"],
            factory=lambda settings, run_id: ExcelSourceReader(
                "budget",
                "FINANCE_PLANNING",
                run_id=run_id,
                files=[settings.paths.raw / "excel" / "budgets.xlsx"],
                sheet_name="budgets",
            ),
        ),
        SourceDefinition(
            name="departments_excel",
            dataset="department",
            source_system="FINANCE_PLANNING",
            source_format="xlsx",
            description="Second sheet of the same workbook.",
            tags=["dimension"],
            factory=lambda settings, run_id: ExcelSourceReader(
                "department",
                "FINANCE_PLANNING",
                run_id=run_id,
                files=[settings.paths.raw / "excel" / "budgets.xlsx"],
                sheet_name="departments",
            ),
        ),
        SourceDefinition(
            name="vendors_api",
            dataset="vendor",
            source_system="VENDOR_MASTER_API",
            source_format="rest_api",
            requires_service=True,
            description="Vendor master pulled from the source system over REST.",
            tags=["dimension", "api"],
            factory=lambda settings, run_id: ApiSourceReader(
                "vendor",
                "VENDOR_MASTER_API",
                run_id=run_id,
                endpoint="/vendors",
                settings=settings,
                page_size=250,
            ),
        ),
        SourceDefinition(
            name="agencies_api",
            dataset="agency",
            source_system="AGENCY_REGISTRY_API",
            source_format="rest_api",
            requires_service=True,
            description="Agency registry pulled over REST.",
            tags=["dimension", "api"],
            factory=lambda settings, run_id: ApiSourceReader(
                "agency",
                "AGENCY_REGISTRY_API",
                run_id=run_id,
                endpoint="/agencies",
                settings=settings,
                page_size=250,
            ),
        ),
        SourceDefinition(
            name="transactions_api",
            dataset="transaction",
            source_system="PAYMENTS_GATEWAY_API",
            source_format="rest_api",
            requires_service=True,
            description=(
                "Near-real-time transaction slice. Overlaps the CSV feed on "
                "purpose so cross-source deduplication has real work to do."
            ),
            tags=["fact", "api"],
            factory=lambda settings, run_id: ApiSourceReader(
                "transaction",
                "PAYMENTS_GATEWAY_API",
                run_id=run_id,
                endpoint="/transactions",
                settings=settings,
                page_size=500,
            ),
        ),
    ]


#: Fallback file readers used when the REST source system is not running, so the
#: pipeline still ingests the same records - from the JSON payloads the API would
#: have served. The demo therefore never depends on a second process.
def build_api_fallbacks() -> list[SourceDefinition]:
    return [
        SourceDefinition(
            name="vendors_api_fallback",
            dataset="vendor",
            source_system="VENDOR_MASTER_API",
            source_format="json",
            description="Offline stand-in for GET /vendors.",
            tags=["dimension", "fallback"],
            factory=lambda settings, run_id: JsonSourceReader(
                "vendor",
                "VENDOR_MASTER_API",
                run_id=run_id,
                files=[settings.paths.raw / "api" / "vendors.json"],
                record_key="vendors",
            ),
        ),
        SourceDefinition(
            name="agencies_api_fallback",
            dataset="agency",
            source_system="AGENCY_REGISTRY_API",
            source_format="json",
            description="Offline stand-in for GET /agencies.",
            tags=["dimension", "fallback"],
            factory=lambda settings, run_id: JsonSourceReader(
                "agency",
                "AGENCY_REGISTRY_API",
                run_id=run_id,
                files=[settings.paths.raw / "api" / "agencies.json"],
                record_key="agencies",
            ),
        ),
        SourceDefinition(
            name="transactions_api_fallback",
            dataset="transaction",
            source_system="PAYMENTS_GATEWAY_API",
            source_format="json",
            description="Offline stand-in for GET /transactions.",
            tags=["fact", "fallback"],
            factory=lambda settings, run_id: JsonSourceReader(
                "transaction",
                "PAYMENTS_GATEWAY_API",
                run_id=run_id,
                files=[settings.paths.raw / "api" / "transactions.json"],
                record_key="transactions",
            ),
        ),
    ]


def select_sources(
    settings: Settings, *, requested: str = "all", service_available: bool | None = None
) -> list[SourceDefinition]:
    """Resolve ``--source`` into the concrete feeds to run.

    ``requested`` accepts ``all``, a format (``csv``, ``json``, ``xml``,
    ``excel``, ``api``) or a specific feed name. When the REST source system is
    not reachable, its feeds are swapped for their file-based fallbacks so the
    pipeline degrades instead of failing.
    """
    catalogue = build_catalogue()
    if service_available is None:
        service_available = any(d.requires_service for d in catalogue) and source_api_available(
            settings
        )

    if not service_available:
        fallbacks = {d.dataset: d for d in build_api_fallbacks()}
        swapped: list[SourceDefinition] = []
        for definition in catalogue:
            if definition.requires_service:
                replacement = fallbacks.get(definition.dataset)
                if replacement is not None:
                    swapped.append(replacement)
                    continue
            swapped.append(definition)
        catalogue = swapped
        log.warning(
            "source_api_unavailable",
            detail="using file-based fallbacks for REST feeds",
            base_url=settings.services.source_api_base_url,
        )

    requested = (requested or "all").lower()
    if requested == "all":
        return catalogue

    aliases = {"excel": "xlsx", "api": "rest_api"}
    normalised = aliases.get(requested, requested)

    selected = [
        d
        for d in catalogue
        if d.source_format == normalised
        or d.name == requested
        or d.dataset == requested
        or (normalised == "rest_api" and "fallback" in d.tags)
    ]
    if not selected:
        known = ", ".join(
            sorted({d.name for d in catalogue} | {d.source_format for d in catalogue})
        )
        raise ValueError(f"No sources match {requested!r}. Known: {known}")
    return selected


def read_source(
    definition: SourceDefinition, settings: Settings, run_id: str
) -> list[IngestionResult]:
    """Instantiate the reader and read, converting a hard failure into an empty
    result so one broken feed cannot stop the others."""
    try:
        reader = definition.factory(settings, run_id)
        return reader.read()
    except Exception as exc:
        log.error(
            "source_failed",
            source=definition.name,
            error=str(exc)[:300],
            error_type=type(exc).__name__,
        )
        return [
            IngestionResult(
                dataset=definition.dataset,
                source_system=definition.source_system,
                source_format=definition.source_format,
                source_file=definition.name,
                frame=pd.DataFrame(),
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        ]


def catalogue_summary() -> list[dict[str, Any]]:
    """Serialisable description of the source landscape, used by docs and the API."""
    return [
        {
            "name": d.name,
            "dataset": d.dataset,
            "source_system": d.source_system,
            "format": d.source_format,
            "requires_service": d.requires_service,
            "description": d.description,
            "tags": d.tags,
        }
        for d in build_catalogue()
    ]
