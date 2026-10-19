"""Generate the complete multi-format source landscape.

    python -m src.data_generation.generate_all
    python -m src.data_generation.generate_all --scale large
    python -m src.data_generation.generate_all --transactions 500000 --accounts 10000

Output layout (each format carries a different part of the domain, so every
reader in :mod:`src.ingestion` has a real job):

    data/raw/csv/transactions_2025.csv     schema v1.0
    data/raw/csv/transactions_2026.csv     schema v1.1 (adds merchant_category)
    data/raw/json/accounts.json            enveloped export
    data/raw/json/locations.json
    data/raw/xml/contracts.xml             attributes + camelCase elements
    data/raw/excel/budgets.xlsx            multi-sheet, title row
    data/raw/api/{vendors,agencies,transactions}.json   served by the source API
    data/raw/streaming/events-*.json       NDJSON micro-batches
    data/raw/labels/fraud_labels.csv       held-out investigation outcomes
    data/sample/                           tiny committed samples

The **transactions split across two CSVs** is not cosmetic: the 2025 file has no
``merchant_category`` column and the 2026 file does. That is the schema-evolution
scenario from the specification, exercised end to end rather than described.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.logging import configure_logging, get_logger, log_stage
from src.data_generation.accounts import generate_accounts
from src.data_generation.agencies import generate_agencies, generate_departments, generate_locations
from src.data_generation.base import GeneratorContext, ReferenceData
from src.data_generation.budgets import generate_budgets
from src.data_generation.contracts import generate_contracts
from src.data_generation.events import generate_events
from src.data_generation.transactions import generate_transactions
from src.data_generation.writers import (
    write_csv,
    write_excel,
    write_json,
    write_streaming_batches,
    write_xml,
)

log = get_logger(__name__)

#: Columns present in schema version 1.0 of the transaction feed.
V1_0_COLUMNS = [
    "transaction_id",
    "account_id",
    "vendor_id",
    "contract_id",
    "department_id",
    "agency_id",
    "amount",
    "currency",
    "transaction_timestamp",
    "posted_timestamp",
    "transaction_type",
    "payment_method",
    "description",
    "location_id",
    "channel",
    "status",
]
#: Version 1.1 adds merchant_category. Everything else is unchanged - additive
#: evolution, which Bronze absorbs without a migration.
V1_1_COLUMNS = [*V1_0_COLUMNS, "merchant_category"]

SCHEMA_SPLIT_DATE = "2026-01-01"


@dataclass
class GenerationSummary:
    """Counts and provenance for one generation run."""

    seed: int
    scale: str
    counts: dict[str, int]
    files: dict[str, str]
    generated_at: str
    fraud_labels: int

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "scale": self.scale,
            "counts": self.counts,
            "files": self.files,
            "generated_at": self.generated_at,
            "fraud_labels": self.fraud_labels,
        }


def build_reference_data(
    ctx: GeneratorContext,
    *,
    agencies: int,
    vendors: int,
    contracts: int,
    accounts: int,
) -> ReferenceData:
    """Generate the dimensions in dependency order."""
    agency_frame = generate_agencies(ctx, agencies)
    department_frame = generate_departments(ctx, agency_frame)
    location_frame = generate_locations(ctx)
    vendor_frame = generate_vendors_frame(ctx, vendors)
    contract_frame = generate_contracts(ctx, contracts, vendor_frame, agency_frame)
    account_frame = generate_accounts(ctx, accounts, department_frame)

    # Budgets are intentionally left empty here and populated after the
    # transactions exist, so allocations can be anchored to real spend. See
    # src/data_generation/budgets.py for why.
    budget_frame = pd.DataFrame()

    return ReferenceData(
        agencies=agency_frame,
        departments=department_frame,
        locations=location_frame,
        vendors=vendor_frame,
        contracts=contract_frame,
        accounts=account_frame,
        budgets=budget_frame,
    )


def generate_vendors_frame(ctx: GeneratorContext, count: int) -> pd.DataFrame:
    from src.data_generation.vendors import generate_vendors

    return generate_vendors(ctx, count)


def write_sources(
    settings: Settings,
    reference: ReferenceData,
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    events: list[dict],
) -> dict[str, str]:
    """Publish every frame into its source format. Returns ``{label: path}``."""
    raw = settings.paths.raw
    files: dict[str, str] = {}

    # --- CSV: transactions, split by schema version ------------------------
    parsed_dates = pd.to_datetime(
        transactions["transaction_timestamp"], errors="coerce", utc=True, format="mixed"
    )
    is_v11 = parsed_dates >= pd.Timestamp(SCHEMA_SPLIT_DATE, tz="UTC")
    # Records with an unparseable timestamp land in the v1.0 file - a real
    # router cannot classify what it cannot read, and Silver must cope.
    is_v11 = is_v11.fillna(False)

    v1_0 = transactions.loc[~is_v11, V1_0_COLUMNS]
    v1_1 = transactions.loc[is_v11, V1_1_COLUMNS]
    files["transactions_csv_v1_0"] = str(write_csv(v1_0, raw / "csv" / "transactions_2025.csv"))
    files["transactions_csv_v1_1"] = str(write_csv(v1_1, raw / "csv" / "transactions_2026.csv"))

    # --- JSON: accounts and locations --------------------------------------
    files["accounts_json"] = str(
        write_json(
            reference.accounts,
            raw / "json" / "accounts.json",
            record_key="accounts",
            source_system="TREASURY_CORE",
        )
    )
    files["locations_json"] = str(
        write_json(
            reference.locations,
            raw / "json" / "locations.json",
            record_key="locations",
            source_system="REFERENCE_DATA_SERVICE",
        )
    )

    # --- XML: contracts -----------------------------------------------------
    files["contracts_xml"] = str(
        write_xml(
            reference.contracts,
            raw / "xml" / "contracts.xml",
            root_tag="contractAwards",
            record_tag="award",
            id_column="contract_id",
            source_system="PROCUREMENT_SYSTEM",
        )
    )

    # --- Excel: budgets + departments --------------------------------------
    files["budgets_xlsx"] = str(
        write_excel(
            {"budgets": reference.budgets, "departments": reference.departments},
            raw / "excel" / "budgets.xlsx",
            title_rows={"budgets"},
        )
    )

    # --- REST source system payloads ---------------------------------------
    # The local FastAPI source system serves these from disk, so the API
    # ingestion path consumes genuinely different records from the file paths.
    api_dir = raw / "api"
    files["vendors_api"] = str(
        write_json(
            reference.vendors,
            api_dir / "vendors.json",
            record_key="vendors",
            source_system="VENDOR_MASTER_API",
        )
    )
    files["agencies_api"] = str(
        write_json(
            reference.agencies,
            api_dir / "agencies.json",
            record_key="agencies",
            source_system="AGENCY_REGISTRY_API",
        )
    )
    # The API serves the most recent 10% of transactions - the slice a real
    # near-real-time endpoint would expose, overlapping the files on purpose so
    # deduplication has cross-source work to do.
    api_slice = transactions.sort_values("transaction_timestamp").tail(
        max(int(len(transactions) * 0.10), 1)
    )
    files["transactions_api"] = str(
        write_json(
            api_slice[V1_1_COLUMNS],
            api_dir / "transactions.json",
            record_key="transactions",
            source_system="PAYMENTS_GATEWAY_API",
        )
    )
    files["contracts_api"] = str(
        write_json(
            reference.contracts,
            api_dir / "contracts.json",
            record_key="contracts",
            source_system="PROCUREMENT_API",
        )
    )

    # --- streaming inbox ----------------------------------------------------
    batches = write_streaming_batches(events, settings.paths.streaming_inbox, batch_size=25)
    files["streaming_batches"] = str(len(batches))

    # --- held-out labels ----------------------------------------------------
    label_path = raw / "labels" / "fraud_labels.csv"
    write_csv(labels, label_path)
    files["fraud_labels"] = str(label_path)

    return files


def write_samples(settings: Settings, reference: ReferenceData, transactions: pd.DataFrame) -> None:
    """Write the tiny samples that are safe to commit.

    Committing samples (not the full lake) means a reviewer can see the shape of
    every format without cloning gigabytes, and the repository stays honest
    about what is generated versus stored.
    """
    sample_dir = settings.paths.sample
    sample_dir.mkdir(parents=True, exist_ok=True)

    write_csv(transactions.head(100), sample_dir / "transactions_sample.csv")
    write_json(
        reference.accounts.head(25),
        sample_dir / "accounts_sample.json",
        record_key="accounts",
        source_system="TREASURY_CORE",
    )
    write_xml(
        reference.contracts.head(25),
        sample_dir / "contracts_sample.xml",
        root_tag="contractAwards",
        record_tag="award",
        id_column="contract_id",
        source_system="PROCUREMENT_SYSTEM",
    )
    write_excel(
        {"budgets": reference.budgets.head(50), "departments": reference.departments.head(25)},
        sample_dir / "budgets_sample.xlsx",
        title_rows={"budgets"},
    )
    write_csv(reference.vendors.head(25), sample_dir / "vendors_sample.csv")


def generate_all(
    settings: Settings | None = None,
    *,
    scale: str = "default",
    seed: int | None = None,
    overrides: dict[str, int] | None = None,
    write_sample_files: bool = True,
) -> GenerationSummary:
    """Generate every source dataset. Returns a reproducibility summary."""
    settings = settings or get_settings()
    profile = settings.generation.profile(scale)
    overrides = overrides or {}

    counts = {
        "agencies": overrides.get("agencies") or profile.agencies,
        "vendors": overrides.get("vendors") or profile.vendors,
        "contracts": overrides.get("contracts") or profile.contracts,
        "accounts": overrides.get("accounts") or profile.accounts,
        "transactions": overrides.get("transactions") or profile.transactions,
        "events": overrides.get("events") or profile.events,
    }

    effective_seed = seed if seed is not None else settings.generation.seed
    ctx = GeneratorContext(settings=settings, seed=effective_seed)
    settings.paths.ensure()

    with log_stage(log, "generate_reference_data") as stage:
        reference = build_reference_data(
            ctx,
            agencies=counts["agencies"],
            vendors=counts["vendors"],
            contracts=counts["contracts"],
            accounts=counts["accounts"],
        )
        stage.update(reference.row_counts())

    with log_stage(log, "generate_transactions") as stage:
        result = generate_transactions(
            ctx,
            counts["transactions"],
            accounts=reference.accounts,
            vendors=reference.vendors,
            contracts=reference.contracts,
            departments=reference.departments,
            locations=reference.locations,
        )
        stage["records"] = len(result.transactions)
        stage["fraud_labels"] = result.fraud_count

    with log_stage(log, "generate_budgets") as stage:
        fiscal_years = tuple(
            range(settings.generation.start_date.year, settings.generation.end_date.year + 1)
        )
        reference.budgets = generate_budgets(
            ctx, reference.departments, fiscal_years, result.transactions
        )
        stage["records"] = len(reference.budgets)

    with log_stage(log, "generate_events") as stage:
        events = generate_events(
            ctx,
            counts["events"],
            accounts=reference.accounts,
            vendors=reference.vendors,
            locations=reference.locations,
        )
        stage["records"] = len(events)

    with log_stage(log, "write_sources") as stage:
        files = write_sources(settings, reference, result.transactions, result.labels, events)
        stage["files"] = len(files)

    if write_sample_files:
        with log_stage(log, "write_samples"):
            write_samples(settings, reference, result.transactions)

    actual_counts = {
        **reference.row_counts(),
        "transactions": len(result.transactions),
        "events": len(events),
    }

    summary = GenerationSummary(
        seed=effective_seed,
        scale=scale,
        counts=actual_counts,
        files=files,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        fraud_labels=result.fraud_count,
    )

    manifest_path = settings.paths.metadata / "generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    log.info("generation_complete", **actual_counts, seed=effective_seed, scale=scale)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data_generation.generate_all",
        description="Generate the multi-format synthetic source landscape.",
    )
    parser.add_argument(
        "--scale",
        default="default",
        choices=("small", "default", "large"),
        help="Named size profile from config/settings.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the generation seed (default: config value, 42).",
    )
    parser.add_argument("--transactions", type=int, default=None)
    parser.add_argument("--accounts", type=int, default=None)
    parser.add_argument("--vendors", type=int, default=None)
    parser.add_argument("--agencies", type=int, default=None)
    parser.add_argument("--contracts", type=int, default=None)
    parser.add_argument("--events", type=int, default=None)
    parser.add_argument(
        "--no-samples", action="store_true", help="Skip writing data/sample/ files."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in {"transactions", "accounts", "vendors", "agencies", "contracts", "events"}
        and value is not None
    }

    summary = generate_all(
        settings,
        scale=args.scale,
        seed=args.seed,
        overrides=overrides,
        write_sample_files=not args.no_samples,
    )

    print("\nGenerated source data")
    print("-" * 46)
    for name, value in summary.counts.items():
        print(f"  {name:<16} {value:>12,}")
    print(f"  {'fraud labels':<16} {summary.fraud_labels:>12,}")
    print(f"  {'seed':<16} {summary.seed:>12}")
    print(f"\nManifest: {Path(get_settings().paths.metadata) / 'generation_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
