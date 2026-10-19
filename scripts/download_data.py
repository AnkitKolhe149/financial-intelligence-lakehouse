"""Fetch the optional external datasets.

    python scripts/download_data.py --list
    python scripts/download_data.py --usaspending
    python scripts/download_data.py --kaggle-instructions

**Nothing here is required.** The platform generates its own multi-format source
data and runs completely offline; these downloads add real-world data on top.

Licensing is handled by not fighting it: USAspending data is US Government work
in the public domain and is fetched directly, while the Kaggle fraud datasets
carry licences that do not permit redistribution, so this script prints
instructions rather than pulling them through a back door. Committing someone
else's dataset into a portfolio repository is a licensing problem, not a
convenience.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger

log = get_logger(__name__)

KAGGLE_DATASETS = [
    {
        "name": "Credit Card Fraud Detection (synthetic, ~1.85M transactions)",
        "slug": "kaushalnandania/credit-card-fraud-detection",
        "url": "https://www.kaggle.com/datasets/kaushalnandania/credit-card-fraud-detection",
        "licence": "See the dataset page; described as synthetic.",
        "shape": "Transaction-level with a fraud label.",
    },
    {
        "name": "Credit Card Fraud Detection (anonymised, 284,807 rows, 492 frauds)",
        "slug": "mlg-ulb/creditcardfraud",
        "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "licence": "Database Contents Licence (DbCL) v1.0 - redistribution restricted.",
        "shape": "Time, Amount, V1..V28 (PCA components), Class.",
    },
]


def list_sources() -> None:
    print("\nExternal data sources")
    print("=" * 72)

    print("\n1. USAspending.gov  (used automatically by the pipeline)")
    print("   https://api.usaspending.gov/docs/endpoints/")
    print("   Licence : US Government work, public domain.")
    print("   Auth    : none required.")
    print("   Used for: federal awards, agencies, recipients, obligations.")
    print("   Status  : fetched live when USASPENDING_ENABLED=true, with a")
    print("             synthetic fallback of identical shape otherwise.")

    print("\n2. Kaggle fraud datasets  (optional, for supervised comparison)")
    for entry in KAGGLE_DATASETS:
        print(f"   - {entry['name']}")
        print(f"     {entry['url']}")
        print(f"     licence: {entry['licence']}")
        print(f"     shape  : {entry['shape']}")
    print("\n   Not downloaded automatically: the licences do not permit")
    print("   redistribution, and the platform ships a synthetic fraud")
    print("   population with ground-truth labels that serves the same purpose.")

    print("\n3. Synthetic data  (the platform's primary source)")
    print("   Generated locally by `make generate-data`.")
    print("   Provides CSV, JSON, XML, Excel, REST and streaming feeds, plus the")
    print("   deliberate data-quality defects the quality framework is tested on.")
    print()


def fetch_usaspending(days: int, pages: int) -> int:
    """Pull a bounded slice of federal awards and store it under data/external."""
    from src.ingestion.usaspending_client import USASpendingClient

    settings = get_settings()
    target = settings.paths.raw.parent / "external"
    target.mkdir(parents=True, exist_ok=True)

    end = date.today()
    start = end - timedelta(days=days)

    with USASpendingClient(settings) as client:
        print(f"Fetching agencies from {settings.usaspending.base_url} ...")
        agencies = client.list_agencies()
        if not agencies.empty:
            path = target / "usaspending_agencies.parquet"
            agencies.to_parquet(path, index=False)
            print(f"  {len(agencies):,} agencies -> {path.name}")

        print(f"Fetching awards {start} to {end} (max {pages} pages) ...")
        result = client.spending_by_award(start_date=start, end_date=end, max_pages=pages)
        if result.frame.empty:
            print("  no records returned; the synthetic fallback will be used by the pipeline")
            return 1
        path = target / "usaspending_awards.parquet"
        result.frame.to_parquet(path, index=False)
        print(f"  {len(result.frame):,} awards -> {path.name} (source: {result.source})")

        print("Fetching monthly obligation totals ...")
        over_time = client.spending_over_time(start_date=start, end_date=end)
        if not over_time.empty:
            path = target / "usaspending_spending_over_time.parquet"
            over_time.to_parquet(path, index=False)
            print(f"  {len(over_time):,} periods -> {path.name}")

        print(f"\nHTTP: {client.client.stats.as_dict()}")
    return 0


def kaggle_instructions() -> None:
    print("\nUsing a Kaggle fraud dataset")
    print("=" * 72)
    print("""
1. Create an API token at https://www.kaggle.com/settings -> API -> Create New Token.
   Put the downloaded kaggle.json at ~/.kaggle/kaggle.json (chmod 600), or set
   KAGGLE_USERNAME and KAGGLE_KEY in your .env.

2. Install the client and download into data/external/:

       pip install kaggle
       kaggle datasets download -d mlg-ulb/creditcardfraud -p data/external --unzip

3. The file is NOT committed to this repository - data/ is gitignored, and the
   licence does not permit redistribution.

4. Nothing in the platform depends on it. The generator already produces a
   fraud population with held-out ground-truth labels
   (data/raw/labels/fraud_labels.csv), which is what the model evaluation in
   src/ml/evaluation.py scores against. The Kaggle data is useful if you want
   to compare the unsupervised detector against a supervised baseline on a
   public benchmark.
""")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/download_data.py",
        description="Fetch or explain the optional external datasets.",
    )
    parser.add_argument("--list", action="store_true", help="List every source and its licence.")
    parser.add_argument("--usaspending", action="store_true", help="Fetch a bounded award slice.")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--kaggle-instructions", action="store_true")
    args = parser.parse_args(argv)

    configure_logging("INFO", "console")

    if args.usaspending:
        return fetch_usaspending(args.days, args.pages)
    if args.kaggle_instructions:
        kaggle_instructions()
        return 0

    list_sources()
    return 0


if __name__ == "__main__":
    sys.exit(main())
