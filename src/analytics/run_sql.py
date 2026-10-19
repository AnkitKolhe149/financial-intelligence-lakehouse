"""Execute the SQL analytics library.

    python -m src.analytics.run_sql                      # every file
    python -m src.analytics.run_sql --file sql/03_vendor_risk.sql
    python -m src.analytics.run_sql --query "SELECT COUNT(*) FROM silver_transactions"
    python -m src.analytics.run_sql --list               # available views

Doubles as a regression test: ``make sql`` runs every query, and a query that
stops parsing - because a column was renamed in the Gold layer, say - fails
here rather than in a dashboard six weeks later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.analytics.duckdb_client import AnalyticsEngine, QueryResult
from src.core.config import get_settings, project_root
from src.core.logging import configure_logging, get_logger

log = get_logger(__name__)


def print_result(result: QueryResult, max_rows: int = 10, quiet: bool = False) -> None:
    header = f"-- {result.name or 'query'} "
    print(f"\n{header}{'-' * max(0, 78 - len(header))}")
    if not result.succeeded:
        print(f"   FAILED: {result.error}")
        return
    print(f"   {result.rows:,} rows in {result.duration_seconds * 1000:.0f} ms")
    if quiet or result.frame.empty:
        if result.frame.empty:
            print("   (no rows)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 24):
        print(result.frame.head(max_rows).to_string(index=False))
    if result.rows > max_rows:
        print(f"   ... {result.rows - max_rows:,} more rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.analytics.run_sql",
        description="Run the SQL analytics library against the lake.",
    )
    parser.add_argument("--file", type=str, help="A single .sql file to execute.")
    parser.add_argument("--query", type=str, help="An ad-hoc SQL statement.")
    parser.add_argument("--dir", type=str, default=None, help="Directory of .sql files.")
    parser.add_argument("--rows", type=int, default=10, help="Rows to display per query.")
    parser.add_argument("--quiet", action="store_true", help="Timings only, no result rows.")
    parser.add_argument("--list", action="store_true", help="List registered views and exit.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    engine = AnalyticsEngine(settings)

    if args.list:
        print(f"\n{len(engine.registered)} views registered\n")
        for view, layer in sorted(engine.registered.items(), key=lambda kv: (kv[1], kv[0])):
            count = engine.df(f'SELECT COUNT(*) AS n FROM "{view}"')
            rows = int(count["n"].iloc[0]) if not count.empty else 0
            print(f"  [{layer:<10}] {view:<34} {rows:>10,} rows")
        return 0

    if args.query:
        print_result(engine.query(args.query, name="ad-hoc"), args.rows, args.quiet)
        return 0

    directory = Path(args.dir) if args.dir else project_root() / "sql"
    files = [Path(args.file)] if args.file else sorted(directory.glob("*.sql"))
    if not files:
        print(f"No SQL files found in {directory}")
        return 1

    total = 0
    failed = 0
    for path in files:
        print(f"\n{'=' * 79}\n{path.name}\n{'=' * 79}")
        for result in engine.run_sql_file(path):
            total += 1
            if not result.succeeded:
                failed += 1
            print_result(result, args.rows, args.quiet)

    print(f"\n{'=' * 79}")
    print(f"{total} queries executed, {total - failed} succeeded, {failed} failed")
    engine.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
