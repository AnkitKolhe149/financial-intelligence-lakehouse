"""DuckDB analytics engine over the Parquet lake.

Every lake dataset is exposed as a view via ``read_parquet(..., hive_partitioning
= true)``. Views rather than tables means no data is copied and no refresh step
can go stale: the next query sees whatever the last pipeline run wrote.

Hive partitioning is enabled explicitly so that ``WHERE transaction_date =
'2026-08-15'`` prunes to a single directory instead of scanning the table - the
partitioning the pipeline writes is only worth having if the query engine knows
about it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.core.config import Settings, get_settings
from src.core.io import dataset_exists
from src.core.logging import get_logger

log = get_logger(__name__)

#: Statements that are never legitimate in an analytics query here. The engine
#: is read-only over the lake; anything that mutates it must go through a
#: pipeline so that lineage and quality metadata are recorded.
_FORBIDDEN = re.compile(
    r"^\s*(insert|update|delete|drop|alter|truncate|copy|attach)\b", re.IGNORECASE
)


@dataclass
class QueryResult:
    """A query's rows plus how it performed."""

    frame: pd.DataFrame
    sql: str
    duration_seconds: float
    name: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.frame)

    @property
    def succeeded(self) -> bool:
        return self.error is None


class AnalyticsEngine:
    """Read-only SQL access to the lake."""

    def __init__(self, settings: Settings | None = None, *, read_only_guard: bool = True) -> None:
        self.settings = settings or get_settings()
        self.read_only_guard = read_only_guard
        self.connection = duckdb.connect(database=":memory:")
        # A modest thread count keeps the dashboard responsive on a laptop while
        # still parallelising scans.
        self.connection.execute("SET threads TO 4")
        self.registered: dict[str, str] = {}
        self.refresh_views()

    # -- view management ----------------------------------------------------

    def refresh_views(self) -> dict[str, str]:
        """(Re)create a view for every dataset present in the lake."""
        self.registered.clear()
        layers = {
            "bronze": self.settings.paths.bronze,
            "silver": self.settings.paths.silver,
            "gold": self.settings.paths.gold,
            "quarantine": self.settings.paths.quarantine,
        }

        for layer, base in layers.items():
            if not base.exists():
                continue
            for directory in sorted(base.iterdir()):
                if not directory.is_dir() or not dataset_exists(directory):
                    continue
                view = directory.name
                try:
                    self._create_view(view, directory)
                    self.registered[view] = layer
                except duckdb.Error as exc:
                    log.warning("view_creation_failed", view=view, error=str(exc)[:200])

        log.info("analytics_views_registered", views=len(self.registered))
        return dict(self.registered)

    def _create_view(self, view: str, directory: Path) -> None:
        pattern = (directory / "**" / "*.parquet").as_posix()
        self.connection.execute(f"""
            CREATE OR REPLACE VIEW "{view}" AS
            SELECT * FROM read_parquet(
                '{pattern}',
                hive_partitioning = true,
                union_by_name = true
            )
            """)

    def views(self) -> list[str]:
        return sorted(self.registered)

    def table_exists(self, name: str) -> bool:
        return name in self.registered

    # -- querying -----------------------------------------------------------

    def query(self, sql: str, name: str = "", **parameters: Any) -> QueryResult:
        """Run a query and return rows plus timing.

        Errors are captured rather than raised: a dashboard panel whose query
        fails should render an error message, not take down the page.
        """
        if self.read_only_guard and _FORBIDDEN.match(sql):
            return QueryResult(
                pd.DataFrame(),
                sql,
                0.0,
                name,
                error="Only read queries are permitted through the analytics engine",
            )

        started = time.perf_counter()
        try:
            if parameters:
                frame = self.connection.execute(sql, parameters).fetch_df()
            else:
                frame = self.connection.execute(sql).fetch_df()
            duration = time.perf_counter() - started
            log.debug(
                "query_executed",
                name=name or "adhoc",
                rows=len(frame),
                duration_seconds=round(duration, 4),
            )
            return QueryResult(frame, sql, duration, name)
        except duckdb.Error as exc:
            duration = time.perf_counter() - started
            log.warning("query_failed", name=name or "adhoc", error=str(exc)[:300])
            return QueryResult(pd.DataFrame(), sql, duration, name, error=str(exc)[:500])

    def df(self, sql: str, **parameters: Any) -> pd.DataFrame:
        """Convenience wrapper returning just the frame."""
        return self.query(sql, **parameters).frame

    def explain(self, sql: str) -> str:
        """Return the query plan - used by docs/performance.md."""
        try:
            rows = self.connection.execute(f"EXPLAIN {sql}").fetchall()
            return "\n".join(str(row[-1]) for row in rows)
        except duckdb.Error as exc:
            return f"EXPLAIN failed: {exc}"

    # -- SQL files ----------------------------------------------------------

    def run_sql_file(self, path: Path) -> list[QueryResult]:
        """Execute every statement in a ``.sql`` file.

        Statements are split on semicolons and named from the ``-- name:``
        comment that precedes them, so results can be labelled in reports rather
        than numbered.
        """
        if not path.exists():
            raise FileNotFoundError(f"SQL file not found: {path}")

        content = path.read_text(encoding="utf-8")
        results: list[QueryResult] = []

        for statement, name in _split_statements(content):
            if not statement.strip():
                continue
            results.append(self.query(statement, name=name or path.stem))
        return results

    def run_all_sql(self, directory: Path | None = None) -> dict[str, list[QueryResult]]:
        """Execute every SQL file in ``sql/``, in filename order."""
        directory = directory or (self.settings.paths.metadata.parent / "sql")
        if not directory.exists():
            log.warning("sql_directory_missing", path=str(directory))
            return {}

        outcomes: dict[str, list[QueryResult]] = {}
        for path in sorted(directory.glob("*.sql")):
            outcomes[path.name] = self.run_sql_file(path)
        return outcomes

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> AnalyticsEngine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _split_statements(content: str) -> list[tuple[str, str]]:
    """Split a SQL file into ``(statement, name)`` pairs.

    Naively splitting on ``;`` is wrong, and wrong in a way that produces
    baffling errors: a semicolon inside a ``--`` comment (or a string literal)
    cuts the file mid-sentence, and DuckDB then tries to parse English prose.
    This walks the text tracking comment and quote state, and splits only on
    semicolons that are genuinely at statement level.

    Statement names come from ``-- name: something`` markers; chunks containing
    only comments are dropped so a trailing explanation block is not submitted
    as a query.
    """
    statements: list[tuple[str, str]] = []
    buffer: list[str] = []

    in_line_comment = False
    in_block_comment = False
    in_single_quote = False
    in_double_quote = False

    index = 0
    length = len(content)

    while index < length:
        char = content[index]
        pair = content[index : index + 2]

        if in_line_comment:
            buffer.append(char)
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            buffer.append(char)
            if pair == "*/":
                buffer.append(content[index + 1])
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if in_single_quote or in_double_quote:
            buffer.append(char)
            if in_single_quote and char == "'":
                in_single_quote = False
            elif in_double_quote and char == '"':
                in_double_quote = False
            index += 1
            continue

        if pair == "--":
            in_line_comment = True
            buffer.append(pair)
            index += 2
            continue
        if pair == "/*":
            in_block_comment = True
            buffer.append(pair)
            index += 2
            continue
        if char == "'":
            in_single_quote = True
            buffer.append(char)
            index += 1
            continue
        if char == '"':
            in_double_quote = True
            buffer.append(char)
            index += 1
            continue

        if char == ";":
            statements.append(_finalise_statement("".join(buffer)))
            buffer = []
            index += 1
            continue

        buffer.append(char)
        index += 1

    if buffer:
        statements.append(_finalise_statement("".join(buffer)))

    return [(sql, name) for sql, name in statements if sql]


def _finalise_statement(chunk: str) -> tuple[str, str]:
    """Extract the statement name and drop comment-only chunks."""
    name = ""
    match = re.search(r"--\s*name:\s*(.+)", chunk)
    if match:
        name = match.group(1).strip()

    without_comments = re.sub(r"/\*.*?\*/", "", chunk, flags=re.DOTALL)
    without_comments = "\n".join(
        line for line in without_comments.splitlines() if not line.strip().startswith("--")
    )
    if not without_comments.strip():
        return "", name
    return chunk.strip(), name


@lru_cache(maxsize=1)
def get_analytics_engine() -> AnalyticsEngine:
    """Process-wide analytics engine."""
    return AnalyticsEngine()


def reset_analytics_engine() -> None:
    engine = (
        get_analytics_engine.__wrapped__ if hasattr(get_analytics_engine, "__wrapped__") else None
    )
    get_analytics_engine.cache_clear()
    del engine
