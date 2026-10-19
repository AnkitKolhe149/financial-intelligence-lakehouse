"""Neo4j connectivity, with graceful degradation as a first-class behaviour.

The graph is an **enrichment and investigation layer**, never a dependency of
the core pipeline. Every entry point here is written so that an absent or broken
Neo4j produces a logged warning and a ``False``/empty result - never an
exception that reaches a pipeline stage. That is a deliberate architectural
position: relationship analysis is valuable, but a payments oversight platform
must not stop ingesting money movements because a graph database is down.

Connectivity is probed once and cached briefly, so that a dashboard page which
asks "is the graph available?" fifteen times does not open fifteen sockets to a
host that is not listening.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from src.core.config import Settings, get_settings
from src.core.exceptions import GraphError
from src.core.logging import get_logger

log = get_logger(__name__)

#: Cached availability probe: ``(checked_at, result)``.
_PROBE_CACHE: tuple[float, bool] | None = None
_PROBE_TTL_SECONDS = 30.0


class Neo4jClient:
    """Thin wrapper over the official driver with query helpers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.neo4j
        self._driver: Any = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> Any:
        """Open (or reuse) the driver. Raises :class:`GraphError` on failure."""
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - neo4j is a dependency
            raise GraphError("neo4j driver is not installed") from exc

        try:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
                connection_timeout=5,
                max_connection_lifetime=300,
            )
            self._driver.verify_connectivity()
            log.info("neo4j_connected", uri=self.config.uri, database=self.config.database)
            return self._driver
        except Exception as exc:
            self._driver = None
            raise GraphError(
                "Could not connect to Neo4j",
                uri=self.config.uri,
                detail=str(exc)[:200],
            ) from exc

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Neo4jClient:
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def session(self) -> Iterator[Any]:
        driver = self.connect()
        session = driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()

    # -- queries ------------------------------------------------------------

    def run(self, cypher: str, **parameters: Any) -> list[dict[str, Any]]:
        """Execute a read query and return plain dictionaries.

        Records are materialised inside the session; returning driver Record
        objects would give the caller values that expire when the session closes.
        """
        with self.session() as session:
            result = session.run(cypher, **parameters)
            return [record.data() for record in result]

    def write(self, cypher: str, **parameters: Any) -> dict[str, Any]:
        """Execute a write query and return the counters it produced."""
        with self.session() as session:
            result = session.run(cypher, **parameters)
            summary = result.consume()
            counters = summary.counters
            return {
                "nodes_created": counters.nodes_created,
                "relationships_created": counters.relationships_created,
                "properties_set": counters.properties_set,
                "labels_added": counters.labels_added,
            }

    def write_batch(
        self, cypher: str, rows: list[dict[str, Any]], batch_size: int = 1000
    ) -> dict[str, int]:
        """UNWIND-based batched write.

        One transaction per row would make loading 20,000 transactions take
        minutes; UNWIND over batches of a thousand turns it into a handful of
        round trips.
        """
        totals = {"nodes_created": 0, "relationships_created": 0, "properties_set": 0}
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            counters = self.write(cypher, rows=chunk)
            for key in totals:
                totals[key] += counters.get(key, 0)
        return totals

    # -- maintenance --------------------------------------------------------

    def clear(self) -> None:
        """Delete every node and relationship.

        Deletes in batches: a single ``DETACH DELETE`` over a large graph builds
        one enormous transaction and runs the server out of heap.
        """
        while True:
            result = self.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
            )
            deleted = result[0]["deleted"] if result else 0
            if not deleted:
                break
        log.info("neo4j_cleared")

    def create_constraints(self) -> list[str]:
        """Create uniqueness constraints for every node key.

        Constraints are not only correctness: each one creates a backing index,
        which is what makes ``MATCH (v:Vendor {vendor_id: $id})`` a lookup rather
        than a scan of every vendor.
        """
        statements = [
            "CREATE CONSTRAINT agency_id IF NOT EXISTS FOR (a:Agency) REQUIRE a.agency_id IS UNIQUE",
            "CREATE CONSTRAINT department_id IF NOT EXISTS FOR (d:Department) REQUIRE d.department_id IS UNIQUE",
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
            "CREATE CONSTRAINT vendor_id IF NOT EXISTS FOR (v:Vendor) REQUIRE v.vendor_id IS UNIQUE",
            "CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE",
            "CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE",
        ]
        applied = []
        for statement in statements:
            try:
                self.write(statement)
                applied.append(statement.split()[2])
            except GraphError:
                raise
            except Exception as exc:
                log.warning("constraint_failed", statement=statement[:60], error=str(exc)[:150])
        log.info("neo4j_constraints_ready", count=len(applied))
        return applied

    def node_counts(self) -> dict[str, int]:
        rows = self.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
        )
        return {row["label"]: row["count"] for row in rows if row["label"]}

    def relationship_counts(self) -> dict[str, int]:
        rows = self.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
        )
        return {row["type"]: row["count"] for row in rows}


def graph_available(settings: Settings | None = None, *, use_cache: bool = True) -> bool:
    """Is Neo4j reachable right now?

    Cached for a few seconds because callers ask repeatedly (a dashboard page
    render, an API health check) and each miss costs a connection attempt with a
    five-second timeout.
    """
    global _PROBE_CACHE

    settings = settings or get_settings()
    if not settings.graph.enabled:
        return False

    now = time.monotonic()
    if use_cache and _PROBE_CACHE is not None:
        checked_at, result = _PROBE_CACHE
        if now - checked_at < _PROBE_TTL_SECONDS:
            return result

    client = Neo4jClient(settings)
    try:
        client.connect()
        available = True
    except GraphError as exc:
        log.info("graph_unavailable", detail=exc.message, uri=settings.neo4j.uri)
        available = False
    finally:
        client.close()

    _PROBE_CACHE = (now, available)
    return available


def reset_probe_cache() -> None:
    global _PROBE_CACHE
    _PROBE_CACHE = None
