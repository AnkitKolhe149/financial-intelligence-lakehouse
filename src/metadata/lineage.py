"""Data lineage capture.

Lineage here is **emitted by the code that does the work**, not reconstructed
afterwards by parsing SQL or guessing from file names. Each transformation
registers the edge it just created, with the list of operations it applied and
the row counts on both sides. That gives two things a parser could never give
you: the transformation semantics, and the row-count delta that tells you where
records were lost.

Output has two forms, deliberately:

* rows in ``lineage_edges`` (queryable, per-run, historical)
* ``metadata/lineage.json`` (a machine-readable snapshot committed alongside the
  code, which is what a governance reviewer will ask for)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import Settings, get_settings
from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class LineageEdge:
    """One directed edge: ``source_dataset`` -> ``target_dataset``."""

    source_dataset: str
    target_dataset: str
    transformations: list[str]
    source_layer: str = ""
    target_layer: str = ""
    pipeline_name: str = ""
    pipeline_version: str = ""
    records_in: int = 0
    records_out: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_row(self, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "source_dataset": self.source_dataset,
            "source_layer": self.source_layer,
            "target_dataset": self.target_dataset,
            "target_layer": self.target_layer,
            "transformations": list(self.transformations),
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "records_in": self.records_in,
            "records_out": self.records_out,
            "created_at": self.created_at,
        }

    def to_json(self) -> dict[str, Any]:
        payload = {
            "source": self.source_dataset,
            "source_layer": self.source_layer,
            "target": self.target_dataset,
            "target_layer": self.target_layer,
            "transformation": list(self.transformations),
            "pipeline": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
        }
        if self.records_in or self.records_out:
            payload["records_in"] = self.records_in
            payload["records_out"] = self.records_out
            # The delta is the interesting number: it is where records went.
            payload["records_dropped"] = max(self.records_in - self.records_out, 0)
        return payload


class LineageCollector:
    """Accumulates edges during a run, then persists them."""

    def __init__(self, run_id: str, pipeline_name: str, pipeline_version: str) -> None:
        self.run_id = run_id
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.edges: list[LineageEdge] = []

    def add(
        self,
        source: str,
        target: str,
        transformations: list[str],
        *,
        source_layer: str = "",
        target_layer: str = "",
        records_in: int = 0,
        records_out: int = 0,
    ) -> LineageEdge:
        edge = LineageEdge(
            source_dataset=source,
            target_dataset=target,
            transformations=transformations,
            source_layer=source_layer,
            target_layer=target_layer,
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            records_in=records_in,
            records_out=records_out,
        )
        self.edges.append(edge)
        log.debug(
            "lineage_edge",
            source=source,
            target=target,
            transformations=len(transformations),
            records_in=records_in,
            records_out=records_out,
        )
        return edge

    def persist(self, store: Any | None = None) -> int:
        """Write edges to the metadata store. Returns the number written."""
        if not self.edges:
            return 0
        if store is None:
            from src.metadata.store import get_metadata_store

            store = get_metadata_store()
        store.record_lineage([edge.to_row(self.run_id) for edge in self.edges])
        return len(self.edges)

    def export_json(self, settings: Settings | None = None, path: Path | None = None) -> Path:
        """Write/merge the human-reviewable ``metadata/lineage.json`` snapshot.

        Edges are merged by ``(source, target)`` so the file reflects the current
        shape of the platform rather than growing by one entry per run.
        """
        settings = settings or get_settings()
        target = path or (Path(settings.paths.metadata) / "lineage.json")
        target.parent.mkdir(parents=True, exist_ok=True)

        existing: dict[str, dict[str, Any]] = {}
        if target.exists():
            try:
                document = json.loads(target.read_text(encoding="utf-8"))
                for entry in document.get("edges", []):
                    existing[f"{entry['source']}->{entry['target']}"] = entry
            except (json.JSONDecodeError, KeyError):
                log.warning("lineage_json_unreadable", path=str(target))

        for edge in self.edges:
            existing[f"{edge.source_dataset}->{edge.target_dataset}"] = edge.to_json()

        document = {
            "platform": "financial-intelligence-lakehouse",
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "last_run_id": self.run_id,
            "pipeline": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "edge_count": len(existing),
            "edges": sorted(existing.values(), key=lambda e: (e["target"], e["source"])),
        }
        target.write_text(json.dumps(document, indent=2), encoding="utf-8")
        log.info("lineage_exported", path=str(target), edges=len(existing))
        return target


def build_lineage_graph(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Adjacency list of ``source -> [targets]`` for traversal and diagrams."""
    graph: dict[str, list[str]] = {}
    for edge in edges:
        source = edge.get("source_dataset") or edge.get("source")
        target = edge.get("target_dataset") or edge.get("target")
        if source and target:
            graph.setdefault(str(source), []).append(str(target))
    return graph


def upstream_of(dataset: str, edges: list[dict[str, Any]], max_depth: int = 10) -> list[str]:
    """Every dataset that feeds ``dataset``, transitively.

    This is the question asked during an incident - "what fed this bad table?" -
    and the reason lineage is worth capturing at all.
    """
    parents: dict[str, list[str]] = {}
    for edge in edges:
        source = str(edge.get("source_dataset") or edge.get("source") or "")
        target = str(edge.get("target_dataset") or edge.get("target") or "")
        if source and target:
            parents.setdefault(target, []).append(source)

    seen: list[str] = []
    frontier = [dataset]
    for _ in range(max_depth):
        next_frontier: list[str] = []
        for node in frontier:
            for parent in parents.get(node, []):
                if parent not in seen:
                    seen.append(parent)
                    next_frontier.append(parent)
        if not next_frontier:
            break
        frontier = next_frontier
    return seen
