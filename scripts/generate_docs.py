"""Generate the documentation that must never drift from the code.

    python scripts/generate_docs.py

Produces:

    docs/data_dictionary.md   from the canonical schema registry
    docs/lineage.md           from the recorded lineage edges

Both are generated because both describe the code, and a hand-maintained
description of code is a description that is wrong within a month. The prose
documents (architecture, decisions, methodology) are written by hand, because
they explain *why*, which no generator can infer.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings, project_root
from src.core.schema import (
    BRONZE_AUDIT_COLUMNS,
    SCHEMA_EVOLUTION,
    SCHEMAS,
    SILVER_AUDIT_COLUMNS,
    table_name,
)

GENERATED_NOTE = (
    "> **Generated file.** Produced by `python scripts/generate_docs.py` from "
    "`src/core/schema.py`. Edit the schema registry, not this document."
)


def data_dictionary() -> str:
    lines: list[str] = [
        "# Data Dictionary",
        "",
        GENERATED_NOTE,
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} - "
        f"{len(SCHEMAS)} canonical datasets.",
        "",
        "Every dataset below is declared once in the schema registry and consumed by "
        "ingestion, Silver casting, the data-quality suites, the published JSON "
        "contracts and this document. A column added there appears in all five.",
        "",
        "## Contents",
        "",
    ]
    for name in SCHEMAS:
        lines.append(f"- [{name}](#{name})")
    lines.append("")

    for name, schema in SCHEMAS.items():
        lines.extend(
            [
                f"## {name}",
                "",
                schema.description,
                "",
                "| Property | Value |",
                "| --- | --- |",
                f"| Business key | `{', '.join(schema.business_key)}` |",
                f"| Bronze table | `{table_name('bronze', name)}` |",
                f"| Silver table | `{table_name('silver', name)}` |",
                f"| Schema version | {schema.schema_version} |",
                f"| Owner | {schema.owner} |",
                f"| Classification | {schema.classification} |",
                f"| Partition column | `{schema.partition_column or 'none'}` |",
                f"| Deduplication hash columns | `{', '.join(schema.effective_hash_columns())}` |",
                "",
                "### Fields",
                "",
                "| Field | Type | Nullable | Description | Business meaning |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for field in schema.fields:
            nullable = "yes" if field.nullable else "**NOT NULL**"
            meaning = field.business_meaning or "-"
            constraints = []
            if field.domain:
                constraints.append(f"one of: {', '.join(field.domain)}")
            if field.minimum is not None:
                constraints.append(f"min {field.minimum}")
            if field.maximum is not None:
                constraints.append(f"max {field.maximum}")
            description = field.description
            if constraints:
                description += f" ({'; '.join(constraints)})"
            lines.append(
                f"| `{field.name}` | {field.dtype} | {nullable} | {description} | {meaning} |"
            )
        lines.append("")

        if schema.foreign_keys:
            lines.extend(
                [
                    "### Referential integrity",
                    "",
                    "| Column | References | On violation |",
                    "| --- | --- | --- |",
                ]
            )
            for fk in schema.foreign_keys:
                action = (
                    f"quarantine (`{fk.error_code}`)"
                    if fk.blocking
                    else "null the reference and keep the record"
                )
                lines.append(
                    f"| `{fk.column}` | `{fk.references_dataset}.{fk.references_column}` | {action} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Audit columns",
            "",
            "Provenance is attached at ingestion and carried through every layer. "
            "Underscore-prefixed names never collide with business columns and are "
            "trivially strippable when publishing.",
            "",
            "### Bronze",
            "",
            "| Column | Type | Purpose |",
            "| --- | --- | --- |",
        ]
    )
    bronze_purpose = {
        "_source_system": "Which upstream system supplied the record.",
        "_source_file": "The file or endpoint it arrived in.",
        "_source_format": "csv | json | xml | xlsx | rest_api.",
        "_ingestion_timestamp": "When the platform read it.",
        "_ingestion_run_id": "The pipeline run that ingested it.",
        "_record_hash": "Content fingerprint; drives deduplication and idempotency.",
        "_schema_version": "Which source schema version the record conforms to.",
        "_source_row_number": "Position within the source file, for tracing back.",
    }
    for column, dtype in BRONZE_AUDIT_COLUMNS.items():
        lines.append(f"| `{column}` | {dtype} | {bronze_purpose.get(column, '')} |")

    lines.extend(
        [
            "",
            "### Silver",
            "",
            "| Column | Type | Purpose |",
            "| --- | --- | --- |",
        ]
    )
    silver_purpose = {
        "_source_system": "Carried forward from Bronze.",
        "_record_hash": "Carried forward; the dedup key.",
        "_ingestion_run_id": "The run that ingested the record.",
        "_silver_run_id": "The run that validated and typed it.",
        "_processed_timestamp": "When Silver processed it.",
        "_schema_version": "Source schema version.",
        "_is_late_arriving": "True when posting lagged occurrence by over 24 hours.",
    }
    for column, dtype in SILVER_AUDIT_COLUMNS.items():
        lines.append(f"| `{column}` | {dtype} | {silver_purpose.get(column, '')} |")

    lines.extend(["", "## Schema evolution", ""])
    for dataset, versions in SCHEMA_EVOLUTION.items():
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Version | Columns added |")
        lines.append("| --- | --- |")
        for version, columns in versions.items():
            lines.append(f"| {version} | {', '.join(f'`{c}`' for c in columns) or '(initial)'} |")
        lines.append("")
    lines.extend(
        [
            "Bronze accepts new columns without a migration (schema-on-read) and Silver "
            "back-fills nulls for records written before a column existed. Both schema "
            "versions coexist in one table; `sql/05_data_quality.sql` reports the split.",
            "",
            "## Derived columns",
            "",
            "| Column | Computed in | Definition |",
            "| --- | --- | --- |",
            "| `amount_usd` | Silver | `amount` x a fixed reference rate for its currency. "
            "A production system would join a rates dimension with effective dating; the "
            "fixed rate keeps the demo deterministic and is documented as a simplification. |",
            "| `transaction_date` | Silver | Date part of `transaction_timestamp` (UTC). |",
            "| `transaction_month` | Silver | `YYYY-MM`; the partition key. Monthly rather "
            "than daily because daily partitions measured badly at this volume - see "
            "`docs/performance.md`. |",
            "| `fiscal_year` | Silver | US federal fiscal year: the calendar year, plus one "
            "from 1 October. |",
            "| `_is_late_arriving` | Silver | `posted_timestamp` more than 24 hours after "
            "`transaction_timestamp`. |",
            "| `risk_score` / `risk_level` / `risk_reasons` | Gold | Rule engine output; "
            "every point is attributable to a named rule. |",
            "| `anomaly_score` / `is_anomaly` | Gold | Isolation Forest output, normalised "
            "to [0, 1] where 1 is most anomalous. |",
            "",
        ]
    )
    return "\n".join(lines)


def lineage_document() -> str:
    settings = get_settings()
    path = Path(settings.paths.metadata) / "lineage.json"

    lines = [
        "# Data Lineage",
        "",
        "> **Generated file.** Produced by `python scripts/generate_docs.py` from "
        "`metadata/lineage.json`, which the pipelines emit as they run.",
        "",
        "Lineage here is **emitted by the code that performs each transformation**, not "
        "reconstructed by parsing SQL afterwards. Each edge records the transformations "
        "applied and the row counts on both sides, so the number of records lost between "
        "any two datasets is a fact rather than an inference.",
        "",
    ]

    if not path.exists():
        lines.extend(
            [
                "No lineage has been recorded yet. Run the pipeline first:",
                "",
                "```bash",
                "python scripts/run_demo.py",
                "python scripts/generate_docs.py",
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    document = json.loads(path.read_text(encoding="utf-8"))
    edges = document.get("edges", [])

    lines.extend(
        [
            f"Captured from run `{document.get('last_run_id', 'unknown')}` "
            f"at {document.get('generated_at', 'unknown')} - {len(edges)} edges.",
            "",
            "## Flow",
            "",
            "```mermaid",
            "graph LR",
        ]
    )

    def node_id(name: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in name)

    seen: set[str] = set()
    for edge in edges:
        for dataset, layer in (
            (edge["source"], edge.get("source_layer")),
            (edge["target"], edge.get("target_layer")),
        ):
            if dataset in seen:
                continue
            seen.add(dataset)
            shape = {
                "source": ("([", "])"),
                "bronze": ("[", "]"),
                "silver": ("[", "]"),
                "gold": ("{{", "}}"),
            }.get(layer or "", ("[", "]"))
            lines.append(f'    {node_id(dataset)}{shape[0]}"{dataset}"{shape[1]}')

    for edge in edges:
        label = str(len(edge.get("transformation", []))) + " steps"
        lines.append(f"    {node_id(edge['source'])} -->|{label}| {node_id(edge['target'])}")
    lines.extend(["```", "", "## Edges", ""])

    by_target: dict[str, list[dict]] = {}
    for edge in edges:
        by_target.setdefault(edge["target"], []).append(edge)

    for target in sorted(by_target):
        lines.extend([f"### `{target}`", ""])
        for edge in by_target[target]:
            lines.append(f"**From `{edge['source']}`**")
            lines.append("")
            if edge.get("records_in") or edge.get("records_out"):
                dropped = edge.get("records_dropped", 0)
                lines.append(
                    f"- Records: {edge.get('records_in', 0):,} in -> "
                    f"{edge.get('records_out', 0):,} out"
                    + (f" ({dropped:,} dropped)" if dropped else "")
                )
            transformations = edge.get("transformation", [])
            if transformations:
                lines.append("- Transformations:")
                lines.extend(f"  - `{t}`" for t in transformations)
            lines.append("")

    lines.extend(
        [
            "## Where records go",
            "",
            "Records dropped between Bronze and Silver are **not deleted**. Every one is "
            "written to the quarantine dataset with its error code, the raw payload as "
            "received, the source file and the run that rejected it. "
            "`sql/05_data_quality.sql` reports them by error code and by source system.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    docs = project_root() / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    outputs = {
        "data_dictionary.md": data_dictionary(),
        "lineage.md": lineage_document(),
    }
    for filename, content in outputs.items():
        path = docs / filename
        path.write_text(content, encoding="utf-8")
        print(f"  wrote {path.relative_to(project_root())} ({len(content.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
