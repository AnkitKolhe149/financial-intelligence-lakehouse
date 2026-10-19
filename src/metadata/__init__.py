"""Operational metadata: pipeline runs, quality results, lineage, model registry.

This is the platform's *control plane*, and it deliberately does not live in the
lake. Parquet files are excellent for analytical data and poor for the small,
frequently-updated, transactional records that describe pipeline executions -
which is exactly what a relational database is for.

PostgreSQL is the intended backend. SQLite is the zero-setup fallback so the
demo runs before anyone starts a container; the schema and every query are
identical across both, because they go through SQLAlchemy Core.
"""

from src.metadata.store import MetadataStore, get_metadata_store

__all__ = ["MetadataStore", "get_metadata_store"]
