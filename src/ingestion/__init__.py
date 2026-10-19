"""Ingestion: heterogeneous sources in, canonical Bronze frames out.

Every reader returns the same :class:`~src.ingestion.base.IngestionResult`
regardless of whether it read a CSV file, an XML document or a paginated REST
endpoint. That uniformity is the whole point of the layer - Bronze does not know
or care where a record came from, only that it arrived with its provenance
attached.

Readers deliberately do **not** clean data. They map field names onto the
canonical schema, attach audit columns, and stop. Type casting, validation and
quarantine belong to Silver, because Bronze must preserve what the source
actually sent.
"""

from src.ingestion.base import IngestionResult, SourceReader, stamp_audit_columns

__all__ = ["IngestionResult", "SourceReader", "stamp_audit_columns"]
