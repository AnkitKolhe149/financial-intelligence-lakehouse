"""SQL analytics over the lake.

DuckDB reads the Parquet files in place. There is no load step, no second copy
of the data and no ETL into a warehouse - which is precisely the lakehouse
argument: analytical SQL runs directly against the same files the pipeline
wrote, so a query cannot be looking at stale data because there is only one
copy.

The division of responsibility between the two databases is deliberate:

* **PostgreSQL** holds operational metadata - pipeline runs, quality results,
  lineage. Small rows, written constantly, read transactionally.
* **DuckDB** answers analytical questions - scans, aggregations, window
  functions over millions of rows. Columnar, vectorised, read-only here.

Using one for both would mean either running analytical scans against an OLTP
database or writing pipeline heartbeats into a columnar store. Neither ends
well.
"""

from src.analytics.duckdb_client import AnalyticsEngine, get_analytics_engine

__all__ = ["AnalyticsEngine", "get_analytics_engine"]
