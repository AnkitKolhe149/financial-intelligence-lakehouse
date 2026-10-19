"""PySpark + Delta Lake execution engine.

Runs inside the ``spark`` container only (PySpark requires a JVM and supports
Python <= 3.12). Everything here is written to be lifted onto Databricks with
configuration changes rather than code changes:

* The session is built with the Delta extensions and catalog, which is exactly
  what a Databricks runtime provides by default.
* Tables are addressed by layer and name through the same
  ``Settings.dataset_path`` used by the pandas engine, so pointing at
  ``abfss://``/``s3://`` or a Unity Catalog three-part name is a settings change.
* Writes go to Delta when ``delta-spark`` is importable and fall back to Parquet
  when it is not, so the container works even without the Delta package.

The Spark-specific optimisations that matter are applied explicitly and
commented, because "we used Spark" is not an optimisation and an interviewer
will ask which knobs were actually turned.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from src.core.config import Settings
from src.core.exceptions import EngineError
from src.core.logging import get_logger
from src.engines.base import EngineCapabilities, LakehouseEngine

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession

log = get_logger(__name__)


def delta_available() -> bool:
    """Is delta-spark importable?"""
    try:
        import delta  # noqa: F401
    except ImportError:
        return False
    return True


def build_session(
    settings: Settings, app_name: str = "financial-intelligence-lakehouse"
) -> SparkSession:
    """Create a tuned SparkSession with Delta enabled when available."""
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        # Adaptive Query Execution: lets Spark re-plan mid-query using actual
        # partition statistics. It is the single highest-value flag for the
        # skewed joins in this workload (a handful of vendors hold most rows).
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # 200 shuffle partitions is the default and is wildly wrong for a
        # single-container dataset: it produces 200 tiny tasks whose scheduling
        # costs more than the work. AQE coalesces, but starting lower helps.
        .config("spark.sql.shuffle.partitions", "16")
        # Broadcast the dimensions. Every dimension here is a few hundred KB, so
        # broadcasting turns each dimension join from a shuffle into a map-side
        # lookup.
        .config("spark.sql.autoBroadcastJoinThreshold", str(32 * 1024 * 1024))
        # Arrow for the pandas boundary - an order of magnitude faster than the
        # row-by-row default when collecting to pandas for the ML step.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.parquet.compression.codec", settings.storage.compression)
        # Dynamic partition overwrite is what makes re-running one day idempotent
        # without rewriting the whole table - the Spark equivalent of the pandas
        # engine's partition replace.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.session.timeZone", "UTC")
        .master("local[*]")
    )

    if delta_available():
        builder = builder.config(
            "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
        ).config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        try:
            from delta import configure_spark_with_delta_pip

            builder = configure_spark_with_delta_pip(builder)
        except Exception:
            pass

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    log.info(
        "spark_session_created",
        version=session.version,
        delta=delta_available(),
        shuffle_partitions=session.conf.get("spark.sql.shuffle.partitions"),
        master=session.sparkContext.master,
    )
    return session


class SparkEngine(LakehouseEngine):
    """Distributed engine over Delta (or Parquet when Delta is absent)."""

    def __init__(
        self, settings: Settings | None = None, session: SparkSession | None = None
    ) -> None:
        super().__init__(settings)
        self.session = session or build_session(self.settings)
        self._format = "delta" if delta_available() else "parquet"
        if self._format == "parquet":
            log.warning(
                "delta_unavailable",
                detail="falling back to Parquet; install delta-spark for ACID writes",
            )

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="spark",
            distributed=True,
            supports_delta=self._format == "delta",
            supports_streaming=True,
            supports_predicate_pushdown=True,
            max_practical_rows=10_000_000_000,
            notes=(
                f"Writing {self._format}. AQE and skew-join handling enabled; "
                "dimensions are broadcast. Maps directly onto Databricks."
            ),
        )

    @property
    def storage_format(self) -> str:
        return self._format

    # -- io -----------------------------------------------------------------

    def _path(self, layer: str, table: str) -> str:
        # Spark takes URIs, not Path objects; as_posix keeps the path valid when
        # the container mounts a Windows-authored volume.
        return self.settings.dataset_path(layer, table).as_posix()

    def read(
        self,
        layer: str,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> DataFrame:
        path = self._path(layer, table)
        reader = self.session.read.format(self._format)
        if self._format == "parquet":
            # Hive-style partition discovery, matching what the pandas engine writes.
            reader = reader.option("basePath", path)

        frame = reader.load(path)

        # Predicate pushdown: applying the filter before the projection lets
        # Spark prune partitions and row groups rather than reading and
        # discarding.
        if filters:
            for column, operator, value in filters:
                frame = frame.filter(_condition(frame, column, operator, value))
        if columns:
            frame = frame.select(*columns)
        return frame

    def write(
        self,
        frame: DataFrame,
        layer: str,
        table: str,
        *,
        mode: str = "overwrite",
        partition_by: Sequence[str] | None = None,
    ) -> int:
        path = self._path(layer, table)
        partitions = list(partition_by or self.settings.storage.partitions_for(table) or [])

        spark_mode = {
            "overwrite": "overwrite",
            "append": "append",
            "overwrite_partitions": "overwrite",
        }.get(mode, mode)

        writer = frame.write.format(self._format).mode(spark_mode)
        if partitions:
            writer = writer.partitionBy(*partitions)
        if self._format == "delta":
            # Absorb additive schema changes, the Delta equivalent of the Bronze
            # schema-on-read behaviour.
            writer = writer.option("mergeSchema", "true")

        writer.save(path)
        count = frame.count()
        log.info(
            "spark_write",
            table=table,
            layer=layer,
            format=self._format,
            rows=count,
            mode=spark_mode,
            partitions=partitions,
        )
        return count

    def exists(self, layer: str, table: str) -> bool:
        from py4j.protocol import Py4JJavaError

        try:
            self.read(layer, table).limit(1).count()
        except (Py4JJavaError, Exception):
            return False
        return True

    def row_count(self, layer: str, table: str) -> int:
        try:
            return int(self.read(layer, table).count())
        except Exception:
            return 0

    def to_pandas(self, frame: DataFrame) -> Any:
        """Collect to pandas via Arrow.

        Only used where a single-node library is genuinely required (the
        scikit-learn model). Collecting a distributed frame is otherwise the
        classic way to turn a cluster into an expensive laptop.
        """
        return frame.toPandas()

    def create_or_replace_view(self, frame: DataFrame, name: str) -> None:
        frame.createOrReplaceTempView(name)

    def sql(self, statement: str) -> DataFrame:
        return self.session.sql(statement)

    def optimise(self, layer: str, table: str, zorder_by: Sequence[str] | None = None) -> None:
        """Run Delta OPTIMIZE (with optional Z-ORDER).

        Compacts the small files that frequent partitioned writes create. Without
        it, a table written daily for a year is thousands of tiny files and every
        query pays for opening them.
        """
        if self._format != "delta":
            log.info("optimise_skipped", reason="not a Delta table", table=table)
            return
        path = self._path(layer, table)
        statement = f"OPTIMIZE delta.`{path}`"
        if zorder_by:
            statement += f" ZORDER BY ({', '.join(zorder_by)})"
        try:
            self.session.sql(statement)
            log.info("delta_optimised", table=table, zorder=list(zorder_by or []))
        except Exception as exc:
            log.warning("delta_optimise_unavailable", table=table, error=str(exc)[:200])

    def vacuum(self, layer: str, table: str, retain_hours: int = 168) -> None:
        """Remove files no longer referenced by the Delta log."""
        if self._format != "delta":
            return
        path = self._path(layer, table)
        try:
            self.session.sql(f"VACUUM delta.`{path}` RETAIN {retain_hours} HOURS")
            log.info("delta_vacuumed", table=table, retain_hours=retain_hours)
        except Exception as exc:
            log.warning("delta_vacuum_failed", table=table, error=str(exc)[:200])

    def stop(self) -> None:
        try:
            self.session.stop()
            log.info("spark_session_stopped")
        except Exception:
            pass


def _condition(frame: DataFrame, column: str, operator: str, value: Any) -> Any:
    """Translate a portable filter tuple into a Spark Column expression."""
    target = frame[column]
    mapping = {
        "=": lambda: target == value,
        "==": lambda: target == value,
        "!=": lambda: target != value,
        ">": lambda: target > value,
        ">=": lambda: target >= value,
        "<": lambda: target < value,
        "<=": lambda: target <= value,
        "in": lambda: target.isin(list(value)),
    }
    builder = mapping.get(operator)
    if builder is None:
        raise EngineError(f"Unsupported filter operator {operator!r}")
    return builder()
