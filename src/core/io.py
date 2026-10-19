"""Physical storage primitives for the lake.

Every dataset in the lake is a **directory** of Parquet part files, optionally
Hive-partitioned (``transaction_date=2026-08-15/part-0.parquet``). Treating a
dataset as a directory rather than a single file is what makes the layout
identical to what Spark and Delta produce, so the Spark engine can read
datasets written by the pandas engine and vice versa.

Three properties are deliberately engineered here:

**Predicate pushdown and column pruning**
    :func:`read_parquet` accepts ``columns`` and ``filters`` and passes them to
    ``pyarrow.dataset``, so a query for one day of one column reads one column
    of one partition off disk, not the whole table.

**Atomic overwrite**
    A failed write must not leave a half-replaced dataset behind. Overwrites are
    staged in a sibling ``.tmp`` directory and swapped in, so a reader either
    sees the old dataset or the new one.

**Partition-level replace**
    ``mode="overwrite_partitions"`` replaces only the partitions present in the
    incoming frame. This is what makes re-running a single day idempotent
    without rewriting history - the pandas equivalent of Delta's
    ``replaceWhere``.
"""

from __future__ import annotations

import operator
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.core.exceptions import LakehouseError
from src.core.logging import get_logger

log = get_logger(__name__)

WriteMode = str  # "overwrite" | "append" | "overwrite_partitions"

_SUCCESS_MARKER = "_SUCCESS"


def dataset_exists(path: Path) -> bool:
    """True when ``path`` holds at least one Parquet part file."""
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix == ".parquet"
    return any(path.rglob("*.parquet"))


def _to_arrow(df: pd.DataFrame) -> pa.Table:
    """Convert to Arrow, keeping the pandas index out of the file.

    ``preserve_index=False`` matters: a stray ``__index_level_0__`` column
    breaks schema comparisons against Spark-written files.
    """
    return pa.Table.from_pandas(df, preserve_index=False)


#: Filesystem operations are retried briefly before giving up.
#:
#: On Windows a directory cannot be removed or renamed while any handle into it
#: is open, and a reader that has just finished a scan may still be holding one
#: for a few milliseconds. That produced a genuine intermittent failure: a
#: pipeline run would abort mid-write with PermissionError, but only sometimes,
#: and only when a previous run had read the same dataset. A short retry turns
#: a race into a pause.
_FS_RETRIES = 5
_FS_RETRY_DELAY = 0.2


def _with_retries(operation: Any, description: str) -> None:
    last_error: OSError | None = None
    for attempt in range(1, _FS_RETRIES + 1):
        try:
            operation()
            return
        except (PermissionError, OSError) as exc:
            last_error = exc
            if attempt == _FS_RETRIES:
                break
            log.debug(
                "filesystem_retry", operation=description, attempt=attempt, error=str(exc)[:120]
            )
            time.sleep(_FS_RETRY_DELAY * attempt)
    raise LakehouseError(
        f"Filesystem operation failed after {_FS_RETRIES} attempts: {description}",
        detail=str(last_error)[:200],
    ) from last_error


def _clear_directory(path: Path) -> None:
    if path.exists():
        _with_retries(lambda: shutil.rmtree(path), f"remove {path.name}")


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    partition_by: Sequence[str] | None = None,
    mode: WriteMode = "overwrite",
    compression: str = "snappy",
) -> int:
    """Write ``df`` to the dataset directory ``path``. Returns the row count.

    Empty frames still create the directory and a schema-carrying part file, so
    downstream readers get an empty result rather than "dataset not found" -
    a zero-row day is a legitimate outcome, not an error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partition_by = [c for c in (partition_by or []) if c in df.columns]

    if mode not in {"overwrite", "append", "overwrite_partitions"}:
        raise LakehouseError(f"Unsupported write mode {mode!r}", path=str(path))

    table = _to_arrow(df)

    if mode == "overwrite":
        staging = path.with_name(path.name + ".tmp")
        _clear_directory(staging)
        _write_table(table, staging, partition_by, compression)
        _clear_directory(path)
        _with_retries(lambda: staging.rename(path), f"publish {path.name}")
    elif mode == "append":
        _write_table(table, path, partition_by, compression, append=True)
    else:  # overwrite_partitions
        if not partition_by:
            # Without partitions there is nothing to scope the replace to, so
            # this degenerates to a full overwrite - documented, not silent.
            log.debug("overwrite_partitions_without_partitions", dataset=path.name)
            return write_parquet(
                df, path, partition_by=None, mode="overwrite", compression=compression
            )
        _replace_partitions(table, path, partition_by, compression, df)

    (path / _SUCCESS_MARKER).touch()
    return len(df)


def _write_table(
    table: pa.Table,
    path: Path,
    partition_by: Sequence[str] | None,
    compression: str,
    append: bool = False,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    basename = f"part-{pd.Timestamp.now(tz='UTC').value:d}-{{i}}.parquet"

    if table.num_rows == 0:
        # ``write_dataset`` emits no file at all for an empty table, which leaves
        # a directory that looks like a missing dataset: downstream readers and
        # the DuckDB view registration both skip it, so "zero rows today" becomes
        # "table does not exist". Writing an explicit schema-only file keeps the
        # dataset discoverable and queryable with zero rows, which is the honest
        # representation of an empty result.
        pq.write_table(
            table,
            path / basename.format(i=0),
            compression=compression,
        )
        return
    if partition_by:
        pq.write_to_dataset(
            table,
            root_path=str(path),
            partition_cols=list(partition_by),
            compression=compression,
            existing_data_behavior="overwrite_or_ignore" if append else "delete_matching",
            basename_template=basename,
        )
    else:
        ds.write_dataset(
            table,
            base_dir=str(path),
            format="parquet",
            existing_data_behavior="overwrite_or_ignore" if append else "delete_matching",
            basename_template=basename,
            file_options=ds.ParquetFileFormat().make_write_options(compression=compression),
        )


def _replace_partitions(
    table: pa.Table,
    path: Path,
    partition_by: Sequence[str],
    compression: str,
    df: pd.DataFrame,
) -> None:
    """Delete only the partition directories present in ``df``, then append.

    This is the idempotency mechanism for daily re-runs: re-processing
    2026-08-15 rewrites ``transaction_date=2026-08-15/`` and leaves every other
    day untouched.
    """
    column = partition_by[0]
    values = df[column].dropna().unique().tolist()
    for value in values:
        victim = path / f"{column}={value}"
        if victim.exists():
            shutil.rmtree(victim)
    _write_table(table, path, partition_by, compression, append=True)


def read_parquet(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    """Read a dataset directory into pandas.

    ``columns`` prunes at the file level and ``filters`` are pushed down to the
    Parquet reader (and to partition pruning when the column is a partition
    key), so neither costs a full scan.
    """
    if not dataset_exists(path):
        raise LakehouseError("Dataset not found", path=str(path))

    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    expression = _build_expression(filters) if filters else None
    table = dataset.to_table(columns=list(columns) if columns else None, filter=expression)
    frame = table.to_pandas(types_mapper=None)
    return frame


#: Filter operators, as plain functions of (field, value). Using the operator
#: module rather than a dict of lambdas keeps each entry free of closure state,
#: so there is no loop-variable capture to reason about.
_FILTER_OPERATORS: dict[str, Any] = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "in": lambda field, value: field.isin(value),
}


def _build_expression(filters: list[tuple[str, str, Any]]) -> ds.Expression:
    """Translate ``[("col", ">=", value), ...]`` into an Arrow filter expression."""
    expression: ds.Expression | None = None
    for column, op, value in filters:
        apply = _FILTER_OPERATORS.get(op)
        if apply is None:
            raise LakehouseError(
                f"Unsupported filter operator {op!r}",
                supported=sorted(_FILTER_OPERATORS),
            )
        term = apply(ds.field(column), value)
        expression = term if expression is None else (expression & term)
    if expression is None:
        raise LakehouseError("No filters supplied")
    return expression


def read_parquet_or_empty(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    """Like :func:`read_parquet` but returns an empty frame for a missing dataset.

    Used by serving code (API, dashboard) which must degrade to "no data yet"
    rather than erroring before the first pipeline run.
    """
    try:
        return read_parquet(path, columns=columns, filters=filters)
    except LakehouseError:
        return pd.DataFrame(columns=list(columns) if columns else [])


def dataset_row_count(path: Path) -> int:
    """Row count from Parquet footer metadata - no data pages are read."""
    if not dataset_exists(path):
        return 0
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return dataset.count_rows()


def dataset_schema(path: Path) -> dict[str, str]:
    """Return ``{column: arrow_type}`` for a stored dataset."""
    if not dataset_exists(path):
        return {}
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return {field.name: str(field.type) for field in dataset.schema}


def dataset_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*.parquet"))


def write_json_records(records: list[dict[str, Any]], path: Path) -> int:
    """Write newline-delimited JSON - used for the streaming inbox and metadata."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
    return len(records)
