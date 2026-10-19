"""Pandas execution engine - the local default.

Worth stating plainly, because it is a real engineering judgement rather than a
compromise: **for the dataset sizes this project runs locally, pandas is the
correct choice and is faster than Spark**. Spark's per-job scheduling, JVM
startup and shuffle machinery cost seconds; the entire 50,000-row batch pipeline
finishes in under thirty. Spark earns its overhead when data exceeds one
machine's memory or when the work genuinely parallelises across a cluster -
neither of which is true at 50k rows on a laptop.

The Spark implementation exists because the *production* answer is different,
and because writing both is what proves the transformation specs are truly
engine-independent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.core.config import Settings
from src.core.io import (
    dataset_exists,
    dataset_row_count,
    read_parquet,
    read_parquet_or_empty,
    write_parquet,
)
from src.core.logging import get_logger
from src.engines.base import EngineCapabilities, LakehouseEngine

log = get_logger(__name__)


class PandasEngine(LakehouseEngine):
    """Single-node engine over Parquet, with Arrow-level pushdown."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="pandas",
            distributed=False,
            supports_delta=False,
            supports_streaming=True,  # micro-batch over the file inbox
            supports_predicate_pushdown=True,  # via pyarrow.dataset filters
            max_practical_rows=5_000_000,
            notes=(
                "Single node. Predicate pushdown and column pruning are handled by "
                "pyarrow.dataset, so partition filters do prune at file level. "
                "Beyond a few million rows, switch to the Spark engine."
            ),
        )

    def read(
        self,
        layer: str,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        path = self.settings.dataset_path(layer, table)
        return read_parquet(path, columns=columns, filters=filters)

    def read_or_empty(
        self,
        layer: str,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        path = self.settings.dataset_path(layer, table)
        return read_parquet_or_empty(path, columns=columns, filters=filters)

    def write(
        self,
        frame: pd.DataFrame,
        layer: str,
        table: str,
        *,
        mode: str = "overwrite",
        partition_by: Sequence[str] | None = None,
    ) -> int:
        path = self.settings.dataset_path(layer, table)
        partitions = partition_by or self.settings.storage.partitions_for(table)
        return write_parquet(
            frame,
            path,
            partition_by=partitions,
            mode=mode,
            compression=self.settings.storage.compression,
        )

    def exists(self, layer: str, table: str) -> bool:
        return dataset_exists(self.settings.dataset_path(layer, table))

    def row_count(self, layer: str, table: str) -> int:
        """Row count from Parquet footers - no data pages are read."""
        return dataset_row_count(self.settings.dataset_path(layer, table))

    def to_pandas(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame
