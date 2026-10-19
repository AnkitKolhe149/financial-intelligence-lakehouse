"""Engine abstraction.

The interface is deliberately narrow: **storage and dataset-level operations
only**. It does not try to abstract over dataframe APIs, because a
lowest-common-denominator dataframe wrapper is where this kind of design usually
goes wrong - you end up with something that is neither pandas nor Spark, is
slower than both, and hides the engine-specific optimisations that were the
reason for using Spark in the first place.

Instead, each engine implements the transformations natively against the same
declarative specs. The abstraction covers what genuinely is common: where
datasets live, how they are read and written, and what the engine can do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.core.config import Settings, get_settings
from src.core.exceptions import EngineError
from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine supports, so callers can adapt rather than guess."""

    name: str
    distributed: bool
    supports_delta: bool
    supports_streaming: bool
    supports_predicate_pushdown: bool
    max_practical_rows: int
    notes: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "distributed": self.distributed,
            "delta": self.supports_delta,
            "streaming": self.supports_streaming,
            "predicate_pushdown": self.supports_predicate_pushdown,
            "max_practical_rows": self.max_practical_rows,
            "notes": self.notes,
        }


class LakehouseEngine(ABC):
    """Storage-level operations every engine must provide."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities: ...

    @abstractmethod
    def read(
        self,
        layer: str,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> Any:
        """Read a dataset. Returns the engine's native frame type."""

    @abstractmethod
    def write(
        self,
        frame: Any,
        layer: str,
        table: str,
        *,
        mode: str = "overwrite",
        partition_by: Sequence[str] | None = None,
    ) -> int:
        """Write a dataset. Returns the row count written."""

    @abstractmethod
    def exists(self, layer: str, table: str) -> bool: ...

    @abstractmethod
    def row_count(self, layer: str, table: str) -> int: ...

    @abstractmethod
    def to_pandas(self, frame: Any) -> Any:
        """Materialise as pandas - used by serving code and tests."""

    def stop(self) -> None:  # noqa: B027 - intentionally optional, not abstract
        """Release resources.

        Deliberately concrete and empty rather than abstract: the pandas engine
        holds nothing to release, and forcing every engine to implement an empty
        method would be ceremony without benefit.
        """


def get_engine(name: str | None = None, settings: Settings | None = None) -> LakehouseEngine:
    """Construct the requested engine.

    Spark is imported lazily: importing PySpark on a machine without a JVM fails,
    and the pandas path must not pay for a dependency it does not use.
    """
    settings = settings or get_settings()
    name = (name or settings.engine).lower()

    if name == "pandas":
        from src.engines.pandas_engine import PandasEngine

        return PandasEngine(settings)

    if name == "spark":
        try:
            from src.engines.spark_engine import SparkEngine
        except ImportError as exc:
            raise EngineError(
                "PySpark is not available in this environment",
                detail=str(exc)[:200],
                hint=(
                    "The Spark engine runs inside the `spark` container: "
                    "docker compose run --rm spark python -m src.pipelines.spark_pipeline"
                ),
            ) from exc
        return SparkEngine(settings)

    raise EngineError(f"Unknown engine {name!r}; expected 'pandas' or 'spark'")
