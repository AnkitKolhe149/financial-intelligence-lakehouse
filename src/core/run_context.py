"""Pipeline run context - the unit of reproducibility and observability.

Every execution of every pipeline creates exactly one :class:`RunContext`. It
carries the identifiers that make a run reproducible (pipeline version, config
version, model version, seed) and accumulates the counters that make it
observable (records read/written/quarantined/failed, per-stage durations).

At the end of a run the context is persisted to the ``pipeline_runs`` table by
:mod:`src.metadata.store`, and the same object is what the ``/metrics`` endpoint
and the dashboard's pipeline-monitoring page read back.
"""

from __future__ import annotations

import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src import __version__ as platform_version


def new_run_id(prefix: str | None = None) -> str:
    """Generate a sortable, human-readable run identifier.

    Format: ``20260815T142530-3f9a2b`` - lexicographic sort equals chronological
    sort, which makes ``ORDER BY run_id`` in the metadata store meaningful, and
    the random suffix keeps concurrent runs from colliding.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    core = f"{stamp}-{suffix}"
    return f"{prefix}-{core}" if prefix else core


@dataclass
class StageResult:
    """Timing and volume for one stage of a pipeline."""

    stage: str
    duration_seconds: float
    records_in: int = 0
    records_out: int = 0
    status: str = "SUCCESS"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_seconds": round(self.duration_seconds, 3),
            "records_in": self.records_in,
            "records_out": self.records_out,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class RunContext:
    """Mutable state for a single pipeline execution."""

    pipeline_name: str
    pipeline_version: str
    run_id: str = field(default_factory=new_run_id)
    source: str = "all"
    engine: str = "pandas"
    config_version: str = "1.0.0"
    model_version: str | None = None
    seed: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    status: str = "RUNNING"
    error_message: str | None = None

    records_read: int = 0
    records_written: int = 0
    records_quarantined: int = 0
    records_failed: int = 0

    stages: list[StageResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    _monotonic_start: float = field(default_factory=time.perf_counter, repr=False)

    # -- environment fingerprint -------------------------------------------
    host: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=platform.python_version)
    platform_version: str = platform_version

    # -- accumulation -------------------------------------------------------

    def add_stage(
        self,
        stage: str,
        duration_seconds: float,
        records_in: int = 0,
        records_out: int = 0,
        status: str = "SUCCESS",
        **detail: Any,
    ) -> StageResult:
        result = StageResult(
            stage=stage,
            duration_seconds=duration_seconds,
            records_in=records_in,
            records_out=records_out,
            status=status,
            detail=detail,
        )
        self.stages.append(result)
        return result

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def count(
        self,
        read: int = 0,
        written: int = 0,
        quarantined: int = 0,
        failed: int = 0,
    ) -> None:
        """Accumulate record counters. Called by each stage as it completes."""
        self.records_read += read
        self.records_written += written
        self.records_quarantined += quarantined
        self.records_failed += failed

    # -- lifecycle ----------------------------------------------------------

    @property
    def duration_seconds(self) -> float:
        if self.end_time is not None:
            return round((self.end_time - self.start_time).total_seconds(), 3)
        return round(time.perf_counter() - self._monotonic_start, 3)

    @property
    def is_finished(self) -> bool:
        return self.end_time is not None

    def finish(
        self, status: str = "SUCCESS", error: BaseException | str | None = None
    ) -> RunContext:
        """Close the run. Idempotent - a second call is ignored."""
        if self.is_finished:
            return self
        self.end_time = datetime.now(UTC)
        self.status = status
        if error is not None:
            self.error_message = str(error)[:2000]
        return self

    def fail(self, error: BaseException | str) -> RunContext:
        return self.finish(status="FAILED", error=error)

    def succeed_with_warnings(self, reason: str) -> RunContext:
        """A run that produced output but tripped a non-blocking quality gate."""
        return self.finish(status="SUCCESS_WITH_WARNINGS", error=reason)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The exact shape stored in ``pipeline_runs``."""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "config_version": self.config_version,
            "model_version": self.model_version,
            "engine": self.engine,
            "source": self.source,
            "seed": self.seed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "records_read": self.records_read,
            "records_written": self.records_written,
            "records_quarantined": self.records_quarantined,
            "records_failed": self.records_failed,
            "status": self.status,
            "error_message": self.error_message,
            "host": self.host,
            "python_version": self.python_version,
            "platform_version": self.platform_version,
            "parameters": self.parameters,
            "stages": [stage.to_dict() for stage in self.stages],
            "metrics": self.metrics,
        }

    def summary_line(self) -> str:
        """One-line human summary used by the CLI and demo script."""
        return (
            f"[{self.status}] {self.pipeline_name} run={self.run_id} "
            f"read={self.records_read:,} written={self.records_written:,} "
            f"quarantined={self.records_quarantined:,} "
            f"duration={self.duration_seconds:.1f}s"
        )
