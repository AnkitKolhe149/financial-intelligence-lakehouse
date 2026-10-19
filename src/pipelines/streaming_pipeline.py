"""CLI entry point for the streaming pipeline.

    python -m src.pipelines.streaming_pipeline --once
    python -m src.pipelines.streaming_pipeline --duration 60

The implementation lives in :mod:`src.streaming.streaming_pipeline`, alongside
the event simulator and the checkpoint. This module exists so that both
pipelines are launchable from the same package - ``src.pipelines.*`` - which is
where anyone looking for "how do I run this thing" will look first.
"""

from __future__ import annotations

import sys

from src.streaming.streaming_pipeline import (
    MicroBatchResult,
    StreamingPipeline,
    StreamingStats,
    main,
    run_streaming,
)

__all__ = [
    "MicroBatchResult",
    "StreamingPipeline",
    "StreamingStats",
    "main",
    "run_streaming",
]


if __name__ == "__main__":
    sys.exit(main())
