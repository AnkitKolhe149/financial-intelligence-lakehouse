"""Streaming: continuous ingestion and near-real-time scoring.

The local implementation is **file-based**: a producer drops NDJSON files into a
landing directory and the pipeline picks them up in micro-batches. That is a
deliberate choice, not a shortcut.

* It is a real streaming pattern. Cloud object-store landing zones (S3/ADLS +
  autoloader-style discovery) work exactly this way, and the concerns that
  matter - checkpointing, exactly-once file handling, watermarks, late arrivals,
  small files - are all present.
* It requires no broker. Kafka would add a service, a schema registry and a
  consumer-group story to a demo whose point is the *processing*, not the
  transport. ``docs/kafka_optional.md`` shows exactly which module changes if
  you swap the transport for Kafka - it is one class.

What is genuinely modelled here: event time versus processing time, watermarked
late-arrival handling, idempotent file processing via a checkpoint, and
alerting on the scored output.
"""

from src.streaming.event_generator import EventSimulator, generate_stream
from src.streaming.streaming_pipeline import StreamingPipeline, run_streaming

__all__ = ["EventSimulator", "StreamingPipeline", "generate_stream", "run_streaming"]
