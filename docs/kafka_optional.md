# Replacing the File Landing Zone with Kafka

The streaming pipeline reads NDJSON files from a landing directory. This
document shows exactly what changes if the transport becomes Kafka, and why it
was not the default.

## Why files are the default

A broker would add a service, a schema registry, a consumer-group story and a
local development story to a project whose subject is the **processing**, not
the transport.

The file-based landing zone is a real pattern, not a toy — cloud object-store
autoloading (Databricks Auto Loader, S3 event notifications) works exactly this
way. More importantly, it exercises every streaming concern that matters:

| Concern | How it appears here |
| --- | --- |
| Exactly-once processing | Content-hash checkpoint per file |
| Event time vs processing time | `event_timestamp` from the producer; `ingest_timestamp` stamped on arrival |
| Late arrivals | Watermark marks them; they are retained, never dropped |
| Bounded state | `VelocityTracker` prunes per-account windows each batch |
| Small files | Measured, and the reason partitioning is monthly |
| Backpressure | `micro_batch_max_files` drains a backlog steadily |

Swapping the transport does not change any of that. It changes where bytes come
from and who tracks the offset.

## What changes

One class: the discovery-and-commit boundary in
`src/streaming/streaming_pipeline.py`.

```
                        UNCHANGED
                        ---------
  discover  ->  parse  ->  validate  ->  Bronze  ->  Silver
                    ->  features  ->  model + rules  ->  alerts
  ^^^^^^^^                                  ^^^^^^
  REPLACED                                  UNCHANGED
```

### Today: file discovery

```python
def discover(self) -> list[Path]:
    candidates = sorted(self.inbox.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return [p for p in candidates
            if not self.checkpoint.is_processed(p, file_content_hash(p))]
```

### With Kafka: consumer poll

```python
from confluent_kafka import Consumer, TopicPartition

class KafkaSource:
    """Drop-in replacement for the file inbox.

    The checkpoint disappears: Kafka's consumer group owns the offset, which is
    what a broker is for. Offsets are committed only after the micro-batch has
    been written, which is what makes processing at-least-once with idempotent
    downstream writes - the same guarantee the file checkpoint provides.
    """

    def __init__(self, settings, topic: str = "transactions.events"):
        self.consumer = Consumer({
            "bootstrap.servers": settings.kafka.bootstrap_servers,
            "group.id": "financial-intelligence-streaming",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,          # commit after the write, not before
            "max.poll.interval.ms": 300_000,
        })
        self.consumer.subscribe([topic])

    def poll_batch(self, max_records: int = 500, timeout: float = 2.0) -> list[dict]:
        records = []
        while len(records) < max_records:
            message = self.consumer.poll(timeout)
            if message is None:
                break
            if message.error():
                log.warning("kafka_error", error=str(message.error()))
                continue
            records.append(json.loads(message.value()))
        return records

    def commit(self) -> None:
        self.consumer.commit(asynchronous=False)
```

And in the pipeline:

```python
- files = self.discover()
- raw_events = [e for path in files for e in _read_ndjson(path)]
+ raw_events = self.source.poll_batch(max_records=self.settings.streaming.batch_size)

  ...  # validate, score, write - all unchanged

- self._commit(files)
+ self.source.commit()
```

Everything between those two lines — contract validation, quarantine, Bronze and
Silver writes, watermarking, feature-store enrichment, model scoring, the rule
engine, alerting — is untouched.

## Ordering and partitioning

Key events by `account_id`:

```python
producer.produce(topic="transactions.events", key=account_id, value=payload)
```

Kafka guarantees ordering **within a partition**, so all events for one account
land in order on one consumer. That matters for the velocity window, which is
per-account: without keying, two consumers could each see half of a burst and
neither would flag it.

The file implementation sidesteps this by processing every file in one process.
At Kafka scale, keying is what preserves the same correctness.

## Schema registry

The Pydantic contracts in `src/core/contracts.py` already define the event
schema. With a registry, they would be generated from — or validated against —
an Avro or Protobuf schema:

```python
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer

deserializer = AvroDeserializer(
    SchemaRegistryClient({"url": settings.kafka.schema_registry_url}),
    schema_str=TRANSACTION_EVENT_AVRO,
)
```

This moves schema enforcement to the broker boundary. The platform's own
validation stays, because a valid Avro record can still carry a negative amount.

## Spark Structured Streaming

For a cluster deployment the whole loop is replaced:

```python
events = (spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", brokers)
    .option("subscribe", "transactions.events")
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), EVENT_SCHEMA).alias("e"))
    .select("e.*")
    .withWatermark("event_timestamp", "30 minutes"))

(events.writeStream
    .foreachBatch(lambda batch, epoch: process_micro_batch(batch, epoch))
    .option("checkpointLocation", checkpoint)
    .trigger(processingTime="30 seconds")
    .start())
```

`withWatermark` is Spark's version of the watermark this pipeline applies by
hand, and `foreachBatch` lets the existing per-batch logic run unchanged.

One behavioural difference worth knowing: Spark's watermark **drops** late
records from stateful aggregations, whereas this platform marks and retains
them. For financial data the retention behaviour is the correct one, so a Spark
implementation would write late records to a side output rather than relying on
the watermark to discard them.

## Local development

```yaml
# docker-compose.kafka.yml
services:
  kafka:
    image: confluentinc/cp-kafka:7.6.0
    environment:
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    ports: ["9092:9092"]
```

```bash
docker compose -f docker-compose.yml -f docker-compose.kafka.yml up -d kafka
```

The event generator would gain a `--sink kafka` flag; its event construction
logic is unchanged.

## Summary

| Aspect | Files (today) | Kafka |
| --- | --- | --- |
| Offset tracking | Content-hash checkpoint | Consumer group |
| Ordering | Single process | Per-partition, keyed by account |
| Backpressure | `micro_batch_max_files` | Consumer lag |
| Schema enforcement | Pydantic contracts | Registry + contracts |
| Local setup | Nothing | A broker |
| Code to change | — | One class |

The reason the file implementation is not a shortcut is that everything
downstream of the source boundary is transport-agnostic, and that is verifiable
by how small this diff is.
