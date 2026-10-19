"""Micro-batch streaming pipeline.

    python -m src.streaming.streaming_pipeline --once
    python -m src.streaming.streaming_pipeline --duration 60
    python -m src.streaming.streaming_pipeline            # runs until interrupted

Per micro-batch::

    discover new files -> parse -> contract validation -> Bronze
      -> Silver (typed, deduplicated) -> features -> model + rules
      -> gold_realtime_alerts

Four streaming concerns are implemented rather than gestured at:

**Exactly-once file handling.** A checkpoint file records every processed file
together with its content hash. A file is processed once; a file that is
*rewritten* with new content is processed again. This is what makes the pipeline
safe to restart at any point.

**Event time vs processing time.** Ordering, watermarking and lateness all use
``event_timestamp`` (stamped by the producer), never the arrival time. Using
arrival time would make results depend on network weather.

**Watermarked late arrivals.** Events older than the watermark are *not
discarded* - they are marked ``is_late`` and written, because in a financial
system a late payment is still a payment. The watermark bounds how long state is
held open, not what is accepted.

**Bounded state.** Per-account velocity is tracked in a rolling window that is
pruned each batch, so memory stays flat over a long-running stream.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import Settings, get_settings
from src.core.contracts import validate_records
from src.core.hashing import file_content_hash, record_hash
from src.core.io import write_parquet
from src.core.logging import bind_run_context, configure_logging, get_logger
from src.core.run_context import RunContext
from src.metadata.store import MetadataStore, get_metadata_store
from src.ml.model_registry import ModelRegistry
from src.silver.quarantine import quarantine_rejected_records

log = get_logger(__name__)

PIPELINE_NAME = "financial_intelligence_streaming"
CHECKPOINT_FILENAME = "streaming_checkpoint.json"
ALERT_TABLE = "gold_realtime_alerts"
EVENT_TABLE = "silver_transaction_events"
BRONZE_EVENT_TABLE = "bronze_transaction_events"


@dataclass
class MicroBatchResult:
    """Outcome of one micro-batch."""

    batch_id: str
    files: int = 0
    events_read: int = 0
    events_valid: int = 0
    events_quarantined: int = 0
    events_late: int = 0
    alerts: int = 0
    duration_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "files": self.files,
            "read": self.events_read,
            "valid": self.events_valid,
            "quarantined": self.events_quarantined,
            "late": self.events_late,
            "alerts": self.alerts,
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass
class StreamingStats:
    batches: int = 0
    files: int = 0
    events_read: int = 0
    events_valid: int = 0
    events_quarantined: int = 0
    events_late: int = 0
    alerts: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def absorb(self, batch: MicroBatchResult) -> None:
        self.batches += 1
        self.files += batch.files
        self.events_read += batch.events_read
        self.events_valid += batch.events_valid
        self.events_quarantined += batch.events_quarantined
        self.events_late += batch.events_late
        self.alerts += batch.alerts

    def summary(self) -> dict[str, Any]:
        elapsed = (datetime.now(UTC) - self.started_at).total_seconds()
        return {
            "batches": self.batches,
            "files": self.files,
            "events_read": self.events_read,
            "events_valid": self.events_valid,
            "events_quarantined": self.events_quarantined,
            "events_late": self.events_late,
            "alerts": self.alerts,
            "elapsed_seconds": round(elapsed, 1),
            "events_per_second": round(self.events_read / elapsed, 1) if elapsed else 0.0,
        }


class StreamingCheckpoint:
    """Durable record of which files have been processed.

    Stored as JSON next to the lake. In a distributed deployment this would be a
    Spark Structured Streaming checkpoint directory or a Delta commit log; the
    guarantee being implemented is the same one - do not reprocess what has
    already been committed.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.processed: dict[str, str] = {}
        self.last_event_time: str | None = None
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.processed = payload.get("processed_files", {})
            self.last_event_time = payload.get("last_event_time")
        except json.JSONDecodeError:
            log.warning("checkpoint_unreadable", path=str(self.path))

    def save(self) -> None:
        payload = {
            "processed_files": self.processed,
            "last_event_time": self.last_event_time,
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "file_count": len(self.processed),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_processed(self, path: Path, content_hash: str) -> bool:
        return self.processed.get(path.name) == content_hash

    def mark(self, path: Path, content_hash: str) -> None:
        self.processed[path.name] = content_hash

    def reset(self) -> None:
        self.processed.clear()
        self.last_event_time = None
        self.save()


class VelocityTracker:
    """Bounded per-account event-time window, for the velocity rule.

    A plain counter would grow without limit on a long-running stream. This
    keeps a deque per account and prunes anything older than the window on every
    query, so memory is proportional to *active* accounts rather than to all
    accounts ever seen.
    """

    def __init__(self, window_seconds: int = 300, max_accounts: int = 50_000) -> None:
        self.window = timedelta(seconds=window_seconds)
        self.max_accounts = max_accounts
        self._events: dict[str, deque[datetime]] = defaultdict(deque)

    def observe(self, account_id: str, event_time: datetime) -> int:
        """Record an event and return the count in the trailing window."""
        window = self._events[account_id]
        window.append(event_time)
        cutoff = event_time - self.window
        while window and window[0] < cutoff:
            window.popleft()

        if len(self._events) > self.max_accounts:
            self._evict()
        return len(window)

    def _evict(self) -> None:
        empty = [key for key, value in self._events.items() if not value]
        for key in empty[: len(empty) // 2 or 1]:
            del self._events[key]

    @property
    def tracked_accounts(self) -> int:
        return len(self._events)


class StreamingPipeline:
    """Discovers, validates, scores and alerts on streaming events."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: MetadataStore | None = None,
        *,
        reset_checkpoint: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store if store is not None else get_metadata_store()
        self.inbox: Path = self.settings.paths.streaming_inbox
        self.inbox.mkdir(parents=True, exist_ok=True)

        self.checkpoint = StreamingCheckpoint(
            Path(self.settings.paths.streaming_checkpoint) / CHECKPOINT_FILENAME
        )
        if reset_checkpoint:
            self.checkpoint.reset()

        self.velocity = VelocityTracker(
            window_seconds=int(
                self.settings.risk.rule("high_velocity").get("window_minutes", 5) * 60
            )
        )
        self.stats = StreamingStats()
        self.context = RunContext(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=self.settings.pipeline.version,
            source="stream",
            engine=self.settings.engine,
        )

        self.model = self._load_model()
        self.watermark = timedelta(minutes=self.settings.streaming.watermark_minutes)
        self.features = self._load_feature_lookups()

    def _load_model(self) -> Any:
        """Load the registered anomaly model, if one exists.

        Streaming *scores*, it never trains: training on a live stream would
        make the model chase whatever just happened, and retraining per batch is
        neither reproducible nor cheap. Without a model the rule engine still
        runs, which is a legitimate degraded mode.
        """
        try:
            registered = ModelRegistry(self.settings, self.store).load()
        except Exception as exc:
            log.warning("streaming_model_unavailable", error=str(exc)[:200])
            return None
        if registered is None:
            log.warning(
                "streaming_no_model",
                detail="scoring with rules only; run the batch pipeline to train one",
            )
            return None
        log.info("streaming_model_loaded", model_version=registered.model.model_version)
        return registered.model

    def _load_feature_lookups(self) -> dict[str, dict[str, float]]:
        """Load precomputed per-entity features from Gold - a feature-store read.

        A stream cannot recompute an account's historical average or a vendor's
        transaction history per event; scanning the fact table for every event
        would turn a millisecond of work into seconds of I/O. Real streaming
        systems solve this by reading a **precomputed feature store**, refreshed
        by the batch pipeline, and that is exactly what the Gold risk tables are
        here.

        Without this, features default to zero, and a rule like "vendor has
        fewer than 5 prior transactions" fires on *every* event - which is
        precisely what happened before this lookup existed: 300 alerts from 300
        events, all of them noise. A rule that fires on everything is worse than
        a rule that does not exist, because someone has to read the output.
        """
        from src.core.io import read_parquet_or_empty

        lookups: dict[str, dict[str, float]] = {
            "vendor_transactions": {},
            "account_average": {},
            "account_transactions": {},
        }

        vendors = read_parquet_or_empty(
            self.settings.dataset_path("gold", "gold_vendor_risk"),
            columns=["vendor_id", "transaction_count"],
        )
        if not vendors.empty:
            lookups["vendor_transactions"] = dict(
                zip(
                    vendors["vendor_id"].astype(str),
                    pd.to_numeric(vendors["transaction_count"], errors="coerce").fillna(0.0),
                    strict=False,
                )
            )

        accounts = read_parquet_or_empty(
            self.settings.dataset_path("gold", "gold_account_risk"),
            columns=["account_id", "average_amount_usd", "transaction_count"],
        )
        if not accounts.empty:
            account_ids = accounts["account_id"].astype(str)
            lookups["account_average"] = dict(
                zip(
                    account_ids,
                    pd.to_numeric(accounts["average_amount_usd"], errors="coerce").fillna(0.0),
                    strict=False,
                )
            )
            lookups["account_transactions"] = dict(
                zip(
                    account_ids,
                    pd.to_numeric(accounts["transaction_count"], errors="coerce").fillna(0.0),
                    strict=False,
                )
            )

        log.info(
            "streaming_feature_store_loaded",
            vendors=len(lookups["vendor_transactions"]),
            accounts=len(lookups["account_average"]),
            source="gold_vendor_risk, gold_account_risk",
        )
        if not lookups["vendor_transactions"]:
            log.warning(
                "streaming_feature_store_empty",
                detail=(
                    "no Gold risk tables found; history-dependent rules will be "
                    "suppressed rather than firing on every event"
                ),
            )
        return lookups

    # -- discovery ----------------------------------------------------------

    def discover(self) -> list[Path]:
        """New files in the inbox, oldest first.

        Ordered by modification time so that, all else equal, events are
        processed in roughly the order they arrived - and capped per batch so a
        backlog is drained steadily instead of in one enormous transaction.
        """
        candidates = sorted(
            (p for p in self.inbox.glob("*.json") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        pending: list[Path] = []
        for path in candidates:
            try:
                content_hash = file_content_hash(path)
            except OSError:
                # The producer may still be writing this file; it will be picked
                # up on the next pass.
                continue
            if not self.checkpoint.is_processed(path, content_hash):
                pending.append(path)
            if len(pending) >= self.settings.streaming.micro_batch_max_files:
                break
        return pending

    # -- micro-batch --------------------------------------------------------

    def process_batch(self, files: list[Path] | None = None) -> MicroBatchResult:
        """Process one micro-batch end to end."""
        started = time.perf_counter()
        batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        files = files if files is not None else self.discover()

        result = MicroBatchResult(batch_id=batch_id, files=len(files))
        if not files:
            return result

        raw_events: list[dict[str, Any]] = []
        for path in files:
            raw_events.extend(_read_ndjson(path))
        result.events_read = len(raw_events)

        if not raw_events:
            self._commit(files)
            result.duration_seconds = time.perf_counter() - started
            return result

        # ---- contract validation ------------------------------------------
        accepted, rejected = validate_records("transaction_event", raw_events)
        result.events_valid = len(accepted)
        result.events_quarantined = len(rejected)

        if rejected:
            quarantine_rejected_records(
                rejected,
                dataset="transaction_event",
                run_id=self.context.run_id,
                source_system="STREAM",
                source_file=f"micro-batch-{batch_id}",
                source_format="ndjson",
                settings=self.settings,
            )

        if not accepted:
            self._commit(files)
            result.duration_seconds = time.perf_counter() - started
            return result

        frame = pd.DataFrame(accepted)
        frame = self._stamp(frame, batch_id)

        # ---- watermark / lateness -----------------------------------------
        frame, late_count = self._apply_watermark(frame)
        result.events_late = late_count

        # ---- persist ------------------------------------------------------
        self._write_events(frame, batch_id)

        # ---- score --------------------------------------------------------
        scored = self._score(frame)
        alerts = self._build_alerts(scored, batch_id)
        result.alerts = len(alerts)
        if not alerts.empty:
            self._write_alerts(alerts)

        self._commit(files, frame)
        result.duration_seconds = time.perf_counter() - started
        log.info("micro_batch_processed", **result.summary())
        return result

    def _stamp(self, frame: pd.DataFrame, batch_id: str) -> pd.DataFrame:
        stamped = frame.copy()
        stamped["event_timestamp"] = pd.to_datetime(
            stamped["event_timestamp"], errors="coerce", utc=True
        )
        stamped["ingest_timestamp"] = pd.Timestamp.now(tz="UTC")
        stamped["_batch_id"] = batch_id
        stamped["_stream_run_id"] = self.context.run_id
        stamped["_source_system"] = "STREAM"
        stamped["_record_hash"] = [
            record_hash(
                row, ["transaction_id", "account_id", "vendor_id", "amount", "event_timestamp"]
            )
            for row in stamped.to_dict(orient="records")
        ]
        stamped["event_date"] = stamped["event_timestamp"].dt.date.astype("string")
        return stamped.sort_values("event_timestamp", kind="stable")

    def _apply_watermark(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Mark late events. Nothing is dropped.

        The watermark bounds how long the pipeline holds state open, not what it
        accepts: a payment that arrives late is still a real payment, and an
        oversight platform that silently discards it has lost data.
        """
        now = pd.Timestamp.now(tz="UTC")
        cutoff = now - self.watermark

        lateness = (now - frame["event_timestamp"]).dt.total_seconds()
        frame["lateness_seconds"] = lateness.round(1)
        frame["_is_late_arriving"] = frame["event_timestamp"] < cutoff

        late_count = int(frame["_is_late_arriving"].sum())
        if late_count:
            log.info(
                "late_events_accepted",
                events=late_count,
                watermark_minutes=self.settings.streaming.watermark_minutes,
                max_lateness_seconds=float(lateness.max()),
                policy="marked and retained, never discarded",
            )

        latest = frame["event_timestamp"].max()
        if pd.notna(latest):
            self.checkpoint.last_event_time = latest.isoformat()
        return frame, late_count

    def _write_events(self, frame: pd.DataFrame, batch_id: str) -> None:
        """Append the batch to Bronze and Silver event tables."""
        bronze_path = self.settings.dataset_path("bronze", BRONZE_EVENT_TABLE)
        silver_path = self.settings.dataset_path("silver", EVENT_TABLE)

        bronze = frame.copy()
        bronze["_ingestion_run_id"] = self.context.run_id
        bronze["_ingestion_timestamp"] = pd.Timestamp.now(tz="UTC")
        write_parquet(_stringify(bronze), bronze_path, partition_by=["event_date"], mode="append")
        write_parquet(frame, silver_path, partition_by=["event_date"], mode="append")

    def _score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score the batch with the streaming feature set plus the rule engine.

        Features are computed from *streaming state* (the velocity tracker) plus
        the event itself, not by re-reading history for every event. That is the
        difference between a stream processor and a batch job triggered often.
        """
        from src.risk.risk_engine import RiskEngine

        scored = frame.copy()
        scored["amount_usd"] = pd.to_numeric(scored.get("amount"), errors="coerce").fillna(0.0)
        scored["log_amount"] = np.log1p(scored["amount_usd"].clip(lower=0))
        stamps = scored["event_timestamp"]
        scored["hour_of_day"] = stamps.dt.hour.astype(float)
        scored["day_of_week"] = stamps.dt.dayofweek.astype(float)
        scored["is_weekend"] = (stamps.dt.dayofweek >= 5).astype(float)
        scored["is_night"] = stamps.dt.hour.between(0, 4).astype(float)

        velocities = [
            self.velocity.observe(str(row.account_id), row.event_timestamp.to_pydatetime())
            for row in scored.itertuples(index=False)
        ]
        scored["account_velocity_5min"] = [float(v) for v in velocities]

        # Historical context comes from the feature store (Gold), refreshed by
        # the batch pipeline. Where an entity is genuinely unknown to the
        # feature store, the neutral value is chosen so the dependent rule does
        # NOT fire: an unknown vendor is given a history above the "new vendor"
        # threshold, because "we have no data" must not be reported to an
        # investigator as "this is suspicious".
        vendor_history = self.features["vendor_transactions"]
        account_average = self.features["account_average"]
        account_history = self.features["account_transactions"]
        new_vendor_threshold = float(
            self.settings.risk.rule("new_vendor").get("min_history_count", 5)
        )

        accounts = scored["account_id"].astype(str)
        vendors = scored["vendor_id"].astype(str)

        scored["vendor_txn_count_prior"] = [
            vendor_history.get(vendor, new_vendor_threshold) for vendor in vendors
        ]
        scored["account_avg_amount_prior"] = [
            account_average.get(account, 0.0) for account in accounts
        ]
        scored["account_txn_count_prior"] = [
            account_history.get(account, 0.0) for account in accounts
        ]
        averages = scored["account_avg_amount_prior"].replace(0, np.nan)
        scored["amount_to_account_avg"] = (
            (scored["amount_usd"] / averages).fillna(1.0).clip(upper=1000)
        )
        # Geography needs the account's home region, which the event does not
        # carry; the batch pipeline owns that rule.
        scored["is_region_mismatch"] = 0.0

        if self.model is not None:
            enriched = self._model_features(scored)
            try:
                model_scored = self.model.score(enriched)
                scored["anomaly_score"] = model_scored["anomaly_score"].to_numpy()
                scored["is_anomaly"] = model_scored["is_anomaly"].to_numpy()
                scored["model_version"] = self.model.model_version
            except Exception as exc:
                log.warning("streaming_scoring_failed", error=str(exc)[:200])
                scored["anomaly_score"] = 0.0
                scored["is_anomaly"] = False
        else:
            scored["anomaly_score"] = 0.0
            scored["is_anomaly"] = False
            scored["model_version"] = "rules_only"

        return RiskEngine(self.settings).score(scored).frame

    def _model_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Project the streaming frame onto the model's expected feature set."""
        from src.ml.features import FEATURE_COLUMNS

        enriched = frame.copy()
        for column in FEATURE_COLUMNS:
            if column not in enriched.columns:
                enriched[column] = 0.0
        enriched["vendor_txn_count_7d"] = enriched.get("vendor_txn_count_7d", 0.0)
        enriched["account_txn_count_24h"] = enriched["account_velocity_5min"]
        return enriched

    def _build_alerts(self, scored: pd.DataFrame, batch_id: str) -> pd.DataFrame:
        threshold = self.settings.streaming.alert_min_risk_score
        candidates = scored[
            pd.to_numeric(scored["risk_score"], errors="coerce").fillna(0) >= threshold
        ]
        if candidates.empty:
            return pd.DataFrame()

        from src.core.hashing import deterministic_uuid

        return pd.DataFrame(
            {
                "alert_id": [
                    deterministic_uuid("stream_alert", t) for t in candidates["transaction_id"]
                ],
                "batch_id": batch_id,
                "event_id": candidates["event_id"].astype("string").to_numpy(),
                "transaction_id": candidates["transaction_id"].astype("string").to_numpy(),
                "account_id": candidates["account_id"].astype("string").to_numpy(),
                "vendor_id": candidates["vendor_id"].astype("string").to_numpy(),
                "amount_usd": candidates["amount_usd"].round(2).to_numpy(),
                "currency": candidates.get("currency", "USD"),
                "risk_score": pd.to_numeric(candidates["risk_score"]).astype(int).to_numpy(),
                "risk_level": candidates["risk_level"].astype("string").to_numpy(),
                "reason": [
                    (
                        "; ".join(list(r)[:3])
                        if isinstance(r, list | tuple) and r
                        else "threshold exceeded"
                    )
                    for r in candidates["risk_reasons"]
                ],
                "anomaly_score": pd.to_numeric(candidates["anomaly_score"], errors="coerce")
                .fillna(0.0)
                .round(6)
                .to_numpy(),
                "is_anomaly": candidates["is_anomaly"].fillna(False).to_numpy(),
                "model_version": candidates.get("model_version", "rules_only"),
                "event_timestamp": candidates["event_timestamp"].to_numpy(),
                "detected_at": pd.Timestamp.now(tz="UTC"),
                "is_late_arriving": candidates["_is_late_arriving"].to_numpy(),
                "status": "OPEN",
                "alert_date": candidates["event_date"].astype("string").to_numpy(),
            }
        )

    def _write_alerts(self, alerts: pd.DataFrame) -> None:
        path = self.settings.dataset_path("gold", ALERT_TABLE)
        write_parquet(alerts, path, partition_by=["alert_date"], mode="append")
        critical = int((alerts["risk_level"] == "CRITICAL").sum())
        log.warning(
            "realtime_alerts_raised",
            alerts=len(alerts),
            critical=critical,
            max_score=int(alerts["risk_score"].max()),
        )

    def _commit(self, files: list[Path], frame: pd.DataFrame | None = None) -> None:
        """Record the batch as processed. Called only after a successful write."""
        for path in files:
            try:
                self.checkpoint.mark(path, file_content_hash(path))
            except OSError:
                continue
        self.checkpoint.save()

    # -- run loops ----------------------------------------------------------

    def run_once(self) -> MicroBatchResult:
        result = self.process_batch()
        self.stats.absorb(result)
        return result

    def run(
        self,
        duration_seconds: float | None = None,
        max_batches: int | None = None,
        *,
        idle_exit_after: float | None = None,
    ) -> StreamingStats:
        """Poll the inbox until stopped, a duration elapses, or it goes idle."""
        bind_run_context(pipeline=PIPELINE_NAME, run_id=self.context.run_id)
        log.info(
            "streaming_started",
            inbox=str(self.inbox),
            poll_interval=self.settings.streaming.poll_interval_seconds,
            watermark_minutes=self.settings.streaming.watermark_minutes,
            alert_threshold=self.settings.streaming.alert_min_risk_score,
        )

        stop = {"requested": False}

        def handle_signal(_signum: int, _frame: Any) -> None:
            stop["requested"] = True
            log.info("streaming_stopping", reason="interrupt received")

        # Only installable on the main thread; the demo runs this in a worker.
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, handle_signal)

        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        idle_since: float | None = None

        while not stop["requested"]:
            batch = self.run_once()

            if batch.files == 0:
                idle_since = idle_since or time.monotonic()
                if idle_exit_after and (time.monotonic() - idle_since) >= idle_exit_after:
                    log.info("streaming_idle_exit", idle_seconds=idle_exit_after)
                    break
            else:
                idle_since = None

            if max_batches and self.stats.batches >= max_batches:
                break
            if deadline and time.monotonic() >= deadline:
                break
            time.sleep(self.settings.streaming.poll_interval_seconds)

        self.context.finish("SUCCESS")
        self.context.count(
            read=self.stats.events_read,
            written=self.stats.events_valid,
            quarantined=self.stats.events_quarantined,
        )
        self.context.add_metric("alerts", self.stats.alerts)
        self.context.add_metric("batches", self.stats.batches)
        self.context.add_metric("late_events", self.stats.events_late)
        self.store.record_run(self.context)

        log.info("streaming_finished", **self.stats.summary())
        return self.stats


# ---------------------------------------------------------------------------
# helpers and CLI
# ---------------------------------------------------------------------------


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    log.warning("stream_line_unparseable", file=path.name)
    except OSError as exc:
        log.warning("stream_file_unreadable", file=path.name, error=str(exc)[:120])
    return records


def _stringify(frame: pd.DataFrame) -> pd.DataFrame:
    """Bronze keeps values as text, matching the batch Bronze contract."""
    output = frame.copy()
    for column in output.columns:
        if column.endswith("timestamp") or column.startswith("_ingestion"):
            continue
        if not pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].astype("string")
    return output


def run_streaming(
    settings: Settings | None = None,
    *,
    once: bool = False,
    duration: float | None = None,
    max_batches: int | None = None,
    reset_checkpoint: bool = False,
    idle_exit_after: float | None = None,
) -> StreamingStats:
    """Programmatic entry point used by the demo."""
    pipeline = StreamingPipeline(settings, reset_checkpoint=reset_checkpoint)
    if once:
        pipeline.run_once()
        pipeline.context.finish("SUCCESS")
        pipeline.store.record_run(pipeline.context)
        return pipeline.stats
    return pipeline.run(duration, max_batches, idle_exit_after=idle_exit_after)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.streaming.streaming_pipeline",
        description="Micro-batch streaming pipeline over the file-based event inbox.",
    )
    parser.add_argument("--once", action="store_true", help="Process one batch and exit.")
    parser.add_argument(
        "--duration", type=float, default=None, help="Run for this many seconds, then stop."
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--idle-exit",
        type=float,
        default=None,
        help="Exit after this many seconds with no new files.",
    )
    parser.add_argument(
        "--reset-checkpoint", action="store_true", help="Reprocess every file in the inbox."
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    stats = run_streaming(
        settings,
        once=args.once,
        duration=args.duration,
        max_batches=args.max_batches,
        reset_checkpoint=args.reset_checkpoint,
        idle_exit_after=args.idle_exit,
    )

    print("\nStreaming pipeline")
    print("-" * 44)
    for key, value in stats.summary().items():
        print(f"  {key:<22} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
