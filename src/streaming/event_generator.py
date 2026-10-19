"""Streaming event simulator.

    python -m src.streaming.event_generator --events 1000 --interval 0.25
    python -m src.streaming.event_generator --events 200 --anomaly-rate 0.15
    python -m src.streaming.event_generator --burst-account ACC-0000123

Writes NDJSON micro-batches into the streaming inbox at a controlled rate, so
the streaming pipeline has something to consume while you watch it.

The generator deliberately emits events that *should* trigger alerts - very
large amounts, 03:00 timestamps, unknown vendors, velocity bursts - because a
streaming demo where nothing ever fires demonstrates nothing. It also emits
out-of-order events, so the watermark logic is exercised rather than assumed.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import Settings, get_settings
from src.core.hashing import deterministic_uuid
from src.core.io import read_parquet_or_empty
from src.core.logging import configure_logging, get_logger
from src.data_generation.writers import write_ndjson

log = get_logger(__name__)

PRODUCERS = ("payments-gateway", "treasury-core", "grants-portal", "card-network")


@dataclass
class SimulatorStats:
    events_emitted: int = 0
    files_written: int = 0
    anomalous_events: int = 0
    late_events: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def summary(self) -> dict[str, Any]:
        elapsed = (datetime.now(UTC) - self.started_at).total_seconds()
        return {
            "events": self.events_emitted,
            "files": self.files_written,
            "anomalous": self.anomalous_events,
            "late": self.late_events,
            "elapsed_seconds": round(elapsed, 1),
            "events_per_second": round(self.events_emitted / elapsed, 2) if elapsed else 0.0,
        }


class EventSimulator:
    """Emits realistic transaction events into the streaming inbox."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        seed: int | None = None,
        anomaly_rate: float = 0.08,
        late_rate: float = 0.05,
        batch_size: int = 20,
    ) -> None:
        self.settings = settings or get_settings()
        self.rng = np.random.default_rng(
            seed if seed is not None else self.settings.generation.seed
        )
        self.anomaly_rate = anomaly_rate
        self.late_rate = late_rate
        self.batch_size = batch_size
        self.inbox: Path = self.settings.paths.streaming_inbox
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.stats = SimulatorStats()
        self._sequence = 0

        self.accounts, self.vendors, self.locations = self._load_reference()

    def _load_reference(self) -> tuple[list[str], list[str], list[str]]:
        """Use real ids from the lake so events join to existing dimensions.

        Falling back to synthetic ids when the lake is empty is intentional: it
        means every one of them looks like an unknown account/vendor downstream,
        which is itself a legitimate scenario to demonstrate.
        """
        accounts = read_parquet_or_empty(
            self.settings.dataset_path("silver", "silver_accounts"), columns=["account_id"]
        )
        vendors = read_parquet_or_empty(
            self.settings.dataset_path("silver", "silver_vendors"), columns=["vendor_id"]
        )
        locations = read_parquet_or_empty(
            self.settings.dataset_path("silver", "silver_locations"), columns=["location_id"]
        )

        account_ids = (
            accounts["account_id"].astype(str).tolist()
            if not accounts.empty
            else [f"ACC-{i:07d}" for i in range(1, 51)]
        )
        vendor_ids = (
            vendors["vendor_id"].astype(str).tolist()
            if not vendors.empty
            else [f"VEN-{i:06d}" for i in range(1, 31)]
        )
        location_ids = (
            locations["location_id"].astype(str).tolist()
            if not locations.empty
            else [f"LOC-{i:04d}" for i in range(1, 11)]
        )

        if accounts.empty:
            log.warning(
                "streaming_reference_missing",
                detail="lake not built; events will reference unknown ids",
            )
        return account_ids, vendor_ids, location_ids

    # -- event construction --------------------------------------------------

    def build_event(self, *, force_account: str | None = None) -> dict[str, Any]:
        self._sequence += 1
        now = datetime.now(UTC)

        account_id = force_account or str(
            self.accounts[int(self.rng.integers(0, len(self.accounts)))]
        )
        vendor_id = str(self.vendors[int(self.rng.integers(0, len(self.vendors)))])
        amount = float(round(self.rng.lognormal(6.3, 1.1), 2))
        event_time = now
        is_anomalous = False

        if self.rng.random() < self.anomaly_rate:
            is_anomalous = True
            flavour = int(self.rng.integers(0, 4))
            if flavour == 0:
                amount = float(round(amount * self.rng.uniform(25, 90), 2))
            elif flavour == 1:
                event_time = now.replace(hour=int(self.rng.integers(0, 4)))
            elif flavour == 2:
                vendor_id = f"VEN-9{self.rng.integers(10000, 99999)}"
            else:
                amount = float(self.rng.choice([9500.0, 9800.0, 9900.0, 24500.0]))

        is_late = self.rng.random() < self.late_rate
        if is_late:
            # Event time well in the past: the record arrives after events that
            # logically follow it, which is what the watermark exists to handle.
            event_time = event_time - timedelta(minutes=int(self.rng.integers(15, 180)))

        transaction_id = f"TXN-S{self._sequence:011d}"
        currencies = self.settings.generation.currencies

        if is_anomalous:
            self.stats.anomalous_events += 1
        if is_late:
            self.stats.late_events += 1

        return {
            "event_id": deterministic_uuid("stream", transaction_id, self._sequence),
            "event_type": "TRANSACTION_CREATED",
            "event_timestamp": event_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ingest_timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "producer": str(PRODUCERS[int(self.rng.integers(0, len(PRODUCERS)))]),
            "schema_version": "1.0",
            "transaction_id": transaction_id,
            "account_id": account_id,
            "vendor_id": vendor_id,
            "amount": amount,
            "currency": str(currencies[int(self.rng.integers(0, len(currencies)))]),
            "location_id": str(self.locations[int(self.rng.integers(0, len(self.locations)))]),
            "channel": str(["ONLINE", "API", "MOBILE", "BATCH"][int(self.rng.integers(0, 4))]),
        }

    def build_burst(self, account_id: str, size: int = 12) -> list[dict[str, Any]]:
        """A velocity burst: many events from one account within a minute."""
        return [self.build_event(force_account=account_id) for _ in range(size)]

    # -- emission ------------------------------------------------------------

    def emit(self, events: list[dict[str, Any]]) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        path = self.inbox / f"events-{stamp}.json"
        write_ndjson(events, path)
        self.stats.events_emitted += len(events)
        self.stats.files_written += 1
        return path

    def run(
        self,
        total_events: int = 1000,
        interval: float = 0.25,
        *,
        burst_account: str | None = None,
        on_batch: Any = None,
    ) -> SimulatorStats:
        """Emit ``total_events`` in batches, pausing ``interval`` between them."""
        log.info(
            "simulator_started",
            events=total_events,
            interval_seconds=interval,
            batch_size=self.batch_size,
            inbox=str(self.inbox),
        )

        stop = {"requested": False}

        def handle_signal(_signum: int, _frame: Any) -> None:
            stop["requested"] = True
            log.info("simulator_stopping", reason="interrupt received")

        # Signal handlers can only be installed on the main thread; when the
        # simulator runs inside the demo script this is not available.
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, handle_signal)

        emitted = 0
        while emitted < total_events and not stop["requested"]:
            size = min(self.batch_size, total_events - emitted)
            events = self.build_event_batch(size, burst_account)
            path = self.emit(events)
            emitted += len(events)

            if on_batch is not None:
                on_batch(path, events)

            log.info(
                "batch_emitted",
                file=path.name,
                events=len(events),
                total=emitted,
                remaining=total_events - emitted,
            )
            if emitted < total_events and interval > 0:
                time.sleep(interval)

        log.info("simulator_finished", **self.stats.summary())
        return self.stats

    def build_event_batch(self, size: int, burst_account: str | None) -> list[dict[str, Any]]:
        # Roughly one batch in eight is a burst from a single account, which is
        # what makes the velocity rule fire on the live stream.
        if burst_account and self.rng.random() < 0.3:
            return self.build_burst(burst_account, size)
        if self.rng.random() < 0.12 and self.accounts:
            victim = str(self.accounts[int(self.rng.integers(0, len(self.accounts)))])
            return self.build_burst(victim, size)
        return [self.build_event() for _ in range(size)]


def generate_stream(
    events: int = 1000,
    interval: float = 0.25,
    settings: Settings | None = None,
    **kwargs: Any,
) -> SimulatorStats:
    """Convenience wrapper used by the demo script."""
    simulator = EventSimulator(settings, **kwargs)
    return simulator.run(events, interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.streaming.event_generator",
        description="Emit simulated transaction events into the streaming inbox.",
    )
    parser.add_argument("--events", type=int, default=1000, help="Total events to emit.")
    parser.add_argument(
        "--interval", type=float, default=0.25, help="Seconds to pause between batches."
    )
    parser.add_argument("--batch-size", type=int, default=20, help="Events per file.")
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.08,
        help="Share of events with alarming characteristics.",
    )
    parser.add_argument(
        "--late-rate",
        type=float,
        default=0.05,
        help="Share of events stamped in the past (out of order).",
    )
    parser.add_argument(
        "--burst-account",
        type=str,
        default=None,
        help="Concentrate velocity bursts on one account id.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--clean", action="store_true", help="Empty the inbox before starting.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    if args.clean:
        inbox = settings.paths.streaming_inbox
        removed = 0
        for path in inbox.glob("*.json"):
            path.unlink()
            removed += 1
        print(f"Removed {removed} file(s) from {inbox}")

    simulator = EventSimulator(
        settings,
        seed=args.seed,
        anomaly_rate=args.anomaly_rate,
        late_rate=args.late_rate,
        batch_size=args.batch_size,
    )
    stats = simulator.run(args.events, args.interval, burst_account=args.burst_account)

    print("\nStreaming simulator")
    print("-" * 40)
    for key, value in stats.summary().items():
        print(f"  {key:<20} {value}")
    print(f"\nInbox: {settings.paths.streaming_inbox}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
