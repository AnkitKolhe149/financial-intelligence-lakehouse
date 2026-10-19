"""Streaming transaction events.

Events wrap a transaction in an envelope carrying **event time** (when the
producer says it happened) separately from **ingest time** (when we received
it). Keeping the two apart is what makes late-arrival handling and watermarking
possible; a stream that only knows processing time cannot be correct.

A configurable share of events is emitted out of order - stamped with an event
time well before the events around them - because that is the normal condition
of any real queue, not an exceptional one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.core.hashing import deterministic_uuid
from src.data_generation.base import GeneratorContext

PRODUCERS = ("payments-gateway", "treasury-core", "grants-portal", "card-network")
EVENT_TYPES = (
    "TRANSACTION_CREATED",
    "TRANSACTION_CREATED",
    "TRANSACTION_CREATED",
    "TRANSACTION_UPDATED",
    "TRANSACTION_REVERSED",
)


def generate_events(
    ctx: GeneratorContext,
    count: int,
    accounts: pd.DataFrame,
    vendors: pd.DataFrame,
    locations: pd.DataFrame,
    anomaly_rate: float = 0.06,
) -> list[dict[str, Any]]:
    """Generate ``count`` streaming events as plain dictionaries.

    ``anomaly_rate`` controls how many events carry deliberately alarming
    characteristics (very large amounts, 03:00 timestamps, unknown vendors) so
    that running the simulator visibly produces alerts rather than a flat feed.
    """
    rng = ctx.rng
    account_ids = accounts["account_id"].to_numpy()
    vendor_ids = vendors["vendor_id"].to_numpy()
    location_ids = locations["location_id"].to_numpy()
    currencies = ctx.settings.generation.currencies

    now = datetime.now(UTC)
    events: list[dict[str, Any]] = []

    for index in range(count):
        account_id = str(account_ids[rng.integers(0, len(account_ids))])
        vendor_id = str(vendor_ids[rng.integers(0, len(vendor_ids))])
        amount = float(round(rng.lognormal(6.3, 1.2), 2))
        # Events arrive "now-ish": within the last hour of stream time.
        event_time = now - timedelta(seconds=int(rng.integers(0, 3600)))
        hour_override: int | None = None

        is_anomalous = bool(rng.random() < anomaly_rate)
        if is_anomalous:
            flavour = int(rng.integers(0, 4))
            if flavour == 0:
                amount = float(round(amount * rng.uniform(25, 90), 2))
            elif flavour == 1:
                hour_override = int(rng.integers(0, 4))
            elif flavour == 2:
                vendor_id = f"VEN-9{rng.integers(10000, 99999)}"  # unknown vendor
            else:
                amount = float(rng.choice([9500.0, 9900.0, 24500.0]))

        if hour_override is not None:
            event_time = event_time.replace(hour=hour_override, minute=int(rng.integers(0, 60)))

        # Out-of-order tail: stamp the event well in the past so the pipeline
        # sees it after events that logically follow it.
        if ctx.occurs(ctx.injection.out_of_order_event_rate if ctx.injection else 0.0):
            event_time = event_time - timedelta(minutes=int(rng.integers(20, 240)))

        transaction_id = f"TXN-S{index + 1:011d}"
        events.append(
            {
                "event_id": deterministic_uuid("event", transaction_id, index),
                "event_type": str(ctx.choice(EVENT_TYPES)),
                "event_timestamp": event_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ingest_timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                "producer": str(ctx.choice(PRODUCERS)),
                "schema_version": "1.0",
                "transaction_id": transaction_id,
                "account_id": account_id,
                "vendor_id": vendor_id,
                "amount": amount,
                "currency": str(np.array(currencies)[rng.integers(0, len(currencies))]),
                "location_id": str(location_ids[rng.integers(0, len(location_ids))]),
                "channel": str(ctx.choice(("ONLINE", "API", "MOBILE", "BATCH"))),
            }
        )

    return events
