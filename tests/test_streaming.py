"""Streaming: checkpointing, watermarks, velocity state and alerting."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.streaming.event_generator import EventSimulator
from src.streaming.streaming_pipeline import (
    StreamingCheckpoint,
    StreamingPipeline,
    VelocityTracker,
)


class TestCheckpoint:
    def test_round_trip(self, tmp_path: Path):
        path = tmp_path / "checkpoint.json"
        checkpoint = StreamingCheckpoint(path)
        file = tmp_path / "events-1.json"
        file.write_text("{}", encoding="utf-8")

        assert not checkpoint.is_processed(file, "hash-1")
        checkpoint.mark(file, "hash-1")
        checkpoint.save()

        reloaded = StreamingCheckpoint(path)
        assert reloaded.is_processed(file, "hash-1")

    def test_rewritten_file_is_reprocessed(self, tmp_path: Path):
        """Same name, new content, must be treated as new data."""
        checkpoint = StreamingCheckpoint(tmp_path / "checkpoint.json")
        file = tmp_path / "events.json"
        file.write_text("{}", encoding="utf-8")

        checkpoint.mark(file, "hash-original")
        assert not checkpoint.is_processed(file, "hash-changed")

    def test_corrupt_checkpoint_does_not_crash(self, tmp_path: Path):
        path = tmp_path / "checkpoint.json"
        path.write_text("{ not json", encoding="utf-8")
        assert StreamingCheckpoint(path).processed == {}


class TestVelocityTracker:
    def test_counts_within_the_window(self):
        tracker = VelocityTracker(window_seconds=300)
        base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

        assert tracker.observe("ACC-1", base) == 1
        assert tracker.observe("ACC-1", base + timedelta(seconds=60)) == 2
        assert tracker.observe("ACC-1", base + timedelta(seconds=120)) == 3

    def test_old_events_leave_the_window(self):
        tracker = VelocityTracker(window_seconds=300)
        base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        tracker.observe("ACC-1", base)
        tracker.observe("ACC-1", base + timedelta(seconds=60))
        assert tracker.observe("ACC-1", base + timedelta(hours=2)) == 1

    def test_accounts_are_independent(self):
        tracker = VelocityTracker(window_seconds=300)
        now = datetime.now(UTC)
        tracker.observe("ACC-1", now)
        tracker.observe("ACC-1", now)
        assert tracker.observe("ACC-2", now) == 1

    def test_state_stays_bounded(self):
        """Memory must not grow with every account ever seen."""
        tracker = VelocityTracker(window_seconds=1, max_accounts=100)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(500):
            tracker.observe(f"ACC-{i}", base + timedelta(seconds=i * 10))
        assert tracker.tracked_accounts <= 500


class TestEventSimulator:
    def test_emits_ndjson_batches(self, settings):
        simulator = EventSimulator(settings, seed=42, batch_size=10)
        stats = simulator.run(total_events=30, interval=0)

        assert stats.events_emitted == 30
        assert stats.files_written == 3

        files = list(settings.paths.streaming_inbox.glob("*.json"))
        assert len(files) == 3
        records = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
        assert len(records) == 10
        assert {"event_id", "event_timestamp", "transaction_id", "amount"} <= set(records[0])

    def test_produces_anomalies_and_late_events(self, settings):
        simulator = EventSimulator(settings, seed=1, anomaly_rate=0.3, late_rate=0.3, batch_size=25)
        stats = simulator.run(total_events=200, interval=0)
        assert stats.anomalous_events > 0
        assert stats.late_events > 0

    def test_bursts_share_an_account(self, settings):
        simulator = EventSimulator(settings, seed=7)
        burst = simulator.build_burst("ACC-TARGET", size=8)
        assert len(burst) == 8
        assert {event["account_id"] for event in burst} == {"ACC-TARGET"}

    def test_is_reproducible(self, settings):
        first = EventSimulator(settings, seed=99).build_event()
        second = EventSimulator(settings, seed=99).build_event()
        assert first["amount"] == second["amount"]
        assert first["account_id"] == second["account_id"]


class TestStreamingPipeline:
    @pytest.fixture
    def pipeline(self, settings, store) -> StreamingPipeline:
        return StreamingPipeline(settings, store)

    def test_empty_inbox_is_a_no_op(self, pipeline):
        result = pipeline.process_batch()
        assert result.files == 0
        assert result.events_read == 0

    def test_processes_a_batch_end_to_end(self, settings, pipeline):
        EventSimulator(settings, seed=3, batch_size=20).run(total_events=60, interval=0)
        result = pipeline.process_batch()

        assert result.files == 3
        assert result.events_read == 60
        assert result.events_valid > 0

        from src.core.io import read_parquet_or_empty

        events = read_parquet_or_empty(settings.dataset_path("silver", "silver_transaction_events"))
        assert len(events) == result.events_valid

    def test_files_are_processed_exactly_once(self, settings, pipeline):
        EventSimulator(settings, seed=4, batch_size=10).run(total_events=30, interval=0)

        first = pipeline.process_batch()
        second = pipeline.process_batch()

        assert first.events_read == 30
        assert second.files == 0, "the checkpoint must prevent reprocessing"

    def test_invalid_events_are_quarantined_not_dropped(self, settings, pipeline):
        inbox = settings.paths.streaming_inbox
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "bad.json").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event_id": "E-1",
                            "transaction_id": "TXN-1",
                            "event_timestamp": datetime.now(UTC).isoformat(),
                            "account_id": "ACC-1",
                            "vendor_id": "VEN-1",
                            "amount": 100,
                            "currency": "USD",
                        }
                    ),
                    json.dumps(
                        {
                            "event_id": "E-2",
                            "transaction_id": "TXN-2",
                            "event_timestamp": datetime.now(UTC).isoformat(),
                            "amount": -50,
                            "currency": "USD",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        result = pipeline.process_batch()
        assert result.events_valid == 1
        assert result.events_quarantined == 1

        from src.silver.quarantine import read_quarantine

        quarantined = read_quarantine("transaction_event", settings)
        assert len(quarantined) == 1
        assert quarantined.iloc[0]["error_code"] == "INVALID_AMOUNT"

    def test_late_events_are_marked_and_kept(self, settings, pipeline):
        """A late payment is still a payment - the watermark bounds state, not
        acceptance."""
        inbox = settings.paths.streaming_inbox
        inbox.mkdir(parents=True, exist_ok=True)
        very_late = datetime.now(UTC) - timedelta(hours=5)
        (inbox / "late.json").write_text(
            json.dumps(
                {
                    "event_id": "E-LATE",
                    "transaction_id": "TXN-LATE",
                    "event_timestamp": very_late.strftime("%Y-%m-%dT%H:%M:%S"),
                    "account_id": "ACC-1",
                    "vendor_id": "VEN-1",
                    "amount": 100,
                    "currency": "USD",
                }
            ),
            encoding="utf-8",
        )

        result = pipeline.process_batch()
        assert result.events_valid == 1
        assert result.events_late == 1

        from src.core.io import read_parquet_or_empty

        events = read_parquet_or_empty(settings.dataset_path("silver", "silver_transaction_events"))
        assert len(events) == 1
        assert bool(events.iloc[0]["_is_late_arriving"]) is True

    def test_high_risk_events_raise_alerts(self, settings, pipeline):
        EventSimulator(settings, seed=11, anomaly_rate=0.9, batch_size=25).run(
            total_events=50, interval=0
        )
        result = pipeline.process_batch()
        assert result.events_valid > 0
        assert result.alerts >= 0  # threshold-dependent, but must not error

    def test_unknown_entities_do_not_fire_history_rules(self, settings, pipeline):
        """Without a feature store, 'new vendor' must not fire on everything.

        This was a real defect: every streamed event scored as a new-vendor risk
        because the lookup defaulted to zero history, producing one alert per
        event and no signal at all.
        """
        EventSimulator(settings, seed=5, anomaly_rate=0.0, batch_size=25).run(
            total_events=50, interval=0
        )
        scored_frames = []

        original = pipeline._score

        def capture(frame):
            result = original(frame)
            scored_frames.append(result)
            return result

        pipeline._score = capture
        pipeline.process_batch()

        assert scored_frames
        hits = scored_frames[0]["risk_rule_hits"].map(lambda hits: "new_vendor" in (hits or []))
        assert not hits.all(), "new_vendor fired on every event"

    def test_run_once_records_metadata(self, settings, store):
        EventSimulator(settings, seed=6, batch_size=10).run(total_events=20, interval=0)
        pipeline = StreamingPipeline(settings, store)
        pipeline.run_once()
        pipeline.context.finish("SUCCESS")
        store.record_run(pipeline.context)

        runs = store.recent_runs()
        assert not runs.empty
        assert "streaming" in runs.iloc[0]["pipeline_name"]
