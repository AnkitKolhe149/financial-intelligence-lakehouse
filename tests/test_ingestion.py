"""Multi-format ingestion and the Bronze layer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.bronze.loader import BronzeLoader
from src.core.config import Settings
from src.core.io import read_parquet
from src.ingestion.base import _to_snake, discover_files, stamp_audit_columns
from src.ingestion.csv_ingestion import CsvSourceReader
from src.ingestion.excel_ingestion import ExcelSourceReader, detect_header_row
from src.ingestion.json_ingestion import JsonSourceReader
from src.ingestion.xml_ingestion import XmlSourceReader


class TestAuditStamping:
    def test_every_audit_column_is_attached(self, clean_transactions, run_context):
        stamped = stamp_audit_columns(
            clean_transactions,
            dataset="transaction",
            source_system="S",
            source_format="csv",
            source_file="f.csv",
            run_id=run_context.run_id,
        )
        for column in (
            "_source_system",
            "_source_file",
            "_source_format",
            "_ingestion_timestamp",
            "_ingestion_run_id",
            "_record_hash",
            "_schema_version",
            "_source_row_number",
            "ingestion_date",
        ):
            assert column in stamped.columns

    def test_record_hash_is_content_based_not_identifier_based(
        self, clean_transactions, run_context
    ):
        """The same business content under a different id must hash the same."""
        original = stamp_audit_columns(
            clean_transactions,
            dataset="transaction",
            source_system="S",
            source_format="csv",
            source_file="f.csv",
            run_id=run_context.run_id,
        )
        renamed = clean_transactions.copy()
        renamed["transaction_id"] = "TXN-DIFFERENT-" + renamed["transaction_id"]
        reissued = stamp_audit_columns(
            renamed,
            dataset="transaction",
            source_system="S",
            source_format="csv",
            source_file="f.csv",
            run_id=run_context.run_id,
        )
        assert list(original["_record_hash"]) == list(reissued["_record_hash"])

    def test_hash_is_stable_across_calls(self, clean_transactions, run_context):
        first = stamp_audit_columns(
            clean_transactions,
            dataset="transaction",
            source_system="S",
            source_format="csv",
            source_file="f.csv",
            run_id=run_context.run_id,
        )
        second = stamp_audit_columns(
            clean_transactions,
            dataset="transaction",
            source_system="OTHER",
            source_format="json",
            source_file="g.json",
            run_id="different-run",
        )
        # Provenance differs, content does not - so the content hash must match.
        assert list(first["_record_hash"]) == list(second["_record_hash"])

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("contractValueUsd", "contract_value_usd"),
            ("Contract Value", "contract_value"),
            ("vendor_id", "vendor_id"),
            ("AwardID", "award_id"),
        ],
    )
    def test_name_normalisation(self, source, expected):
        assert _to_snake(source) == expected


class TestCsvIngestion:
    def test_reads_and_preserves_source_text(self, settings: Settings, tmp_path: Path, run_context):
        path = settings.paths.raw / "csv" / "transactions_test.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "transaction_id,account_id,vendor_id,amount,currency,transaction_timestamp\n"
            "TXN-1,ACC-1,VEN-1,100.50,USD,2026-01-01T10:00:00\n"
            "TXN-2,ACC-2,VEN-2,-50,not-a-currency,not-a-date\n",
            encoding="utf-8",
        )

        reader = CsvSourceReader(
            "transaction",
            "TEST",
            run_id=run_context.run_id,
            directory=path.parent,
            pattern="transactions_test.csv",
        )
        results = reader.read()
        assert len(results) == 1
        frame = results[0].frame
        assert len(frame) == 2
        # Bronze keeps the invalid values verbatim; judging them is Silver's job.
        assert frame.loc[1, "amount"] == "-50"
        assert frame.loc[1, "transaction_timestamp"] == "not-a-date"

    def test_schema_version_is_inferred_from_columns(self, settings: Settings, run_context):
        directory = settings.paths.raw / "csv"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "v10.csv").write_text(
            "transaction_id,account_id,vendor_id,amount,currency,transaction_timestamp\n"
            "TXN-1,ACC-1,VEN-1,10,USD,2026-01-01T00:00:00\n",
            encoding="utf-8",
        )
        (directory / "v11.csv").write_text(
            "transaction_id,account_id,vendor_id,amount,currency,transaction_timestamp,merchant_category\n"
            "TXN-2,ACC-1,VEN-1,10,USD,2026-01-01T00:00:00,IT_SOFTWARE\n",
            encoding="utf-8",
        )

        reader = CsvSourceReader(
            "transaction", "TEST", run_id=run_context.run_id, directory=directory, pattern="v1*.csv"
        )
        versions = {r.source_file: r.schema_version for r in reader.read()}
        assert versions["v10.csv"] == "1.0"
        assert versions["v11.csv"] == "1.1"

    def test_a_broken_file_does_not_stop_the_others(self, settings: Settings, run_context):
        directory = settings.paths.raw / "csv"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "good.csv").write_text(
            "transaction_id,account_id,vendor_id,amount,currency,transaction_timestamp\n"
            "TXN-1,ACC-1,VEN-1,10,USD,2026-01-01T00:00:00\n",
            encoding="utf-8",
        )
        # Binary content that pandas cannot decode as UTF-8.
        (directory / "broken.csv").write_bytes(b"\xff\xfe\x00\x00bad")

        reader = CsvSourceReader(
            "transaction", "TEST", run_id=run_context.run_id, directory=directory, pattern="*.csv"
        )
        results = reader.read()
        assert len(results) == 2
        assert any(r.record_count == 1 for r in results)
        assert any(r.errors for r in results)


class TestJsonIngestion:
    def test_reads_enveloped_document(self, settings: Settings, run_context):
        path = settings.paths.raw / "json" / "accounts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "metadata": {"source_system": "TREASURY", "record_count": 2},
                    "accounts": [
                        {"account_id": "ACC-1", "department_id": "DEP-1"},
                        {"account_id": "ACC-2", "department_id": "DEP-1"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        reader = JsonSourceReader(
            "account", "FALLBACK", run_id=run_context.run_id, files=[path], record_key="accounts"
        )
        result = reader.read()[0]
        assert result.record_count == 2
        # The envelope's declared source system wins over the caller's guess.
        assert result.source_system == "TREASURY"

    def test_record_count_mismatch_is_reported(self, settings: Settings, run_context):
        path = settings.paths.raw / "json" / "truncated.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"metadata": {"record_count": 5}, "accounts": [{"account_id": "ACC-1"}]}),
            encoding="utf-8",
        )
        reader = JsonSourceReader(
            "account", "TEST", run_id=run_context.run_id, files=[path], record_key="accounts"
        )
        result = reader.read()[0]
        assert any("declared 5" in error for error in result.errors)

    def test_bare_array_is_supported(self, settings: Settings, run_context):
        path = settings.paths.raw / "json" / "bare.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([{"account_id": "ACC-1", "department_id": "D"}]), encoding="utf-8"
        )
        reader = JsonSourceReader("account", "TEST", run_id=run_context.run_id, files=[path])
        assert reader.read()[0].record_count == 1

    def test_ndjson_skips_bad_lines(self, settings: Settings, run_context):
        path = settings.paths.raw / "json" / "events.ndjson"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"account_id": "ACC-1", "department_id": "D"}\n'
            "{ this is not json }\n"
            '{"account_id": "ACC-2", "department_id": "D"}\n',
            encoding="utf-8",
        )
        reader = JsonSourceReader(
            "account", "TEST", run_id=run_context.run_id, files=[path], ndjson=True
        )
        result = reader.read()[0]
        assert result.record_count == 2
        assert len(result.errors) == 1


class TestXmlIngestion:
    def test_attributes_and_camelcase_elements(self, settings: Settings, run_context):
        path = settings.paths.raw / "xml" / "contracts.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            """<?xml version="1.0" encoding="utf-8"?>
            <contractAwards sourceSystem="PROCUREMENT" recordCount="1" schemaVersion="1.0">
              <award id="CON-0000001">
                <vendorId>VEN-000001</vendorId>
                <agencyId>AGY-0001</agencyId>
                <contractValueUsd>150000.00</contractValueUsd>
                <competitionType>SOLE_SOURCE</competitionType>
              </award>
            </contractAwards>""",
            encoding="utf-8",
        )
        reader = XmlSourceReader(
            "contract",
            "TEST",
            run_id=run_context.run_id,
            files=[path],
            record_tag="award",
            id_column="contract_id",
        )
        frame = reader.read()[0].frame
        assert frame.loc[0, "contract_id"] == "CON-0000001"  # from the attribute
        assert frame.loc[0, "vendor_id"] == "VEN-000001"  # camelCase -> snake_case
        assert frame.loc[0, "contract_value_usd"] == "150000.00"

    def test_entity_expansion_is_refused(self, settings: Settings, run_context):
        """defusedxml must reject the billion-laughs attack."""
        path = settings.paths.raw / "xml" / "evil.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            """<?xml version="1.0"?>
            <!DOCTYPE lolz [
              <!ENTITY lol "lol">
              <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
            ]>
            <contractAwards><award id="X"><vendorId>&lol2;</vendorId></award></contractAwards>""",
            encoding="utf-8",
        )
        reader = XmlSourceReader(
            "contract", "TEST", run_id=run_context.run_id, files=[path], record_tag="award"
        )
        result = reader.read()[0]
        # Either the parser refuses the document or it yields nothing - both are
        # acceptable; silently expanding the entity is not.
        assert result.errors or result.record_count == 0


class TestExcelIngestion:
    def test_detects_header_below_a_title_row(self, settings: Settings, run_context):
        path = settings.paths.raw / "excel" / "budgets.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {
                "budget_id": ["BUD-1", "BUD-2"],
                "fiscal_year": [2026, 2026],
                "agency_id": ["AGY-0001", "AGY-0002"],
                "allocated_amount_usd": [1000.0, 2000.0],
            }
        )
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="budgets", index=False, startrow=2)
            writer.sheets["budgets"].cell(row=1, column=1).value = "BUDGETS EXTRACT - 2026"

        assert detect_header_row(path, "budgets") == 2

        reader = ExcelSourceReader(
            "budget", "FINANCE", run_id=run_context.run_id, files=[path], sheet_name="budgets"
        )
        result = reader.read()[0]
        assert result.record_count == 2
        assert "budget_id" in result.frame.columns


class TestBronzeLoader:
    def test_writes_partitioned_bronze(self, settings, store, run_context, bronze_transactions):
        from src.ingestion.base import IngestionResult

        loader = BronzeLoader(run_context.run_id, settings, store)
        result = loader.load(
            "transaction",
            [
                IngestionResult(
                    dataset="transaction",
                    source_system="TEST",
                    source_format="csv",
                    source_file="test.csv",
                    frame=bronze_transactions,
                )
            ],
        )
        assert result.records_written == 10
        assert result.table == "bronze_transactions"

        stored = read_parquet(settings.dataset_path("bronze", "bronze_transactions"))
        assert len(stored) == 10
        assert "_record_hash" in stored.columns

    def test_reingesting_identical_content_is_a_no_op(
        self, settings, store, run_context, bronze_transactions
    ):
        from src.ingestion.base import IngestionResult

        payload = [
            IngestionResult(
                dataset="transaction",
                source_system="TEST",
                source_format="csv",
                source_file="idempotent.csv",
                frame=bronze_transactions,
            )
        ]
        first = BronzeLoader(run_context.run_id, settings, store).load("transaction", payload)
        second = BronzeLoader("second-run", settings, store).load("transaction", payload)

        assert first.records_written == 10
        assert second.records_written == 0
        assert second.records_skipped == 10

    def test_force_reingests(self, settings, store, run_context, bronze_transactions):
        from src.ingestion.base import IngestionResult

        payload = [
            IngestionResult(
                dataset="transaction",
                source_system="TEST",
                source_format="csv",
                source_file="forced.csv",
                frame=bronze_transactions,
            )
        ]
        BronzeLoader(run_context.run_id, settings, store).load("transaction", payload)
        forced = BronzeLoader("run-2", settings, store, force=True).load("transaction", payload)
        assert forced.records_written == 10

    def test_schema_evolution_unions_columns(
        self, settings, store, run_context, clean_transactions
    ):
        """A v1.0 file and a v1.1 file must land in one table, nulls back-filled."""
        from src.ingestion.base import IngestionResult

        v10 = clean_transactions.drop(columns=["merchant_category"]).head(4)
        v11 = clean_transactions.tail(6)

        results = [
            IngestionResult(
                dataset="transaction",
                source_system="TEST",
                source_format="csv",
                source_file="v10.csv",
                frame=stamp_audit_columns(
                    v10,
                    dataset="transaction",
                    source_system="TEST",
                    source_format="csv",
                    source_file="v10.csv",
                    run_id=run_context.run_id,
                    schema_version="1.0",
                ),
                schema_version="1.0",
            ),
            IngestionResult(
                dataset="transaction",
                source_system="TEST",
                source_format="csv",
                source_file="v11.csv",
                frame=stamp_audit_columns(
                    v11,
                    dataset="transaction",
                    source_system="TEST",
                    source_format="csv",
                    source_file="v11.csv",
                    run_id=run_context.run_id,
                    schema_version="1.1",
                ),
                schema_version="1.1",
            ),
        ]

        loader = BronzeLoader(run_context.run_id, settings, store)
        outcome = loader.load("transaction", results)
        assert outcome.records_written == 10
        assert sorted(set(outcome.schema_versions)) == ["1.0", "1.1"]

        stored = read_parquet(settings.dataset_path("bronze", "bronze_transactions"))
        assert "merchant_category" in stored.columns
        assert stored["merchant_category"].isna().sum() == 4

    def test_ledger_records_the_load(self, settings, store, run_context, bronze_transactions):
        from src.ingestion.base import IngestionResult

        BronzeLoader(run_context.run_id, settings, store).load(
            "transaction",
            [
                IngestionResult(
                    dataset="transaction",
                    source_system="TEST",
                    source_format="csv",
                    source_file="ledger.csv",
                    frame=bronze_transactions,
                )
            ],
        )
        history = store.ingestion_history()
        assert not history.empty
        assert "ledger.csv" in set(history["source_file"])


class TestDiscovery:
    def test_files_are_returned_in_a_stable_order(self, tmp_path: Path):
        for name in ("c.csv", "a.csv", "b.csv"):
            (tmp_path / name).write_text("x", encoding="utf-8")
        assert [p.name for p in discover_files(tmp_path, "*.csv")] == ["a.csv", "b.csv", "c.csv"]

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        assert discover_files(tmp_path / "nope", "*.csv") == []
