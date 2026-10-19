"""Anomalous transactions and the review queue.

Produces two datasets:

``gold_anomaly_transactions``
    Every transaction the platform considers suspicious, with the full
    explanation attached. This is the investigator's working table.

``gold_risk_alerts``
    The subset that warrants review, shaped as a **case queue** with a
    lifecycle status. Alert ids are deterministic (UUIDv5 over the transaction
    and rule set), so re-running the pipeline updates the existing alert rather
    than creating a duplicate case - which is what would happen with a random
    id, and is the fastest way to make a review queue useless.
"""

from __future__ import annotations

import json

import pandas as pd

from src.core.config import Settings
from src.core.hashing import deterministic_uuid
from src.core.logging import get_logger
from src.gold.base import GoldResult, write_gold
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore

log = get_logger(__name__)

ANOMALY_TABLE = "gold_anomaly_transactions"
ALERT_TABLE = "gold_risk_alerts"

ANOMALY_COLUMNS = [
    "transaction_id",
    "account_id",
    "vendor_id",
    "contract_id",
    "department_id",
    "agency_id",
    "amount",
    "currency",
    "amount_usd",
    "transaction_timestamp",
    "transaction_date",
    "transaction_month",
    "location_id",
    "merchant_category",
    "payment_method",
    "anomaly_score",
    "is_anomaly",
    "risk_score",
    "risk_level",
    "risk_reasons",
    "risk_rule_hits",
    "model_version",
    "scored_at",
]

ALERT_STATUSES = ("OPEN", "REVIEWING", "RESOLVED", "FALSE_POSITIVE")


def build_anomaly_transactions(
    scored: pd.DataFrame,
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    """Publish suspicious transactions with their explanations."""
    if scored.empty:
        return write_gold(
            pd.DataFrame(columns=ANOMALY_COLUMNS),
            ANOMALY_TABLE,
            settings,
            store=store,
            lineage=lineage,
            run_id=run_id,
            description="Transactions flagged by the model or the risk rules.",
        )

    is_anomaly = (
        scored.get("is_anomaly", pd.Series(False, index=scored.index)).fillna(False).astype(bool)
    )
    risk_level = scored.get("risk_level", pd.Series("LOW", index=scored.index)).astype("string")
    # Union, not intersection: a transaction is worth publishing if *either*
    # the model or the rules flagged it. Requiring both would discard exactly
    # the novel patterns the model exists to find.
    selected = scored.loc[is_anomaly | risk_level.isin(["MEDIUM", "HIGH", "CRITICAL"])].copy()

    available = [c for c in ANOMALY_COLUMNS if c in selected.columns]
    output = selected[available].copy()

    # Reason lists are serialised to JSON strings: Parquet handles nested lists,
    # but DuckDB, pandas and every BI tool downstream handle strings uniformly,
    # and the dashboard parses them back when it needs the structure.
    for column in ("risk_reasons", "risk_rule_hits"):
        if column in output.columns:
            output[column] = output[column].map(
                lambda v: json.dumps(list(v)) if isinstance(v, list | tuple) else json.dumps([])
            )

    stamps = pd.to_datetime(output["transaction_timestamp"], errors="coerce", utc=True)
    if "transaction_date" not in output.columns:
        output["transaction_date"] = stamps.dt.date.astype("string")
    if "transaction_month" not in output.columns:
        output["transaction_month"] = stamps.dt.strftime("%Y-%m")

    output = output.sort_values("risk_score", ascending=False).reset_index(drop=True)

    return write_gold(
        output,
        ANOMALY_TABLE,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["silver_transactions"],
        transformations=["anomaly_selection", "risk_scoring", "explanation_attachment"],
        description=(
            "Transactions flagged by the anomaly model or the risk rules, with "
            "per-transaction explanations and model provenance."
        ),
        partition_by=["transaction_month"],
        records_in=len(scored),
        run_id=run_id,
        anomalies=int(is_anomaly.sum()),
        critical=int(risk_level.eq("CRITICAL").sum()),
    )


def build_risk_alerts(
    scored: pd.DataFrame,
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
    minimum_score: int | None = None,
) -> GoldResult:
    """Build the review queue from high-scoring transactions."""
    threshold = (
        minimum_score if minimum_score is not None else settings.streaming.alert_min_risk_score
    )

    if scored.empty:
        return write_gold(
            pd.DataFrame(columns=_ALERT_COLUMNS),
            ALERT_TABLE,
            settings,
            store=store,
            lineage=lineage,
            run_id=run_id,
            description="Review queue of transactions above the alert threshold.",
        )

    risk_score = pd.to_numeric(
        scored.get("risk_score", pd.Series(0, index=scored.index)), errors="coerce"
    ).fillna(0)
    candidates = scored.loc[risk_score >= threshold].copy()

    if candidates.empty:
        return write_gold(
            pd.DataFrame(columns=_ALERT_COLUMNS),
            ALERT_TABLE,
            settings,
            store=store,
            lineage=lineage,
            run_id=run_id,
            description="Review queue of transactions above the alert threshold.",
            threshold=threshold,
        )

    reasons = candidates.get(
        "risk_reasons", pd.Series([[]] * len(candidates), index=candidates.index)
    )
    rule_hits = candidates.get(
        "risk_rule_hits", pd.Series([[]] * len(candidates), index=candidates.index)
    )

    alerts = pd.DataFrame(
        {
            "alert_id": [
                deterministic_uuid("alert", txn, "|".join(sorted(hits or [])))
                for txn, hits in zip(
                    candidates["transaction_id"].astype(str), rule_hits, strict=False
                )
            ],
            "transaction_id": candidates["transaction_id"].astype("string").to_numpy(),
            "account_id": candidates.get("account_id", pd.Series(pd.NA, index=candidates.index))
            .astype("string")
            .to_numpy(),
            "vendor_id": candidates.get("vendor_id", pd.Series(pd.NA, index=candidates.index))
            .astype("string")
            .to_numpy(),
            "agency_id": candidates.get("agency_id", pd.Series(pd.NA, index=candidates.index))
            .astype("string")
            .to_numpy(),
            "amount_usd": pd.to_numeric(candidates.get("amount_usd"), errors="coerce")
            .fillna(0.0)
            .round(2)
            .to_numpy(),
            "risk_score": pd.to_numeric(candidates["risk_score"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy(),
            "risk_level": candidates.get("risk_level", pd.Series("LOW", index=candidates.index))
            .astype("string")
            .to_numpy(),
            "reason": [
                (
                    "; ".join(list(r)[:3])
                    if isinstance(r, list | tuple) and r
                    else "Threshold exceeded"
                )
                for r in reasons
            ],
            "reason_count": [len(r) if isinstance(r, list | tuple) else 0 for r in reasons],
            "anomaly_score": pd.to_numeric(candidates.get("anomaly_score"), errors="coerce")
            .fillna(0.0)
            .round(6)
            .to_numpy(),
            "model_version": candidates.get("model_version", pd.Series("", index=candidates.index))
            .astype("string")
            .to_numpy(),
            "transaction_timestamp": candidates.get("transaction_timestamp").to_numpy(),
            "created_at": pd.Timestamp.now(tz="UTC"),
            # Every alert starts OPEN. Triage state would be updated by a case
            # management system; the lifecycle is modelled so the dataset is
            # shaped correctly for one.
            "status": "OPEN",
            "assigned_to": pd.NA,
            "resolved_at": pd.NaT,
            "pipeline_run_id": run_id,
        }
    )

    alerts["alert_date"] = pd.to_datetime(
        alerts["transaction_timestamp"], errors="coerce", utc=True
    ).dt.date.astype("string")
    alerts = alerts.sort_values("risk_score", ascending=False).reset_index(drop=True)

    return write_gold(
        alerts,
        ALERT_TABLE,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["gold_anomaly_transactions"],
        transformations=["alert_threshold_filter", "deterministic_alert_id", "case_lifecycle"],
        description=(
            f"Review queue of transactions scoring at or above {threshold}. "
            "Alert ids are deterministic so re-runs update rather than duplicate cases."
        ),
        records_in=len(scored),
        run_id=run_id,
        threshold=threshold,
        alerts=len(alerts),
        critical_alerts=int((alerts["risk_level"] == "CRITICAL").sum()),
    )


_ALERT_COLUMNS = [
    "alert_id",
    "transaction_id",
    "account_id",
    "vendor_id",
    "agency_id",
    "amount_usd",
    "risk_score",
    "risk_level",
    "reason",
    "reason_count",
    "anomaly_score",
    "model_version",
    "transaction_timestamp",
    "created_at",
    "status",
    "assigned_to",
    "resolved_at",
    "pipeline_run_id",
    "alert_date",
]
