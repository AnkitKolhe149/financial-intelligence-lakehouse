"""Financial Intelligence Lakehouse - serving API.

    uvicorn src.api.main:app --port 8000
    open http://127.0.0.1:8000/docs

Endpoints
---------
    GET /health                     liveness + what the platform can currently see
    GET /metrics                    operational metrics for monitoring
    GET /transactions/{id}          one transaction with its risk explanation
    GET /accounts/{id}/risk         account risk profile
    GET /vendors/{id}/risk          vendor risk profile
    GET /anomalies                  flagged transactions, filterable
    GET /alerts                     the review queue
    GET /data-quality               quality scorecards
    GET /quarantine                 rejected records
    GET /pipeline-runs              execution history
    GET /lineage/{dataset}          upstream/downstream lineage
    GET /catalog                    dataset catalogue
    GET /risk-rules                 the configured rules and their weights
    GET /sources                    the source landscape
    GET /graph/vendor/{id}          relationship view (graph, degrades to SQL)

Every endpoint degrades rather than fails when a dataset has not been built yet:
the API returns an empty result with a hint about running the pipeline. A
serving layer that 500s before the first load is a serving layer nobody trusts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.core.config import get_settings
from src.core.io import dataset_exists, dataset_row_count, read_parquet_or_empty
from src.core.logging import get_logger
from src.core.schema import SCHEMAS
from src.gold.builder import read_gold
from src.metadata.lineage import upstream_of
from src.metadata.store import get_metadata_store
from src.silver.quarantine import quarantine_summary, read_quarantine

log = get_logger(__name__)

app = FastAPI(
    title="Financial Intelligence Lakehouse API",
    description=(
        "Serving layer for financial oversight analytics: risk profiles, "
        "anomalies, data quality, lineage and pipeline health."
    ),
    version=__version__,
    contact={"name": "Data Engineering"},
)

app.add_middleware(
    CORSMiddleware,
    # The dashboard runs on a different port locally; in a deployed environment
    # this list would be the actual front-end origins, not a wildcard.
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

PIPELINE_HINT = "Run `python scripts/run_demo.py` or `make pipeline` to build the lake."


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _jsonify(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe records.

    Timestamps become ISO strings and NaN becomes null, because ``NaN`` is not
    valid JSON and a client parsing it will fail in a way that is annoying to
    trace back to here.
    """
    if frame.empty:
        return []
    prepared = frame.copy()
    for column in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[column]):
            prepared[column] = prepared[column].astype("string")
    prepared = prepared.replace({pd.NA: None})
    records = json.loads(prepared.to_json(orient="records", date_format="iso"))
    return records


def _parse_reasons(value: Any) -> list[str]:
    """Risk reasons are stored as a JSON string in Gold; return them as a list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return [str(v) for v in parsed] if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return []


def _paginate(frame: pd.DataFrame, limit: int, offset: int) -> dict[str, Any]:
    total = len(frame)
    window = frame.iloc[offset : offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "returned": len(window),
        "results": _jsonify(window),
    }


def _empty_payload(limit: int, offset: int, hint: str = PIPELINE_HINT) -> dict[str, Any]:
    return {
        "total": 0,
        "limit": limit,
        "offset": offset,
        "returned": 0,
        "results": [],
        "hint": hint,
    }


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------


@app.get("/", tags=["operations"])
def root() -> dict[str, Any]:
    return {
        "service": "financial-intelligence-lakehouse",
        "version": __version__,
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["operations"])
def health() -> dict[str, Any]:
    settings = get_settings()
    store = get_metadata_store()

    lake = {
        "bronze": dataset_exists(settings.dataset_path("bronze", "bronze_transactions")),
        "silver": dataset_exists(settings.dataset_path("silver", "silver_transactions")),
        "gold": dataset_exists(settings.dataset_path("gold", "gold_daily_financial_summary")),
    }
    graph_available = False
    if settings.graph.enabled:
        try:
            from src.graph.neo4j_client import graph_available as probe

            graph_available = probe(settings)
        except Exception:
            graph_available = False

    healthy = lake["silver"] and not store.degraded
    return {
        "status": "healthy" if healthy else "degraded",
        "service": "financial-intelligence-lakehouse",
        "version": __version__,
        "timestamp": datetime.now(UTC),
        "engine": settings.engine,
        "metadata_backend": store.backend if store.available else "unavailable",
        "lake": lake,
        "graph_available": graph_available,
    }


@app.get("/metrics", tags=["operations"])
def metrics() -> dict[str, Any]:
    """Operational metrics, in the shape a monitoring system would scrape."""
    settings = get_settings()
    store = get_metadata_store()

    runs = store.recent_runs(limit=20)
    last_run = runs.iloc[0].to_dict() if not runs.empty else {}

    quality = read_gold("gold_data_quality_summary", settings)
    alerts = read_gold("gold_risk_alerts", settings)
    anomalies = read_gold("gold_anomaly_transactions", settings)
    accounts = read_gold("gold_account_risk", settings)
    vendors = read_gold("gold_vendor_risk", settings)

    model = store.active_model(settings.ml.model_name) or {}

    return {
        "generated_at": datetime.now(UTC),
        "pipeline": {
            "last_run_id": last_run.get("run_id"),
            "last_run_status": last_run.get("status"),
            "last_run_duration_seconds": last_run.get("duration_seconds"),
            "last_run_at": str(last_run.get("start_time")) if last_run else None,
            "records_read": int(last_run.get("records_read") or 0),
            "records_written": int(last_run.get("records_written") or 0),
            "records_quarantined": int(last_run.get("records_quarantined") or 0),
            "runs_recorded": len(runs),
            "failed_runs": int((runs["status"] == "FAILED").sum()) if not runs.empty else 0,
        },
        "data": {
            "bronze_transactions": dataset_row_count(
                settings.dataset_path("bronze", "bronze_transactions")
            ),
            "silver_transactions": dataset_row_count(
                settings.dataset_path("silver", "silver_transactions")
            ),
            "gold_datasets": (
                len(store.catalog().query("layer == 'gold'")) if not store.catalog().empty else 0
            ),
            "quarantined_records": (
                int(quarantine_summary(settings)["records"].sum())
                if not quarantine_summary(settings).empty
                else 0
            ),
        },
        "quality": {
            "overall_score": (
                round(float(quality["overall_score"].mean()), 2) if not quality.empty else None
            ),
            "datasets_scored": len(quality),
            "datasets_failing": (
                int((quality["status"] == "FAIL").sum()) if not quality.empty else 0
            ),
        },
        "risk": {
            "anomalies": len(anomalies),
            "open_alerts": int((alerts["status"] == "OPEN").sum()) if not alerts.empty else 0,
            "critical_alerts": (
                int((alerts["risk_level"] == "CRITICAL").sum()) if not alerts.empty else 0
            ),
            "high_risk_accounts": (
                int(accounts["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
                if not accounts.empty
                else 0
            ),
            "high_risk_vendors": (
                int(vendors["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
                if not vendors.empty
                else 0
            ),
        },
        "model": {
            "active_version": model.get("model_version"),
            "algorithm": model.get("algorithm"),
            "trained_at": str(model.get("trained_at")) if model.get("trained_at") else None,
            "training_rows": model.get("training_rows"),
        },
    }


# ---------------------------------------------------------------------------
# transactions and risk
# ---------------------------------------------------------------------------


@app.get("/transactions/{transaction_id}", tags=["transactions"])
def get_transaction(transaction_id: str) -> dict[str, Any]:
    """One transaction with its risk explanation, source and lineage."""
    settings = get_settings()

    silver = read_parquet_or_empty(
        settings.dataset_path("silver", "silver_transactions"),
        filters=[("transaction_id", "=", transaction_id)],
    )
    if silver.empty:
        # Not in Silver: it may have been quarantined, which is a far more
        # useful answer than "not found".
        quarantined = read_quarantine("transaction", settings)
        if not quarantined.empty:
            hit = quarantined[quarantined["raw_record"].str.contains(transaction_id, na=False)]
            if not hit.empty:
                row = hit.iloc[0]
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Transaction was quarantined and never reached Silver",
                        "error_code": row["error_code"],
                        "error_message": row["error_message"],
                        "source_system": row["source_system"],
                        "source_file": row["source_file"],
                        "detected_at": str(row["detected_at"]),
                    },
                )
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")

    record = _jsonify(silver.head(1))[0]

    scored = read_parquet_or_empty(
        settings.dataset_path("gold", "gold_anomaly_transactions"),
        filters=[("transaction_id", "=", transaction_id)],
    )
    if not scored.empty:
        enrichment = _jsonify(scored.head(1))[0]
        record.update(
            {
                "risk_score": int(enrichment.get("risk_score") or 0),
                "risk_level": enrichment.get("risk_level", "LOW"),
                "risk_reasons": _parse_reasons(enrichment.get("risk_reasons")),
                "rules_fired": _parse_reasons(enrichment.get("risk_rule_hits")),
                "anomaly_score": enrichment.get("anomaly_score"),
                "is_anomaly": bool(enrichment.get("is_anomaly", False)),
                "model_version": enrichment.get("model_version"),
            }
        )
    else:
        record.update(
            {
                "risk_score": 0,
                "risk_level": "LOW",
                "risk_reasons": [],
                "rules_fired": [],
                "is_anomaly": False,
            }
        )

    record["source_system"] = record.get("_source_system")
    record["lineage"] = _lineage_chain("silver_transactions")
    return record


@app.get("/accounts/{account_id}/risk", tags=["risk"])
def account_risk(account_id: str) -> dict[str, Any]:
    settings = get_settings()
    frame = read_gold("gold_account_risk", settings)
    if frame.empty:
        raise HTTPException(
            status_code=503, detail={"error": "Gold layer not built", "hint": PIPELINE_HINT}
        )

    match = frame[frame["account_id"].astype(str) == account_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    payload = _jsonify(match.head(1))[0]
    anomalies = read_parquet_or_empty(
        settings.dataset_path("gold", "gold_anomaly_transactions"),
        filters=[("account_id", "=", account_id)],
    )
    payload["recent_flagged_transactions"] = _jsonify(
        anomalies.nlargest(10, "risk_score") if not anomalies.empty else anomalies
    )
    return payload


@app.get("/vendors/{vendor_id}/risk", tags=["risk"])
def vendor_risk(vendor_id: str) -> dict[str, Any]:
    settings = get_settings()
    frame = read_gold("gold_vendor_risk", settings)
    if frame.empty:
        raise HTTPException(
            status_code=503, detail={"error": "Gold layer not built", "hint": PIPELINE_HINT}
        )

    match = frame[frame["vendor_id"].astype(str) == vendor_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")

    payload = _jsonify(match.head(1))[0]
    anomalies = read_parquet_or_empty(
        settings.dataset_path("gold", "gold_anomaly_transactions"),
        filters=[("vendor_id", "=", vendor_id)],
    )
    payload["flagged_transaction_count"] = len(anomalies)
    payload["recent_flagged_transactions"] = _jsonify(
        anomalies.nlargest(10, "risk_score") if not anomalies.empty else anomalies
    )
    return payload


@app.get("/anomalies", tags=["risk"])
def anomalies(
    risk_level: str | None = Query(None, description="LOW | MEDIUM | HIGH | CRITICAL"),
    account_id: str | None = None,
    vendor_id: str | None = None,
    agency_id: str | None = None,
    min_amount: float | None = Query(None, ge=0),
    min_risk_score: int | None = Query(None, ge=0, le=100),
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Flagged transactions, filtered."""
    settings = get_settings()
    frame = read_gold("gold_anomaly_transactions", settings)
    if frame.empty:
        return _empty_payload(limit, offset)

    if risk_level:
        frame = frame[frame["risk_level"].astype(str).str.upper() == risk_level.upper()]
    if account_id:
        frame = frame[frame["account_id"].astype(str) == account_id]
    if vendor_id:
        frame = frame[frame["vendor_id"].astype(str) == vendor_id]
    if agency_id:
        frame = frame[frame["agency_id"].astype(str) == agency_id]
    if min_amount is not None:
        frame = frame[pd.to_numeric(frame["amount_usd"], errors="coerce") >= min_amount]
    if min_risk_score is not None:
        frame = frame[pd.to_numeric(frame["risk_score"], errors="coerce") >= min_risk_score]
    if start_date:
        frame = frame[frame["transaction_date"].astype(str) >= start_date]
    if end_date:
        frame = frame[frame["transaction_date"].astype(str) <= end_date]

    frame = frame.sort_values("risk_score", ascending=False)
    payload = _paginate(frame, limit, offset)
    for record in payload["results"]:
        record["risk_reasons"] = _parse_reasons(record.get("risk_reasons"))
        record["risk_rule_hits"] = _parse_reasons(record.get("risk_rule_hits"))
    return payload


@app.get("/alerts", tags=["risk"])
def alerts(
    status: str | None = Query(None, description="OPEN | REVIEWING | RESOLVED | FALSE_POSITIVE"),
    risk_level: str | None = None,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    settings = get_settings()
    frame = read_gold("gold_risk_alerts", settings)
    if frame.empty:
        return _empty_payload(limit, offset)

    if status:
        frame = frame[frame["status"].astype(str).str.upper() == status.upper()]
    if risk_level:
        frame = frame[frame["risk_level"].astype(str).str.upper() == risk_level.upper()]
    return _paginate(frame.sort_values("risk_score", ascending=False), limit, offset)


# ---------------------------------------------------------------------------
# quality, quarantine, runs, lineage
# ---------------------------------------------------------------------------


@app.get("/data-quality", tags=["quality"])
def data_quality(dataset: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    frame = read_gold("gold_data_quality_summary", settings)
    if frame.empty:
        return {"datasets": [], "hint": PIPELINE_HINT}
    if dataset:
        frame = frame[frame["dataset"].astype(str) == dataset]
    return {
        "datasets": _jsonify(frame),
        # Weighted by records evaluated, matching how the pipeline computes it.
        # An unweighted mean would let six small dimension tables at 100% mask a
        # problem in the fact table, and the API would then disagree with the
        # pipeline's own reported score for the same run.
        "platform_score": _platform_score(frame),
    }


def _platform_score(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    weights = pd.to_numeric(frame.get("records_evaluated"), errors="coerce").fillna(1).clip(lower=1)
    scores = pd.to_numeric(frame["overall_score"], errors="coerce").fillna(0)
    total = float(weights.sum())
    return round(float((scores * weights).sum() / total), 2) if total else None


@app.get("/data-quality/checks", tags=["quality"])
def quality_checks(run_id: str | None = None, failed_only: bool = True) -> dict[str, Any]:
    """Individual check results for a run."""
    store = get_metadata_store()
    frame = store.failed_checks(run_id) if failed_only else store.checks_for_run(run_id or "")
    return {"checks": _jsonify(frame), "count": len(frame)}


@app.get("/quarantine", tags=["quality"])
def quarantine(
    dataset: str = "transaction",
    error_code: str | None = None,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Rejected records - nothing is deleted, so everything is inspectable."""
    settings = get_settings()
    frame = read_quarantine(dataset, settings, error_code=error_code)
    if frame.empty:
        return _empty_payload(limit, offset, hint="No records have been quarantined.")

    summary = quarantine_summary(settings)
    payload = _paginate(frame.sort_values("detected_at", ascending=False), limit, offset)
    payload["summary_by_error_code"] = (
        frame["error_code"].value_counts().to_dict() if not frame.empty else {}
    )
    payload["summary_by_source"] = (
        summary.groupby("source_system")["records"].sum().to_dict() if not summary.empty else {}
    )
    return payload


@app.get("/pipeline-runs", tags=["operations"])
def pipeline_runs(
    pipeline_name: str | None = None,
    limit: int = Query(25, ge=1, le=200),
) -> dict[str, Any]:
    store = get_metadata_store()
    frame = store.recent_runs(limit=limit, pipeline_name=pipeline_name)
    if frame.empty:
        return {"runs": [], "count": 0, "hint": PIPELINE_HINT}
    return {"runs": _jsonify(frame), "count": len(frame)}


@app.get("/pipeline-runs/{run_id}", tags=["operations"])
def pipeline_run(run_id: str) -> dict[str, Any]:
    store = get_metadata_store()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    stages = store.run_stages(run_id)
    quality = store.checks_for_run(run_id)
    return {
        "run": json.loads(pd.Series(run).to_json(date_format="iso")),
        "stages": _jsonify(stages),
        "quality_checks": len(quality),
        "failed_checks": _jsonify(quality[~quality["passed"]]) if not quality.empty else [],
    }


@app.get("/lineage/{dataset}", tags=["governance"])
def lineage(dataset: str) -> dict[str, Any]:
    """Upstream and downstream lineage for a dataset."""
    store = get_metadata_store()
    edges = store.lineage_for(dataset)
    if edges.empty:
        return {
            "dataset": dataset,
            "upstream": [],
            "downstream": [],
            "edges": [],
            "hint": PIPELINE_HINT,
        }

    records = edges.to_dict(orient="records")
    downstream = sorted(
        {str(e["target_dataset"]) for e in records if e["source_dataset"] == dataset}
    )
    return {
        "dataset": dataset,
        "upstream": upstream_of(dataset, store.all_lineage().to_dict(orient="records")),
        "downstream": downstream,
        "edges": _jsonify(edges),
    }


@app.get("/catalog", tags=["governance"])
def catalog(layer: str | None = None) -> dict[str, Any]:
    """The dataset catalogue - the local stand-in for Unity Catalog."""
    store = get_metadata_store()
    frame = store.catalog()
    if frame.empty:
        return {"datasets": [], "hint": PIPELINE_HINT}
    if layer:
        frame = frame[frame["layer"].astype(str) == layer]
    return {"datasets": _jsonify(frame), "count": len(frame)}


@app.get("/schemas", tags=["governance"])
def schemas() -> dict[str, Any]:
    """Published data contracts for every canonical dataset."""
    return {
        "datasets": {
            name: {
                "description": schema.description,
                "business_key": list(schema.business_key),
                "schema_version": schema.schema_version,
                "owner": schema.owner,
                "classification": schema.classification,
                "fields": [
                    {
                        "name": f.name,
                        "type": f.dtype,
                        "nullable": f.nullable,
                        "description": f.description,
                        "business_meaning": f.business_meaning or None,
                    }
                    for f in schema.fields
                ],
            }
            for name, schema in SCHEMAS.items()
        }
    }


@app.get("/risk-rules", tags=["risk"])
def risk_rules() -> dict[str, Any]:
    """The configured risk rules, their weights and thresholds."""
    from src.risk.rules import describe_rules

    settings = get_settings()
    rules = describe_rules(settings.risk)
    return {
        "max_score": settings.risk.max_score,
        "levels": {
            name: {"min": band.min, "max": band.max} for name, band in settings.risk.levels.items()
        },
        "rules": rules,
        "max_achievable_score": min(sum(r["points"] for r in rules), settings.risk.max_score),
    }


@app.get("/sources", tags=["governance"])
def sources() -> dict[str, Any]:
    """The source landscape the platform ingests from."""
    from src.ingestion.sources import catalogue_summary

    return {"sources": catalogue_summary()}


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


@app.get("/graph/vendor/{vendor_id}", tags=["graph"])
def graph_vendor(vendor_id: str, depth: int = Query(2, ge=1, le=3)) -> dict[str, Any]:
    """Relationship neighbourhood of a vendor.

    Uses Neo4j when available and falls back to the in-process graph built from
    Parquet otherwise, so the endpoint answers either way.
    """
    settings = get_settings()
    try:
        from src.graph.graph_queries import vendor_neighbourhood

        return vendor_neighbourhood(vendor_id, depth=depth, settings=settings)
    except Exception as exc:
        log.warning("graph_query_failed", vendor_id=vendor_id, error=str(exc)[:200])
        return {
            "vendor_id": vendor_id,
            "available": False,
            "reason": str(exc)[:200],
            "hint": "Start Neo4j with `docker compose up -d neo4j`, or the "
            "in-process fallback will be used once the lake is built.",
        }


@app.exception_handler(Exception)
async def unhandled_exception(_request: Any, exc: Exception) -> JSONResponse:
    """Return a structured error rather than an HTML stack trace."""
    log.error("api_unhandled_error", error=str(exc)[:300], error_type=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal error", "detail": str(exc)[:200], "hint": PIPELINE_HINT},
    )


def _lineage_chain(dataset: str) -> list[str]:
    store = get_metadata_store()
    edges = store.all_lineage()
    if edges.empty:
        return [dataset]
    chain = upstream_of(dataset, edges.to_dict(orient="records"))
    return [*reversed(chain), dataset]


def run() -> None:  # pragma: no cover - entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.services.serving_api_host,
        port=settings.services.serving_api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
