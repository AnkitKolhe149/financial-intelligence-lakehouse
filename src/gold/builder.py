"""Gold orchestration.

Reads the Silver layer once, scores it once, and fans the result out into every
Gold dataset. Reading Silver once matters: each builder needs the same scored
transaction frame, and re-reading it per builder would turn one scan into ten.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.io import dataset_exists, read_parquet, read_parquet_or_empty
from src.core.logging import get_logger
from src.core.run_context import RunContext
from src.gold.account_risk import build_account_risk
from src.gold.anomaly_summary import build_anomaly_transactions, build_risk_alerts
from src.gold.base import GoldResult
from src.gold.financial_summary import build_daily_financial_summary
from src.gold.quality_summary import (
    build_data_quality_summary,
    build_pipeline_run_summary,
    build_quality_trend,
)
from src.gold.spending import (
    build_agency_spending,
    build_department_spending,
    build_transaction_velocity,
)
from src.gold.star_schema import build_star_schema
from src.gold.vendor_risk import build_vendor_risk
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore, get_metadata_store

log = get_logger(__name__)

SILVER_INPUTS = {
    "transactions": "silver_transactions",
    "accounts": "silver_accounts",
    "vendors": "silver_vendors",
    "agencies": "silver_agencies",
    "departments": "silver_departments",
    "contracts": "silver_contracts",
    "budgets": "silver_budgets",
    "locations": "silver_locations",
}


@dataclass
class GoldBuildResult:
    """Every Gold dataset produced by one run."""

    datasets: dict[str, GoldResult] = field(default_factory=dict)
    duration_seconds: float = 0.0
    scored_transactions: pd.DataFrame | None = None

    @property
    def total_records(self) -> int:
        return sum(result.records for result in self.datasets.values())

    def summary(self) -> dict[str, Any]:
        return {
            "datasets": len(self.datasets),
            "total_records": self.total_records,
            "duration_seconds": round(self.duration_seconds, 3),
        }

    def counts(self) -> dict[str, int]:
        return {name: result.records for name, result in self.datasets.items()}


def load_silver(settings: Settings) -> dict[str, pd.DataFrame]:
    """Load every Silver table the Gold layer needs."""
    frames: dict[str, pd.DataFrame] = {}
    for alias, table in SILVER_INPUTS.items():
        path = settings.dataset_path("silver", table)
        frames[alias] = read_parquet(path) if dataset_exists(path) else pd.DataFrame()
        if frames[alias].empty:
            log.debug("silver_input_empty", table=table)
    return frames


def score_transactions(
    frames: dict[str, pd.DataFrame],
    settings: Settings,
    context: RunContext,
    *,
    store: MetadataStore | None = None,
    train: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build features, apply the anomaly model and the risk rules.

    When ``train`` is False the active registered model is loaded instead of
    fitting a new one - the normal production posture, where scoring runs on
    every batch but training runs on a schedule.
    """
    from src.ml.anomaly_model import train_anomaly_model
    from src.ml.evaluation import rule_model_agreement
    from src.ml.features import build_features
    from src.ml.model_registry import ModelRegistry
    from src.risk.risk_engine import RiskEngine

    transactions = frames.get("transactions", pd.DataFrame())
    if transactions.empty:
        return pd.DataFrame(), {"status": "no_transactions"}

    registry = ModelRegistry(settings, store)
    labels = _load_labels(settings)
    metrics: dict[str, Any] = {}

    if train:
        model, report, features = train_anomaly_model(
            transactions, frames.get("accounts"), frames.get("locations"), settings, labels
        )
        registry.save(model, report, context.run_id)
        metrics["model_evaluation"] = report.evaluation
        metrics["training_rows"] = report.training_rows
    else:
        registered = registry.load()
        if registered is None:
            log.warning("no_registered_model", detail="training a new model instead")
            return score_transactions(frames, settings, context, store=store, train=True)
        model = registered.model
        features = build_features(
            transactions, frames.get("accounts"), frames.get("locations")
        ).frame

    scored = model.score(features)
    context.model_version = model.model_version

    risk_result = RiskEngine(settings).score(scored)
    scored = risk_result.frame

    metrics["model_version"] = model.model_version
    metrics["anomalies"] = int(scored["is_anomaly"].sum())
    metrics["risk_levels"] = risk_result.level_counts
    metrics["rule_hits"] = risk_result.rule_hits
    metrics["rule_model_agreement"] = rule_model_agreement(scored)
    return scored, metrics


def build_all_gold(
    context: RunContext,
    settings: Settings | None = None,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    scored: pd.DataFrame | None = None,
    frames: dict[str, pd.DataFrame] | None = None,
    train_model: bool = True,
    build_star: bool = True,
) -> GoldBuildResult:
    """Build every Gold dataset from Silver."""
    settings = settings or get_settings()
    store = store if store is not None else get_metadata_store()
    started = time.perf_counter()

    frames = frames if frames is not None else load_silver(settings)
    if scored is None:
        scored, ml_metrics = score_transactions(
            frames, settings, context, store=store, train=train_model
        )
        for key, value in ml_metrics.items():
            context.add_metric(key, value)

    common = {"store": store, "lineage": lineage, "run_id": context.run_id}
    results: dict[str, GoldResult] = {}

    builders = [
        (
            "gold_daily_financial_summary",
            lambda: build_daily_financial_summary(scored, settings, **common),
        ),
        (
            "gold_account_risk",
            lambda: build_account_risk(
                scored, frames.get("accounts", pd.DataFrame()), settings, **common
            ),
        ),
        (
            "gold_vendor_risk",
            lambda: build_vendor_risk(
                scored,
                frames.get("vendors", pd.DataFrame()),
                frames.get("contracts", pd.DataFrame()),
                settings,
                **common,
            ),
        ),
        (
            "gold_agency_spending",
            lambda: build_agency_spending(
                scored,
                frames.get("agencies", pd.DataFrame()),
                frames.get("budgets", pd.DataFrame()),
                settings,
                **common,
            ),
        ),
        (
            "gold_department_spending",
            lambda: build_department_spending(
                scored,
                frames.get("departments", pd.DataFrame()),
                frames.get("budgets", pd.DataFrame()),
                settings,
                **common,
            ),
        ),
        (
            "gold_anomaly_transactions",
            lambda: build_anomaly_transactions(scored, settings, **common),
        ),
        ("gold_risk_alerts", lambda: build_risk_alerts(scored, settings, **common)),
        (
            "gold_transaction_velocity",
            lambda: build_transaction_velocity(scored, settings, **common),
        ),
        ("gold_data_quality_summary", lambda: build_data_quality_summary(settings, **common)),
        ("gold_data_quality_trend", lambda: build_quality_trend(settings, **common)),
        ("gold_pipeline_runs", lambda: build_pipeline_run_summary(settings, **common)),
    ]

    for name, builder in builders:
        try:
            results[name] = builder()
        except Exception as exc:
            log.error(
                "gold_dataset_failed",
                dataset=name,
                error=str(exc)[:300],
                error_type=type(exc).__name__,
            )
            if settings.pipeline.fail_fast:
                raise
            results[name] = GoldResult(table=name, records=0)

    if build_star:
        try:
            results.update(build_star_schema(scored, frames, settings, **common))
        except Exception as exc:
            log.error("star_schema_failed", error=str(exc)[:300])
            if settings.pipeline.fail_fast:
                raise

    result = GoldBuildResult(
        datasets=results,
        duration_seconds=time.perf_counter() - started,
        scored_transactions=scored,
    )
    log.info("gold_layer_complete", **result.summary())
    return result


def _load_labels(settings: Settings) -> pd.DataFrame | None:
    """Load held-out investigation labels, if they exist.

    Used only for evaluation. Absent labels are the normal production case and
    must not break anything.
    """
    path = settings.paths.raw / "labels" / "fraud_labels.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("labels_unreadable", path=str(path), error=str(exc)[:200])
        return None


def read_gold(table: str, settings: Settings | None = None, **kwargs) -> pd.DataFrame:
    """Read a Gold dataset, returning an empty frame when it does not exist yet."""
    settings = settings or get_settings()
    return read_parquet_or_empty(settings.dataset_path("gold", table), **kwargs)
