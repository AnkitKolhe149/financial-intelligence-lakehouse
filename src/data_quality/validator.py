"""Quality suite execution.

The validator loads a dataset, assembles the context each check needs (reference
key sets, the stored distribution baseline), runs every check with per-check
error isolation, scores the results and persists them.

Per-check isolation matters: a check that raises must be recorded as a *failed
check*, not allowed to abort the quality run. Losing the whole scorecard because
one check hit an unexpected dtype is exactly the fragility that makes teams
switch quality monitoring off.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.exceptions import DataQualityError
from src.core.io import dataset_exists, read_parquet, read_parquet_or_empty
from src.core.logging import get_logger
from src.core.schema import table_name as build_table_name
from src.data_quality.base_check import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    DataQualityCheck,
    Dimension,
)
from src.data_quality.checks import profile_numeric
from src.data_quality.scoring import QualityScorecard, score_dataset
from src.data_quality.suites import DEFAULT_SCORED_DATASETS, build_suite
from src.metadata.store import MetadataStore, get_metadata_store
from src.silver.spec import get_silver_spec

log = get_logger(__name__)

BASELINE_FILENAME = "quality_baselines.json"


@dataclass
class QualityRunResult:
    """Scorecards for every dataset evaluated in one run."""

    scorecards: dict[str, QualityScorecard]
    duration_seconds: float

    @property
    def overall(self) -> float:
        from src.data_quality.scoring import aggregate_platform_score

        return aggregate_platform_score(list(self.scorecards.values()))

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [check for card in self.scorecards.values() for check in card.blocking_failures]

    def summary(self) -> dict[str, Any]:
        return {
            "datasets": len(self.scorecards),
            "overall_score": self.overall,
            "blocking_failures": len(self.blocking_failures),
            "duration_seconds": round(self.duration_seconds, 3),
        }


class DataQualityValidator:
    """Runs check suites against Silver datasets and records the results."""

    def __init__(
        self,
        run_id: str,
        settings: Settings | None = None,
        store: MetadataStore | None = None,
    ) -> None:
        self.run_id = run_id
        self.settings = settings or get_settings()
        self.store = store if store is not None else get_metadata_store()
        self._baselines = self._load_baselines()

    # -- public API ---------------------------------------------------------

    def evaluate_dataset(
        self,
        dataset: str,
        frame: pd.DataFrame | None = None,
        *,
        layer: str = "silver",
        persist: bool = True,
    ) -> QualityScorecard:
        """Score one dataset."""
        if frame is None:
            frame = self._load_dataset(dataset, layer)

        checks = build_suite(dataset, self.settings)
        context = CheckContext(
            dataset=dataset,
            layer=layer,
            references=self._load_references(dataset),
            baseline=self._baselines.get(dataset, {}),
            settings=self.settings,
        )

        results = [self._run_check(check, frame, context) for check in checks]
        scorecard = score_dataset(
            dataset,
            results,
            self.run_id,
            layer=layer,
            records_evaluated=len(frame),
            settings=self.settings,
        )

        self._capture_baseline(dataset, frame, results)

        if persist:
            self.store.record_quality_result(scorecard.to_result_row(), scorecard.to_check_rows())

        log.info(
            "quality_evaluated",
            dataset=dataset,
            overall_score=scorecard.overall_score,
            status=scorecard.status,
            checks_passed=scorecard.checks_passed,
            checks_total=len(scorecard.checks),
            records=len(frame),
        )
        return scorecard

    def evaluate_all(
        self,
        datasets: tuple[str, ...] = DEFAULT_SCORED_DATASETS,
        *,
        persist: bool = True,
    ) -> QualityRunResult:
        """Score every configured dataset."""
        started = time.perf_counter()
        scorecards: dict[str, QualityScorecard] = {}

        for dataset in datasets:
            try:
                scorecards[dataset] = self.evaluate_dataset(dataset, persist=persist)
            except Exception as exc:
                log.error(
                    "quality_dataset_failed",
                    dataset=dataset,
                    error=str(exc)[:300],
                    error_type=type(exc).__name__,
                )

        self._save_baselines()
        result = QualityRunResult(scorecards, time.perf_counter() - started)
        log.info("quality_run_complete", **result.summary())
        return result

    def enforce(self, result: QualityRunResult) -> None:
        """Raise if a blocking check failed.

        Called by the pipeline only when the caller opted into strict mode. The
        default posture is to record and continue: an oversight platform that
        refuses to publish yesterday's data because 0.6% of records were
        malformed has failed at its job more completely than one that publishes
        with a documented quality score.
        """
        failures = result.blocking_failures
        if not failures:
            return
        raise DataQualityError(
            "Blocking data-quality checks failed",
            failed_checks=[c.check_name for c in failures],
            details=[c.message for c in failures][:5],
        )

    # -- internals ----------------------------------------------------------

    def _run_check(
        self, check: DataQualityCheck, frame: pd.DataFrame, context: CheckContext
    ) -> CheckResult:
        try:
            return check.evaluate(frame, context)
        except Exception as exc:
            log.error(
                "quality_check_errored",
                check=check.name,
                dataset=context.dataset,
                error=str(exc)[:200],
                error_type=type(exc).__name__,
            )
            return CheckResult(
                check_name=check.name,
                dimension=check.dimension,
                severity=CheckSeverity.WARNING,
                passed=False,
                score=0.0,
                records_evaluated=len(frame),
                records_failed=0,
                message=f"check raised {type(exc).__name__}: {exc}"[:400],
                details={"errored": True},
            )

    def _load_dataset(self, dataset: str, layer: str) -> pd.DataFrame:
        table = build_table_name(layer, dataset)
        path = self.settings.dataset_path(layer, table)
        if not dataset_exists(path):
            log.warning("quality_dataset_missing", dataset=dataset, table=table)
            return pd.DataFrame()
        return read_parquet(path)

    def _load_references(self, dataset: str) -> dict[str, pd.Series]:
        """Load only the key columns of referenced dimensions.

        Column pruning is the whole point: an integrity check needs one column,
        not the dimension.
        """
        try:
            spec = get_silver_spec(dataset)
        except KeyError:
            return {}

        references: dict[str, pd.Series] = {}
        for table, key_column in spec.reference_tables.items():
            path = self.settings.dataset_path("silver", table)
            if not dataset_exists(path):
                continue
            frame = read_parquet_or_empty(path, columns=[key_column])
            if not frame.empty:
                references[table] = frame[key_column]
        return references

    # -- distribution baselines ---------------------------------------------

    def _baseline_path(self) -> Path:
        return Path(self.settings.paths.metadata) / BASELINE_FILENAME

    def _load_baselines(self) -> dict[str, dict[str, Any]]:
        path = self._baseline_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("quality_baseline_unreadable", path=str(path))
            return {}

    def _capture_baseline(
        self, dataset: str, frame: pd.DataFrame, results: list[CheckResult]
    ) -> None:
        """Store the numeric profile for future PSI comparisons."""
        drift_columns = [
            check.column_name
            for check in results
            if check.dimension == Dimension.DISTRIBUTION and check.column_name
        ]
        if not drift_columns or frame.empty:
            return
        profile = self._baselines.setdefault(dataset, {})
        for column in drift_columns:
            if column in frame.columns:
                captured = profile_numeric(frame[column])
                if captured:
                    profile[column] = captured

    def _save_baselines(self) -> None:
        if not self._baselines:
            return
        path = self._baseline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._baselines, indent=2), encoding="utf-8")


def run_quality_suite(
    run_id: str,
    settings: Settings | None = None,
    store: MetadataStore | None = None,
    datasets: tuple[str, ...] = DEFAULT_SCORED_DATASETS,
) -> QualityRunResult:
    """Convenience entry point used by the pipelines."""
    validator = DataQualityValidator(run_id, settings, store)
    return validator.evaluate_all(datasets)
