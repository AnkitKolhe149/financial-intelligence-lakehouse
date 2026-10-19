"""Isolation Forest anomaly detection.

Isolation Forest isolates points by randomly partitioning feature space:
anomalies need fewer splits to isolate, so they land in shallower leaves. Three
properties make it the right first model here:

* **No labels required**, which matches the operating reality.
* **Robust to heavy tails** - transaction amounts are lognormal, and a
  distance-based method would flag every large-but-legitimate payment.
* **Fast and explainable enough** - contamination maps directly to "how many
  alerts per day can the review team absorb", which is a business decision the
  configuration makes explicit.

The scores are normalised to ``[0, 1]`` where 1 is most anomalous, because the
raw ``score_samples`` output is a signed, unbounded quantity that nobody outside
the model can interpret and that would leak into the risk engine as an
uncalibrated number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.core.config import MLConfig, Settings, get_settings
from src.core.logging import get_logger
from src.ml.features import FEATURE_COLUMNS, build_features, temporal_split

log = get_logger(__name__)


@dataclass
class TrainingReport:
    """Everything needed to reproduce and judge a training run."""

    model_version: str
    algorithm: str
    #: The *configured* model name (``ml.model_name``), which is the key the
    #: registry looks models up by. Storing the algorithm here instead would
    #: mean ``active_model("anomaly_model")`` never finds anything.
    model_name: str
    trained_at: datetime
    training_rows: int
    validation_rows: int
    test_rows: int
    feature_list: list[str]
    hyperparameters: dict[str, Any]
    training_window_start: datetime | None
    training_window_end: datetime | None
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_registry_entry(self, artifact_path: str, run_id: str) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "trained_at": self.trained_at,
            "training_rows": self.training_rows,
            "feature_list": self.feature_list,
            "hyperparameters": self.hyperparameters,
            "training_window_start": self.training_window_start,
            "training_window_end": self.training_window_end,
            "evaluation": self.evaluation,
            "artifact_path": artifact_path,
            "run_id": run_id,
            "is_active": True,
        }


@dataclass
class ScoringResult:
    """Scored transactions plus the score distribution."""

    frame: pd.DataFrame
    model_version: str
    anomaly_count: int
    threshold: float
    scored_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def summary(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "records_scored": len(self.frame),
            "anomalies": self.anomaly_count,
            "anomaly_rate_pct": (
                round(self.anomaly_count / len(self.frame) * 100, 3) if len(self.frame) else 0.0
            ),
            "threshold": round(self.threshold, 6),
        }


class AnomalyModel:
    """Wraps the scaler + Isolation Forest pipeline and its metadata."""

    def __init__(self, config: MLConfig, model_version: str | None = None) -> None:
        self.config = config
        self.model_version = model_version or _new_version(config.model_name)
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)
        self.pipeline: Pipeline | None = None
        self.trained_at: datetime | None = None
        self.training_rows = 0
        #: Raw-score cut-off corresponding to the configured contamination,
        #: learned on the training data and reused at scoring time.
        self.raw_threshold: float = 0.0
        self._score_min: float = -1.0
        self._score_max: float = 1.0

    # -- training -----------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> AnomalyModel:
        """Fit the pipeline on a feature frame."""
        matrix = self._matrix(features)
        if len(matrix) < self.config.min_training_rows:
            log.warning(
                "training_set_small",
                rows=len(matrix),
                minimum=self.config.min_training_rows,
                detail="model will still be fitted, but scores are less reliable",
            )

        self.pipeline = Pipeline(
            steps=[
                # RobustScaler, not StandardScaler: transaction amounts are
                # heavy-tailed, and scaling by mean/std lets a handful of very
                # large payments dominate every feature.
                ("scale", RobustScaler(quantile_range=(5.0, 95.0))),
                (
                    "forest",
                    IsolationForest(
                        n_estimators=self.config.n_estimators,
                        contamination=self.config.contamination,
                        max_samples=self.config.max_samples,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        bootstrap=False,
                    ),
                ),
            ]
        )
        self.pipeline.fit(matrix)

        raw = self.pipeline.named_steps["forest"].score_samples(
            self.pipeline.named_steps["scale"].transform(matrix)
        )
        # Calibrate on training scores so scoring is stable across batches: a
        # per-batch percentile would make a transaction's classification depend
        # on which batch it happened to arrive in.
        self.raw_threshold = float(np.quantile(raw, self.config.contamination))
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        self.trained_at = datetime.now(UTC)
        self.training_rows = len(matrix)

        log.info(
            "model_trained",
            model_version=self.model_version,
            algorithm=self.config.algorithm,
            rows=self.training_rows,
            features=len(self.feature_columns),
            contamination=self.config.contamination,
            raw_threshold=round(self.raw_threshold, 6),
        )
        return self

    # -- scoring ------------------------------------------------------------

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Attach ``anomaly_score``, ``is_anomaly``, ``model_version``, ``scored_at``."""
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before scoring")
        if features.empty:
            scored = features.copy()
            for column, dtype in (
                ("anomaly_score", "float64"),
                ("is_anomaly", "bool"),
                ("model_version", "object"),
                ("scored_at", "datetime64[ns, UTC]"),
            ):
                scored[column] = pd.Series(dtype=dtype)
            return scored

        matrix = self._matrix(features)
        raw = self.pipeline.named_steps["forest"].score_samples(
            self.pipeline.named_steps["scale"].transform(matrix)
        )

        scored = features.copy()
        scored["anomaly_score"] = _normalise(raw, self._score_min, self._score_max)
        # Lower raw score = more isolated = more anomalous.
        scored["is_anomaly"] = raw <= self.raw_threshold
        scored["model_version"] = self.model_version
        scored["scored_at"] = datetime.now(UTC)
        scored["_raw_anomaly_score"] = raw
        return scored

    # -- internals ----------------------------------------------------------

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Project to the model's feature columns, in the model's order."""
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"Feature frame is missing columns required by {self.model_version}: "
                + ", ".join(missing)
            )
        matrix = features[self.feature_columns].to_numpy(dtype=float)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def describe(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "algorithm": self.config.algorithm,
            "features": len(self.feature_columns),
            "feature_list": self.feature_columns,
            "contamination": self.config.contamination,
            "n_estimators": self.config.n_estimators,
            "training_rows": self.training_rows,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }


def _normalise(raw: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Map raw scores onto [0, 1] where 1 is most anomalous.

    Clipped to the training range so that a new extreme value scores 1.0 rather
    than 1.4 - a probability-like output that leaves its range is worse than
    useless downstream.
    """
    span = maximum - minimum
    if span <= 0:
        return np.full_like(raw, 0.5, dtype=float)
    scaled = (raw - minimum) / span
    return np.clip(1.0 - scaled, 0.0, 1.0)


def _new_version(model_name: str) -> str:
    return f"{model_name}_v{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def train_anomaly_model(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame | None = None,
    locations: pd.DataFrame | None = None,
    settings: Settings | None = None,
    labels: pd.DataFrame | None = None,
) -> tuple[AnomalyModel, TrainingReport, pd.DataFrame]:
    """Build features, split chronologically, fit, and evaluate.

    Returns the fitted model, a training report and the full feature frame (so
    the caller can score everything without rebuilding features).
    """
    settings = settings or get_settings()
    config = settings.ml

    built = build_features(transactions, accounts, locations)
    features = built.frame
    split = temporal_split(features, config.train_fraction, config.validation_fraction)

    # Warm-up rows are excluded from *training* only.
    #
    # An account's first few transactions have no history to compare against, so
    # their relative features take neutral defaults (ratio 1.0, z-score 0). Those
    # rows are disproportionately concentrated at the start of the timeline -
    # exactly the period the chronological split assigns to training. Fitting on
    # them teaches the model that "neutral" is normal, and every mature account
    # later scores as unusual simply for having a history.
    #
    # They are still *scored*: a new account's first payment is a legitimate
    # thing to assess, and the rule engine handles it explicitly. This only
    # changes what the model learns "normal" from.
    training_frame = split.train if len(split.train) else features
    mature = training_frame.loc[training_frame["account_txn_count_prior"] >= 5]
    if len(mature) >= config.min_training_rows:
        log.info(
            "warmup_rows_excluded_from_training",
            excluded=len(training_frame) - len(mature),
            retained=len(mature),
            detail="rows with fewer than 5 prior account transactions",
        )
        training_frame = mature

    model = AnomalyModel(config)
    model.fit(training_frame)

    # Evaluation happens on the held-out test period only - the part of the
    # timeline the model has never seen, which is the only honest measurement.
    from src.ml.evaluation import evaluate_model

    scored_test = model.score(split.test) if len(split.test) else model.score(features)
    evaluation = evaluate_model(scored_test, labels)

    report = TrainingReport(
        model_version=model.model_version,
        algorithm=config.algorithm,
        model_name=config.model_name,
        trained_at=model.trained_at or datetime.now(UTC),
        training_rows=model.training_rows,
        validation_rows=len(split.validation),
        test_rows=len(split.test),
        feature_list=model.feature_columns,
        hyperparameters={
            "n_estimators": config.n_estimators,
            "contamination": config.contamination,
            "max_samples": config.max_samples,
            "random_state": config.random_state,
            "scaler": "RobustScaler(5-95)",
        },
        training_window_start=(
            pd.to_datetime(split.train["transaction_timestamp"]).min().to_pydatetime()
            if len(split.train)
            else None
        ),
        training_window_end=(
            split.train_end.to_pydatetime() if split.train_end is not None else None
        ),
        evaluation=evaluation,
    )
    return model, report, features
