"""A minimal local model registry.

MLflow would be overkill here and would add a service to run for what is, at
this scale, two concerns: *where is the artifact* and *what produced it*. Those
are handled by a joblib file on disk plus a row in the ``model_registry`` table,
which together answer every question that matters for reproducibility:

* which algorithm and hyperparameters
* which features, **in which order**
* how many rows, over which time window
* which pipeline run trained it
* how it scored on the held-out period

Loading a model asserts that its stored feature list matches the current
:data:`src.ml.features.FEATURE_COLUMNS`. If someone adds a feature without
retraining, scoring fails loudly instead of silently feeding the model a
differently-shaped matrix - the failure mode that produces confidently wrong
fraud alerts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.metadata.store import MetadataStore, get_metadata_store
from src.ml.anomaly_model import AnomalyModel, TrainingReport

log = get_logger(__name__)

MANIFEST_NAME = "model_manifest.json"


@dataclass
class RegisteredModel:
    model: AnomalyModel
    metadata: dict[str, Any]


class ModelRegistry:
    """Stores model artifacts on disk and their provenance in the metadata store."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: MetadataStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store if store is not None else get_metadata_store()
        self.root = Path(self.settings.paths.models)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- write --------------------------------------------------------------

    def save(self, model: AnomalyModel, report: TrainingReport, run_id: str) -> Path:
        """Persist the artifact and register its metadata."""
        artifact = self.root / f"{model.model_version}.joblib"
        joblib.dump(
            {
                "pipeline": model.pipeline,
                "feature_columns": model.feature_columns,
                "raw_threshold": model.raw_threshold,
                "score_min": model._score_min,
                "score_max": model._score_max,
                "model_version": model.model_version,
                "config": model.config.model_dump(),
                "trained_at": model.trained_at,
                "training_rows": model.training_rows,
            },
            artifact,
        )

        self.store.register_model(report.to_registry_entry(str(artifact), run_id))
        self._write_manifest(model, report, artifact, run_id)

        log.info(
            "model_registered",
            model_version=model.model_version,
            artifact=artifact.name,
            training_rows=report.training_rows,
            features=len(report.feature_list),
        )
        return artifact

    def _write_manifest(
        self, model: AnomalyModel, report: TrainingReport, artifact: Path, run_id: str
    ) -> None:
        """Write a human-readable manifest next to the artifacts.

        Duplicates what is in the database on purpose: the manifest survives a
        database reset and is what someone reads when they find a stray
        ``.joblib`` and need to know what it is.
        """
        manifest_path = self.root / MANIFEST_NAME
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}

        manifest[model.model_version] = {
            "algorithm": report.algorithm,
            "trained_at": report.trained_at.isoformat(),
            "training_rows": report.training_rows,
            "validation_rows": report.validation_rows,
            "test_rows": report.test_rows,
            "feature_count": len(report.feature_list),
            "feature_list": report.feature_list,
            "hyperparameters": report.hyperparameters,
            "training_window": {
                "start": (
                    report.training_window_start.isoformat()
                    if report.training_window_start
                    else None
                ),
                "end": (
                    report.training_window_end.isoformat() if report.training_window_end else None
                ),
            },
            "evaluation": report.evaluation,
            "artifact": artifact.name,
            "run_id": run_id,
        }
        manifest["_active"] = model.model_version
        manifest["_updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # -- read ---------------------------------------------------------------

    def load(self, model_version: str | None = None) -> RegisteredModel | None:
        """Load a model by version, or the active one when omitted."""
        entry = None
        if model_version is None:
            entry = self.store.active_model(self.settings.ml.model_name)
            if entry:
                model_version = str(entry["model_version"])

        if model_version is None:
            model_version = self._latest_from_manifest()
        if model_version is None:
            log.warning("no_model_available")
            return None

        artifact = self.root / f"{model_version}.joblib"
        if not artifact.exists():
            log.warning("model_artifact_missing", model_version=model_version, path=str(artifact))
            return None

        payload = joblib.load(artifact)
        stored_features = list(payload.get("feature_columns", []))

        from src.ml.features import FEATURE_COLUMNS

        if stored_features != list(FEATURE_COLUMNS):
            # Fail loudly. A mismatch means the code's feature definition moved
            # since training, and scoring would feed the model a matrix whose
            # columns no longer mean what it learned.
            raise ValueError(
                f"Model {model_version} was trained on a different feature set "
                f"({len(stored_features)} features) than the current code defines "
                f"({len(FEATURE_COLUMNS)}). Retrain before scoring.\n"
                f"  stored : {stored_features}\n"
                f"  current: {list(FEATURE_COLUMNS)}"
            )

        from src.core.config import MLConfig

        model = AnomalyModel(MLConfig(**payload["config"]), model_version=model_version)
        model.pipeline = payload["pipeline"]
        model.feature_columns = stored_features
        model.raw_threshold = float(payload["raw_threshold"])
        model._score_min = float(payload["score_min"])
        model._score_max = float(payload["score_max"])
        model.trained_at = payload.get("trained_at")
        model.training_rows = int(payload.get("training_rows", 0))

        log.info("model_loaded", model_version=model_version, features=len(stored_features))
        return RegisteredModel(model=model, metadata=entry or {})

    def _latest_from_manifest(self) -> str | None:
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        active = manifest.get("_active")
        if isinstance(active, str):
            return active
        versions = [k for k in manifest if not k.startswith("_")]
        return sorted(versions)[-1] if versions else None

    def history(self) -> list[dict[str, Any]]:
        """Registered models, newest first."""
        frame = self.store.model_history()
        return [] if frame.empty else frame.to_dict(orient="records")

    def describe_active(self) -> dict[str, Any]:
        entry = self.store.active_model(self.settings.ml.model_name)
        if entry:
            return entry
        version = self._latest_from_manifest()
        return {"model_version": version} if version else {}
