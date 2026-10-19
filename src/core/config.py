"""Typed configuration loading.

Design decisions worth knowing:

* **Three-tier resolution** - code defaults < ``config/settings.yaml`` <
  environment variables. Non-secret tunables live in YAML (reviewable in a pull
  request); secrets and host-specific values live only in the environment.
* **Everything is typed** - the YAML is parsed into Pydantic models, so a typo
  such as ``contaminaton: 0.01`` fails at start-up with a clear message instead
  of silently changing model behaviour at 3 a.m.
* **Paths are absolute ``Path`` objects** resolved against the repository root,
  never strings glued together. The project therefore runs identically on
  Linux, macOS and Windows, and from any working directory.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# repository location
# ---------------------------------------------------------------------------

_ROOT_MARKERS = ("pyproject.toml", ".git")


def project_root() -> Path:
    """Return the repository root.

    Walks up from this file looking for a marker. Falls back to three levels up
    (``src/core/config.py`` -> repo root) so the package still works when
    installed as a wheel where ``pyproject.toml`` is absent.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return here.parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {name} must be an integer", value=raw
        ) from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {name} must be a number", value=raw
        ) from exc


# ---------------------------------------------------------------------------
# configuration sections
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    """Reject unknown keys so configuration typos surface immediately."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Base):
    raw: Path
    bronze: Path
    silver: Path
    gold: Path
    quarantine: Path
    sample: Path
    models: Path
    metadata: Path
    streaming_inbox: Path
    streaming_checkpoint: Path
    duckdb: Path

    def all_directories(self) -> list[Path]:
        """Directories the platform is allowed to create on demand."""
        return [
            self.raw,
            self.bronze,
            self.silver,
            self.gold,
            self.quarantine,
            self.models,
            self.metadata,
            self.streaming_inbox,
            self.streaming_checkpoint,
        ]

    def ensure(self) -> None:
        for directory in self.all_directories():
            directory.mkdir(parents=True, exist_ok=True)
        self.duckdb.parent.mkdir(parents=True, exist_ok=True)


class Storage(_Base):
    format: str = "parquet"
    compression: str = "snappy"
    partition_columns: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("format")
    @classmethod
    def _known_format(cls, value: str) -> str:
        if value not in {"parquet", "delta"}:
            raise ValueError(f"storage.format must be 'parquet' or 'delta', got {value!r}")
        return value

    def partitions_for(self, dataset: str) -> list[str]:
        return self.partition_columns.get(dataset, [])


class PipelineConfig(_Base):
    name: str = "financial_intelligence"
    version: str = "1.0.0"
    batch_size: int = 10_000
    enable_streaming: bool = True
    idempotent_bronze: bool = True
    fail_fast: bool = False


class ScaleProfile(_Base):
    transactions: int
    accounts: int
    vendors: int
    agencies: int
    contracts: int
    events: int


class GenerationConfig(_Base):
    seed: int = 42
    scale: dict[str, ScaleProfile]
    start_date: date
    end_date: date
    currencies: list[str]

    @model_validator(mode="after")
    def _dates_ordered(self) -> GenerationConfig:
        if self.end_date <= self.start_date:
            raise ValueError("generation.end_date must be after generation.start_date")
        return self

    def profile(self, name: str) -> ScaleProfile:
        try:
            return self.scale[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.scale))
            raise ConfigurationError(
                f"Unknown generation scale {name!r}; known profiles: {known}"
            ) from exc


class InjectionConfig(_Base):
    """Rates of deliberately injected data-quality defects."""

    enabled: bool = True
    duplicate_rate: float = 0.0
    null_account_rate: float = 0.0
    null_vendor_rate: float = 0.0
    null_department_rate: float = 0.0
    invalid_amount_rate: float = 0.0
    extreme_amount_rate: float = 0.0
    invalid_currency_rate: float = 0.0
    orphan_account_rate: float = 0.0
    orphan_vendor_rate: float = 0.0
    invalid_contract_rate: float = 0.0
    future_timestamp_rate: float = 0.0
    invalid_timestamp_rate: float = 0.0
    whitespace_rate: float = 0.0
    malformed_email_rate: float = 0.0
    duplicate_vendor_rate: float = 0.0
    out_of_order_event_rate: float = 0.0

    @model_validator(mode="after")
    def _rates_are_proportions(self) -> InjectionConfig:
        for name, value in self.model_dump().items():
            if name == "enabled":
                continue
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"data_quality_injection.{name} must be within [0, 1], got {value}"
                )
        return self

    def scaled(self, factor: float) -> InjectionConfig:
        """Return the same profile with every rate multiplied by ``factor``.

        Used by tests that want a heavily-polluted dataset without editing the
        committed configuration.
        """
        payload = self.model_dump()
        for key, value in payload.items():
            if key != "enabled":
                payload[key] = min(1.0, float(value) * factor)
        return InjectionConfig(**payload)


class QualityThresholds(_Base):
    overall_warn: float = 95.0
    overall_fail: float = 90.0


class VolumeConfig(_Base):
    max_deviation_pct: float = 40.0


class DistributionConfig(_Base):
    psi_warn: float = 0.10
    psi_fail: float = 0.25


class QualityConfig(_Base):
    quarantine_invalid_records: bool = True
    score_weights: dict[str, float]
    thresholds: QualityThresholds = Field(default_factory=QualityThresholds)
    freshness_max_age_hours: int = 48
    volume: VolumeConfig = Field(default_factory=VolumeConfig)
    distribution: DistributionConfig = Field(default_factory=DistributionConfig)
    allowed_currencies: list[str]
    max_transaction_amount: float = 50_000_000.0

    @field_validator("score_weights")
    @classmethod
    def _weights_sum_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        total = sum(value.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"quality.score_weights must sum to 1.0, got {total:.4f} "
                f"({', '.join(f'{k}={v}' for k, v in value.items())})"
            )
        return value

    @field_validator("allowed_currencies")
    @classmethod
    def _currencies_upper(cls, value: list[str]) -> list[str]:
        return [code.upper() for code in value]


class MLConfig(_Base):
    algorithm: str = "IsolationForest"
    contamination: float = 0.01
    n_estimators: int = 200
    max_samples: str | int = "auto"
    random_state: int = 42
    model_name: str = "anomaly_model"
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    min_training_rows: int = 500

    @model_validator(mode="after")
    def _fractions_leave_room_for_test(self) -> MLConfig:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError(
                "ml.train_fraction + ml.validation_fraction must be < 1.0 so that a "
                "held-out test period remains"
            )
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("ml.contamination must be within (0, 0.5)")
        return self

    @property
    def test_fraction(self) -> float:
        return round(1.0 - self.train_fraction - self.validation_fraction, 6)


class RiskLevelBand(_Base):
    min: int
    max: int


class RiskConfig(_Base):
    max_score: int = 100
    levels: dict[str, RiskLevelBand]
    rules: dict[str, dict[str, Any]]

    @model_validator(mode="after")
    def _bands_cover_range(self) -> RiskConfig:
        bands = sorted(self.levels.values(), key=lambda b: b.min)
        if not bands:
            raise ValueError("risk.levels must not be empty")
        if bands[0].min != 0:
            raise ValueError("risk.levels must start at 0")
        if bands[-1].max != self.max_score:
            raise ValueError(f"risk.levels must extend to risk.max_score ({self.max_score})")
        for lower, upper in pairwise(bands):
            if upper.min != lower.max + 1:
                raise ValueError(
                    f"risk.levels must be contiguous; gap between {lower.max} and {upper.min}"
                )
        return self

    def level_for(self, score: float) -> str:
        """Map a numeric score onto its band name."""
        bounded = max(0, min(round(score), self.max_score))
        for name, band in self.levels.items():
            if band.min <= bounded <= band.max:
                return name
        return "LOW"  # pragma: no cover - unreachable given the validator above

    def rule(self, name: str) -> dict[str, Any]:
        try:
            return self.rules[name]
        except KeyError as exc:
            raise ConfigurationError(f"Risk rule {name!r} is not configured") from exc


class GraphConfig(_Base):
    enabled: bool = True
    batch_size: int = 1000
    max_transactions: int = 20_000
    high_risk_only: bool = False


class StreamingConfig(_Base):
    poll_interval_seconds: float = 2.0
    micro_batch_max_files: int = 50
    alert_min_risk_score: int = 50
    watermark_minutes: int = 30


class ObservabilityConfig(_Base):
    metrics_enabled: bool = True
    log_record_counts: bool = True


class USASpendingConfig(_Base):
    """Public API - no credentials required, but everything is still tunable."""

    enabled: bool = True
    base_url: str = "https://api.usaspending.gov"
    page_size: int = 100
    timeout: int = 30
    max_retries: int = 4
    rate_limit_per_sec: float = 4.0


class ServiceConfig(_Base):
    source_api_base_url: str = "http://127.0.0.1:8001"
    source_api_host: str = "127.0.0.1"
    source_api_port: int = 8001
    serving_api_host: str = "127.0.0.1"
    serving_api_port: int = 8000
    dashboard_port: int = 8501


class Neo4jConfig(_Base):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "fil_local_password"
    database: str = "neo4j"


# ---------------------------------------------------------------------------
# root settings object
# ---------------------------------------------------------------------------


class Settings(_Base):
    environment: str = "local"
    engine: str = "pandas"
    log_level: str = "INFO"
    log_format: str = "console"

    paths: Paths
    storage: Storage
    pipeline: PipelineConfig
    generation: GenerationConfig
    injection: InjectionConfig
    quality: QualityConfig
    ml: MLConfig
    risk: RiskConfig
    graph: GraphConfig
    streaming: StreamingConfig
    observability: ObservabilityConfig
    usaspending: USASpendingConfig
    services: ServiceConfig
    neo4j: Neo4jConfig

    postgres_dsn: str | None = None

    @field_validator("engine")
    @classmethod
    def _known_engine(cls, value: str) -> str:
        value = value.lower()
        if value not in {"pandas", "spark"}:
            raise ValueError(f"engine must be 'pandas' or 'spark', got {value!r}")
        return value

    @property
    def metadata_backend(self) -> str:
        """``postgres`` when a DSN is configured, otherwise the SQLite fallback."""
        return "postgres" if self.postgres_dsn else "sqlite"

    def dataset_path(self, layer: str, dataset: str) -> Path:
        """Resolve the on-disk location of a lake dataset.

        ``dataset_path("silver", "silver_transactions")`` ->
        ``<root>/data/lake/silver/silver_transactions``
        """
        layers = {
            "bronze": self.paths.bronze,
            "silver": self.paths.silver,
            "gold": self.paths.gold,
            "quarantine": self.paths.quarantine,
        }
        try:
            base = layers[layer]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown lake layer {layer!r}; expected one of {sorted(layers)}"
            ) from exc
        return base / dataset


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError("Configuration file not found", path=str(path))
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration root must be a mapping", path=str(path))
    return data


def _load_dotenv(root: Path) -> None:
    """Load ``.env`` if present, without overriding a real environment variable.

    Real environment variables win so that ``FIL_ENGINE=spark`` in a container
    beats whatever a developer left in their local ``.env``.
    """
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a hard dependency
        return
    load_dotenv(env_file, override=False)


def load_settings(config_path: Path | None = None) -> Settings:
    """Build a :class:`Settings` instance from YAML + environment variables."""
    root = project_root()
    _load_dotenv(root)

    path = config_path or Path(_env("FIL_CONFIG_FILE") or root / "config" / "settings.yaml")
    raw = _load_yaml(path)

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (root / candidate)

    raw_paths = raw.get("paths", {})
    paths = Paths(**{key: resolve(value) for key, value in raw_paths.items()})

    settings = Settings(
        environment=_env("FIL_ENVIRONMENT", raw.get("environment", "local")) or "local",
        engine=_env("FIL_ENGINE", raw.get("engine", "pandas")) or "pandas",
        log_level=(_env("FIL_LOG_LEVEL", "INFO") or "INFO").upper(),
        log_format=(_env("FIL_LOG_FORMAT", "console") or "console").lower(),
        paths=paths,
        storage=Storage(**raw.get("storage", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        injection=InjectionConfig(**raw.get("data_quality_injection", {})),
        quality=QualityConfig(**raw.get("quality", {})),
        ml=MLConfig(**raw.get("ml", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        graph=GraphConfig(
            **{
                **raw.get("graph", {}),
                "enabled": _env_bool(
                    "FIL_GRAPH_ENABLED", raw.get("graph", {}).get("enabled", True)
                ),
            }
        ),
        streaming=StreamingConfig(**raw.get("streaming", {})),
        observability=ObservabilityConfig(**raw.get("observability", {})),
        usaspending=USASpendingConfig(
            enabled=_env_bool("USASPENDING_ENABLED", True),
            base_url=_env("USASPENDING_BASE_URL", "https://api.usaspending.gov") or "",
            page_size=_env_int("USASPENDING_PAGE_SIZE", 100),
            timeout=_env_int("USASPENDING_TIMEOUT", 30),
            max_retries=_env_int("USASPENDING_MAX_RETRIES", 4),
            rate_limit_per_sec=_env_float("USASPENDING_RATE_LIMIT_PER_SEC", 4.0),
        ),
        services=ServiceConfig(
            source_api_base_url=_env("SOURCE_API_BASE_URL", "http://127.0.0.1:8001") or "",
            source_api_host=_env("SOURCE_API_HOST", "127.0.0.1") or "",
            source_api_port=_env_int("SOURCE_API_PORT", 8001),
            serving_api_host=_env("SERVING_API_HOST", "127.0.0.1") or "",
            serving_api_port=_env_int("SERVING_API_PORT", 8000),
            dashboard_port=_env_int("DASHBOARD_PORT", 8501),
        ),
        neo4j=Neo4jConfig(
            uri=_env("NEO4J_URI", "bolt://localhost:7687") or "",
            user=_env("NEO4J_USER", "neo4j") or "",
            password=_env("NEO4J_PASSWORD", "fil_local_password") or "",
            database=_env("NEO4J_DATABASE", "neo4j") or "",
        ),
        postgres_dsn=_env("FIL_POSTGRES_DSN"),
    )
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return load_settings()


def reset_settings_cache() -> None:
    """Drop the cached settings - used by tests that patch the environment."""
    get_settings.cache_clear()
