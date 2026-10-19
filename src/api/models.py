"""Response models for the serving API.

Typed responses are not decoration. They give the endpoint a published contract,
generate accurate OpenAPI documentation at ``/docs``, and - most usefully -
fail loudly if a Gold column is renamed and the API silently starts returning a
different shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = Field(description="healthy | degraded")
    service: str
    version: str
    timestamp: datetime
    engine: str
    metadata_backend: str
    lake: dict[str, bool] = Field(description="Which lake layers currently hold data")
    graph_available: bool


class MetricsResponse(BaseModel):
    """Operational metrics - the ``/metrics`` payload."""

    generated_at: datetime
    pipeline: dict[str, Any]
    data: dict[str, Any]
    quality: dict[str, Any]
    risk: dict[str, Any]
    model: dict[str, Any]


class TransactionDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    transaction_id: str
    account_id: str | None = None
    vendor_id: str | None = None
    contract_id: str | None = None
    agency_id: str | None = None
    department_id: str | None = None
    amount: float | None = None
    currency: str | None = None
    amount_usd: float | None = None
    transaction_timestamp: datetime | None = None
    risk_score: int = 0
    risk_level: str = "LOW"
    risk_reasons: list[str] = Field(default_factory=list)
    rules_fired: list[str] = Field(default_factory=list)
    anomaly_score: float | None = None
    is_anomaly: bool = False
    model_version: str | None = None
    source_system: str | None = None
    lineage: list[str] = Field(
        default_factory=list, description="Datasets this record flowed through"
    )


class AccountRiskResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    account_id: str
    transaction_count: int
    total_amount_usd: float
    average_amount_usd: float
    maximum_amount_usd: float
    unique_vendors: int
    unique_locations: int
    anomaly_count: int
    high_risk_count: int
    risk_score: int
    risk_level: str


class VendorRiskResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    vendor_id: str
    vendor_name: str | None = None
    transaction_count: int
    total_amount_usd: float
    unique_accounts: int
    unique_agencies: int
    anomaly_count: int
    contract_count: int
    risk_score: int
    risk_level: str


class AnomalyRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    transaction_id: str
    account_id: str | None = None
    vendor_id: str | None = None
    amount_usd: float | None = None
    transaction_timestamp: datetime | None = None
    anomaly_score: float | None = None
    risk_score: int = 0
    risk_level: str = "LOW"
    risk_reasons: list[str] = Field(default_factory=list)
    model_version: str | None = None


class PaginatedResponse(BaseModel):
    """Envelope for list endpoints.

    The platform's own API paginates for the same reason the source system does:
    an endpoint that can return a million rows will eventually be asked to.
    """

    total: int
    limit: int
    offset: int
    returned: int
    results: list[dict[str, Any]]


class QualityResponse(BaseModel):
    dataset: str
    layer: str | None = None
    overall_score: float
    completeness: float | None = None
    validity: float | None = None
    uniqueness: float | None = None
    integrity: float | None = None
    freshness: float | None = None
    volume: float | None = None
    records_evaluated: int | None = None
    checks_total: int | None = None
    checks_passed: int | None = None
    status: str | None = None
    quarantined_records: int | None = None


class PipelineRunResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    pipeline_name: str
    pipeline_version: str | None = None
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None
    records_read: int | None = None
    records_written: int | None = None
    records_quarantined: int | None = None
    error_message: str | None = None


class LineageResponse(BaseModel):
    dataset: str
    upstream: list[str]
    downstream: list[str]
    edges: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    hint: str | None = None
