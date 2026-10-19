"""Shared helpers for the Streamlit dashboard.

Two things live here rather than in each page:

**Cached data access.** Streamlit re-runs the whole script on every widget
interaction. Without caching, moving a slider would re-read every Parquet file
in the lake. ``@st.cache_data`` with a TTL means interaction is instant while
the data still refreshes after a pipeline run.

**Consistent presentation.** Currency, percentages and risk colours are
formatted in one place, so the same number does not appear as ``1234.5`` on one
page and ``$1,234.50`` on another.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.core.config import Settings, get_settings
from src.core.io import dataset_exists, read_parquet_or_empty
from src.gold.builder import read_gold
from src.metadata.store import MetadataStore

CACHE_TTL = 60  # seconds

RISK_COLOURS = {
    "CRITICAL": "#b91c1c",
    "HIGH": "#ea580c",
    "MEDIUM": "#ca8a04",
    "LOW": "#15803d",
}

STATUS_COLOURS = {
    "SUCCESS": "#15803d",
    "SUCCESS_WITH_WARNINGS": "#ca8a04",
    "RUNNING": "#0284c7",
    "FAILED": "#b91c1c",
    "PASS": "#15803d",
    "WARN": "#ca8a04",
    "FAIL": "#b91c1c",
}


def settings() -> Settings:
    return get_settings()


def configure_page(title: str, icon: str = "*") -> None:
    """Apply the shared page configuration. Safe to call from every page."""
    try:
        st.set_page_config(
            page_title=f"{title} - Financial Intelligence Lakehouse",
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except st.errors.StreamlitAPIException:
        # set_page_config may only be called once per session; on reruns from a
        # sub-page it is already set, which is not an error worth surfacing.
        return


# ---------------------------------------------------------------------------
# cached data access
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_gold(table: str) -> pd.DataFrame:
    return read_gold(table, get_settings())


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_silver(table: str, columns: list[str] | None = None) -> pd.DataFrame:
    return read_parquet_or_empty(get_settings().dataset_path("silver", table), columns=columns)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_quarantine(limit: int = 5000) -> pd.DataFrame:
    from src.silver.quarantine import read_quarantine

    return read_quarantine("transaction", get_settings(), limit=limit)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_pipeline_runs(limit: int = 50) -> pd.DataFrame:
    return MetadataStore(get_settings()).recent_runs(limit=limit)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_quality_history(limit: int = 500) -> pd.DataFrame:
    return MetadataStore(get_settings()).quality_history(limit=limit)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_failed_checks(limit: int = 200) -> pd.DataFrame:
    return MetadataStore(get_settings()).failed_checks(limit=limit)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_catalog() -> pd.DataFrame:
    return MetadataStore(get_settings()).catalog()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_lineage() -> pd.DataFrame:
    return MetadataStore(get_settings()).all_lineage()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def active_model() -> dict[str, Any]:
    config = get_settings()
    return MetadataStore(config).active_model(config.ml.model_name) or {}


def lake_is_built() -> bool:
    return dataset_exists(get_settings().dataset_path("gold", "gold_daily_financial_summary"))


def require_lake() -> bool:
    """Render a helpful message instead of an exception when there is no data."""
    if lake_is_built():
        return True
    st.warning("The lake has not been built yet.")
    st.markdown("""
        Run the demo to generate data and populate every layer:

        ```bash
        python scripts/run_demo.py
        ```

        or run the pipeline directly:

        ```bash
        python -m src.pipelines.batch_pipeline
        ```
        """)
    return False


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def money(value: float | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.{decimals}f}"


def number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def percent(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}%"


def risk_badge(level: str) -> str:
    colour = RISK_COLOURS.get(str(level).upper(), "#6b7280")
    return (
        f"<span style='background:{colour};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.8rem;font-weight:600'>{level}</span>"
    )


def status_badge(status: str) -> str:
    colour = STATUS_COLOURS.get(str(status).upper(), "#6b7280")
    return (
        f"<span style='background:{colour};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.8rem;font-weight:600'>{status}</span>"
    )


def parse_reasons(value: Any) -> list[str]:
    """Risk reasons are stored as a JSON string in Gold."""
    import json

    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return [str(v) for v in parsed] if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return []


def sidebar_footer() -> None:
    """Shared sidebar showing platform state - visible on every page."""
    config = get_settings()
    with st.sidebar:
        st.divider()
        st.caption("Platform")
        runs = load_pipeline_runs(limit=1)
        if not runs.empty:
            last = runs.iloc[0]
            st.caption(f"Last run: `{last['run_id']}`")
            st.markdown(status_badge(str(last["status"])), unsafe_allow_html=True)
            st.caption(f"{pd.to_datetime(last['start_time']):%Y-%m-%d %H:%M}")
        else:
            st.caption("No pipeline runs recorded")

        st.caption(f"Engine: `{config.engine}`")
        st.caption(f"Storage: `{config.storage.format}`")

        try:
            from src.graph.neo4j_client import graph_available

            connected = graph_available(config)
        except Exception:
            connected = False
        st.caption(f"Graph: {'connected' if connected else 'fallback (in-process)'}")

        if st.button("Refresh data", width="stretch"):
            st.cache_data.clear()
            st.rerun()


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a row of metric tiles: ``(label, value, delta)``."""
    columns = st.columns(len(metrics))
    for column, (label, value, delta) in zip(columns, metrics, strict=True):
        with column:
            st.metric(label, value, delta)


def empty_state(message: str, hint: str = "") -> None:
    st.info(message)
    if hint:
        st.caption(hint)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
