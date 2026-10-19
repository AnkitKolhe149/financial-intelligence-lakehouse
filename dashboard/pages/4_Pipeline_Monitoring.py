"""Pipeline monitoring: runs, stages, throughput and the dataset catalogue."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import (
    configure_page,
    empty_state,
    load_catalog,
    load_lineage,
    load_pipeline_runs,
    metric_row,
    number,
    page_header,
    percent,
    sidebar_footer,
)
from src.core.config import get_settings
from src.metadata.store import MetadataStore

configure_page("Pipeline Monitoring")


def main() -> None:
    page_header("Pipeline Monitoring", "Execution history, throughput, lineage and the catalogue")

    runs = load_pipeline_runs(limit=100)
    if runs.empty:
        empty_state(
            "No pipeline runs recorded yet.",
            "Run `python -m src.pipelines.batch_pipeline` to produce run metadata.",
        )
        sidebar_footer()
        return

    _headline(runs)
    st.divider()

    tabs = st.tabs(["Run history", "Stage timings", "Throughput", "Lineage", "Catalogue"])
    with tabs[0]:
        _run_history(runs)
    with tabs[1]:
        _stage_timings(runs)
    with tabs[2]:
        _throughput(runs)
    with tabs[3]:
        _lineage()
    with tabs[4]:
        _catalogue()

    sidebar_footer()


def _headline(runs: pd.DataFrame) -> None:
    successful = int(runs["status"].isin(["SUCCESS", "SUCCESS_WITH_WARNINGS"]).sum())
    failed = int((runs["status"] == "FAILED").sum())
    success_rate = successful / len(runs) * 100 if len(runs) else 0.0
    mean_duration = float(pd.to_numeric(runs["duration_seconds"], errors="coerce").mean() or 0)
    total_written = int(pd.to_numeric(runs["records_written"], errors="coerce").fillna(0).sum())
    total_quarantined = int(
        pd.to_numeric(runs["records_quarantined"], errors="coerce").fillna(0).sum()
    )

    metric_row(
        [
            ("Runs recorded", number(len(runs)), None),
            ("Success rate", percent(success_rate), None),
            ("Failed", number(failed), None),
            ("Mean duration", f"{mean_duration:.1f}s", None),
            ("Records written", number(total_written), None),
            ("Quarantined", number(total_quarantined), None),
        ]
    )


def _run_history(runs: pd.DataFrame) -> None:
    columns = [
        c
        for c in [
            "run_id",
            "pipeline_name",
            "pipeline_version",
            "engine",
            "source",
            "start_time",
            "duration_seconds",
            "records_read",
            "records_written",
            "records_quarantined",
            "status",
        ]
        if c in runs.columns
    ]
    st.dataframe(runs[columns], width="stretch", hide_index=True)

    failures = runs[runs["status"] == "FAILED"]
    if not failures.empty:
        st.error(f"{len(failures)} failed run(s)")
        for _, run in failures.head(5).iterrows():
            with st.expander(f"{run['run_id']} - {run.get('error_message', '')[:80]}"):
                st.code(str(run.get("error_message", "")), language="text")


def _stage_timings(runs: pd.DataFrame) -> None:
    run_id = st.selectbox("Run", runs["run_id"].tolist())
    store = MetadataStore(get_settings())
    stages = store.run_stages(run_id)

    if stages.empty:
        empty_state("No stage timings recorded for this run.")
        return

    figure = px.bar(
        stages,
        x="duration_seconds",
        y="stage",
        orientation="h",
        color="status",
        labels={"duration_seconds": "Seconds", "stage": ""},
        color_discrete_map={"SUCCESS": "#15803d", "FAILED": "#b91c1c"},
    )
    figure.update_layout(
        height=320,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        yaxis={
            "categoryorder": "array",
            "categoryarray": stages.sort_values("sequence", ascending=False)["stage"].tolist(),
        },
    )
    st.plotly_chart(figure, width="stretch")

    st.dataframe(
        stages[["sequence", "stage", "duration_seconds", "records_in", "records_out", "status"]],
        width="stretch",
        hide_index=True,
    )

    total = float(pd.to_numeric(stages["duration_seconds"], errors="coerce").sum())
    if total:
        slowest = stages.loc[pd.to_numeric(stages["duration_seconds"], errors="coerce").idxmax()]
        st.caption(
            f"Slowest stage: **{slowest['stage']}** at {float(slowest['duration_seconds']):.2f}s "
            f"({float(slowest['duration_seconds']) / total * 100:.0f}% of the run)."
        )


def _throughput(runs: pd.DataFrame) -> None:
    frame = runs.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["duration_seconds"] = pd.to_numeric(frame["duration_seconds"], errors="coerce")
    frame["records_written"] = pd.to_numeric(frame["records_written"], errors="coerce")
    frame["records_per_second"] = (
        frame["records_written"] / frame["duration_seconds"].replace(0, pd.NA)
    ).round(1)
    frame = frame.dropna(subset=["start_time"]).sort_values("start_time")

    figure = px.line(
        frame,
        x="start_time",
        y="records_per_second",
        markers=True,
        labels={"start_time": "", "records_per_second": "Records per second"},
    )
    figure.update_layout(height=300, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(figure, width="stretch")

    st.caption(
        "Duration alone is misleading - it should grow with volume. Records per second "
        "is what separates 'more data' from a performance regression."
    )

    duration = px.scatter(
        frame,
        x="records_read",
        y="duration_seconds",
        color="status",
        labels={"records_read": "Records read", "duration_seconds": "Duration (s)"},
    )
    duration.update_layout(height=300, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(duration, width="stretch")


def _lineage() -> None:
    edges = load_lineage()
    if edges.empty:
        empty_state("No lineage recorded yet.")
        return

    datasets = sorted(set(edges["target_dataset"]) | set(edges["source_dataset"]))
    selected = st.selectbox(
        "Dataset",
        datasets,
        index=datasets.index("silver_transactions") if "silver_transactions" in datasets else 0,
    )

    upstream = edges[edges["target_dataset"] == selected]
    downstream = edges[edges["source_dataset"] == selected]

    left, right = st.columns(2)
    with left:
        st.markdown("**Upstream (feeds this dataset)**")
        if upstream.empty:
            st.caption("None - this is a source dataset.")
        else:
            for _, edge in upstream.iterrows():
                dropped = int(edge.get("records_in") or 0) - int(edge.get("records_out") or 0)
                st.markdown(
                    f"`{edge['source_dataset']}` -> **{int(edge.get('records_out') or 0):,}** rows"
                    + (f" ({dropped:,} dropped)" if dropped > 0 else "")
                )
                st.caption(", ".join(edge.get("transformations") or []))
    with right:
        st.markdown("**Downstream (built from this dataset)**")
        if downstream.empty:
            st.caption("None - this is a terminal dataset.")
        else:
            for _, edge in downstream.iterrows():
                st.markdown(f"-> `{edge['target_dataset']}`")

    st.dataframe(
        edges[
            [
                "source_dataset",
                "target_dataset",
                "records_in",
                "records_out",
                "pipeline_name",
                "created_at",
            ]
        ].sort_values("created_at", ascending=False),
        width="stretch",
        hide_index=True,
    )


def _catalogue() -> None:
    catalog = load_catalog()
    if catalog.empty:
        empty_state("The catalogue is empty.")
        return

    st.caption(
        "The local stand-in for Unity Catalog: every dataset with its owner, "
        "classification, row count, size and partitioning."
    )
    columns = [
        c
        for c in [
            "dataset",
            "layer",
            "owner",
            "classification",
            "schema_version",
            "row_count",
            "size_bytes",
            "partition_columns",
            "last_updated",
            "last_run_id",
        ]
        if c in catalog.columns
    ]
    st.dataframe(catalog[columns], width="stretch", hide_index=True)

    by_layer = (
        catalog.groupby("layer")
        .agg(
            datasets=("dataset", "count"),
            rows=("row_count", "sum"),
            bytes=("size_bytes", "sum"),
        )
        .reset_index()
    )
    st.dataframe(by_layer, width="stretch", hide_index=True)


main()
