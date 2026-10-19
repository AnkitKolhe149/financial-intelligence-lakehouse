"""Data quality: scores, failed checks, quarantine and schema evolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.common import (
    configure_page,
    empty_state,
    load_failed_checks,
    load_gold,
    load_quarantine,
    load_silver,
    metric_row,
    number,
    page_header,
    percent,
    require_lake,
    sidebar_footer,
    status_badge,
)

configure_page("Data Quality")

DIMENSIONS = ["completeness", "validity", "uniqueness", "integrity", "freshness", "volume"]


def main() -> None:
    page_header("Data Quality", "Six-dimension scoring, failed checks and quarantined records")
    if not require_lake():
        sidebar_footer()
        return

    summary = load_gold("gold_data_quality_summary")
    trend = load_gold("gold_data_quality_trend")
    quarantine = load_quarantine()
    failed = load_failed_checks()

    _headline(summary, quarantine)
    st.divider()

    tabs = st.tabs(["Scorecard", "Trend", "Failed checks", "Quarantine", "Schema evolution"])
    with tabs[0]:
        _scorecard(summary)
    with tabs[1]:
        _trend(trend)
    with tabs[2]:
        _failed_checks(failed)
    with tabs[3]:
        _quarantine(quarantine)
    with tabs[4]:
        _schema_evolution()

    sidebar_footer()


def _headline(summary: pd.DataFrame, quarantine: pd.DataFrame) -> None:
    if summary.empty:
        empty_state("No quality results yet.")
        return

    weights = (
        pd.to_numeric(summary.get("records_evaluated"), errors="coerce").fillna(1).clip(lower=1)
    )
    scores = pd.to_numeric(summary["overall_score"], errors="coerce").fillna(0)
    overall = float((scores * weights).sum() / weights.sum())

    failing = int((summary["status"] == "FAIL").sum())
    warning = int((summary["status"] == "WARN").sum())

    metric_row(
        [
            ("Platform score", percent(overall, 2), None),
            ("Datasets scored", number(len(summary)), None),
            ("Failing", number(failing), None),
            ("Warning", number(warning), None),
            ("Quarantined (all time)", number(len(quarantine)), None),
        ]
    )
    st.caption(
        "The platform score is weighted by records evaluated, so the transaction fact "
        "table dominates it rather than being averaged away by small dimension tables."
    )


def _scorecard(summary: pd.DataFrame) -> None:
    if summary.empty:
        empty_state("No quality results yet.")
        return

    available = [d for d in DIMENSIONS if d in summary.columns]
    melted = summary.melt(
        id_vars=["dataset"],
        value_vars=available,
        var_name="dimension",
        value_name="score",
    ).dropna(subset=["score"])

    if not melted.empty:
        figure = px.imshow(
            melted.pivot(index="dataset", columns="dimension", values="score"),
            color_continuous_scale="RdYlGn",
            zmin=80,
            zmax=100,
            aspect="auto",
            labels={"color": "Score %"},
        )
        figure.update_layout(height=320, margin={"l": 0, "r": 0, "t": 20, "b": 0})
        st.plotly_chart(figure, width="stretch")

    for _, row in summary.sort_values("overall_score").iterrows():
        columns = st.columns([3, 2, 6])
        columns[0].markdown(f"**{row['dataset']}**")
        columns[1].markdown(status_badge(str(row.get("status", ""))), unsafe_allow_html=True)
        columns[2].progress(
            min(float(row["overall_score"]) / 100, 1.0),
            text=f"{float(row['overall_score']):.2f}%",
        )

    st.dataframe(summary, width="stretch", hide_index=True)


def _trend(trend: pd.DataFrame) -> None:
    if trend.empty:
        empty_state(
            "No quality history yet.",
            "The trend builds up as the pipeline runs repeatedly.",
        )
        return

    frame = trend.copy()
    frame["evaluated_date"] = pd.to_datetime(frame["evaluated_date"], errors="coerce")
    figure = px.line(
        frame.sort_values("evaluated_date"),
        x="evaluated_date",
        y="overall_score",
        color="dataset",
        markers=True,
        labels={"evaluated_date": "", "overall_score": "Overall score (%)"},
    )
    figure.update_layout(height=360, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        frame.sort_values("evaluated_date", ascending=False), width="stretch", hide_index=True
    )


def _failed_checks(failed: pd.DataFrame) -> None:
    if failed.empty:
        st.success("No failing checks in recent runs.")
        return

    st.caption(f"{len(failed):,} failing check result(s), most recent first")
    columns = [
        c
        for c in [
            "dataset",
            "check_name",
            "dimension",
            "column_name",
            "severity",
            "records_failed",
            "records_evaluated",
            "score",
            "message",
            "evaluated_at",
        ]
        if c in failed.columns
    ]
    st.dataframe(failed[columns], width="stretch", hide_index=True)

    blocking = (
        failed[failed["severity"] == "blocking"] if "severity" in failed.columns else pd.DataFrame()
    )
    if not blocking.empty:
        st.error(f"{len(blocking)} blocking check failure(s) - these would stop a strict-mode run.")


def _quarantine(quarantine: pd.DataFrame) -> None:
    if quarantine.empty:
        st.success("Nothing has been quarantined.")
        return

    left, right = st.columns(2)
    with left:
        counts = quarantine["error_code"].value_counts().reset_index()
        counts.columns = ["error_code", "records"]
        figure = px.bar(
            counts,
            x="records",
            y="error_code",
            orientation="h",
            labels={"records": "Records", "error_code": ""},
        )
        figure.update_layout(
            height=320,
            margin={"l": 0, "r": 0, "t": 20, "b": 0},
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        by_source = quarantine.groupby("source_system").size().reset_index(name="records")
        figure = px.pie(by_source, values="records", names="source_system", hole=0.45)
        figure.update_layout(height=320, margin={"l": 0, "r": 0, "t": 20, "b": 0})
        st.plotly_chart(figure, width="stretch")

    code = st.selectbox(
        "Inspect an error code", ["(all)", *sorted(quarantine["error_code"].dropna().unique())]
    )
    frame = quarantine if code == "(all)" else quarantine[quarantine["error_code"] == code]

    st.caption(
        f"{len(frame):,} record(s). Nothing is ever deleted - every rejection is inspectable."
    )
    columns = [
        c
        for c in [
            "record_id",
            "error_code",
            "error_message",
            "error_field",
            "source_system",
            "source_file",
            "detected_at",
            "pipeline_run_id",
        ]
        if c in frame.columns
    ]
    st.dataframe(frame[columns].head(300), width="stretch", hide_index=True)

    if not frame.empty:
        with st.expander("Raw rejected record"):
            st.code(str(frame.iloc[0]["raw_record"])[:2000], language="json")


def _schema_evolution() -> None:
    transactions = load_silver(
        "silver_transactions", columns=["_schema_version", "merchant_category"]
    )
    if transactions.empty:
        empty_state("silver_transactions is not available.")
        return

    grouped = (
        transactions.assign(has_category=transactions["merchant_category"].notna())
        .groupby("_schema_version")
        .agg(records=("has_category", "size"), with_merchant_category=("has_category", "sum"))
        .reset_index()
    )
    grouped["populated_pct"] = (grouped["with_merchant_category"] / grouped["records"] * 100).round(
        2
    )

    st.markdown("""
        The transaction feed arrives in two schema versions. Version 1.1 added
        `merchant_category`; version 1.0 records predate the column and carry a
        null. Both live in the same Silver table with no migration and no
        backfill - Bronze absorbed the new field on read, and Silver back-fills
        nulls for records written before it existed.
        """)
    st.dataframe(grouped, width="stretch", hide_index=True)

    figure = px.bar(
        grouped,
        x="_schema_version",
        y="records",
        color="_schema_version",
        labels={"_schema_version": "Schema version", "records": "Records"},
    )
    figure.update_layout(height=280, margin={"l": 0, "r": 0, "t": 20, "b": 0}, showlegend=False)
    st.plotly_chart(figure, width="stretch")


main()
