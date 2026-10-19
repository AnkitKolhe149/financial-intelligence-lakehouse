"""Financial Intelligence Lakehouse - dashboard entry point (Overview).

    streamlit run dashboard/app.py

The overview answers the question an oversight lead asks first: *is the platform
working, and is anything on fire?* Everything else lives on a dedicated page.

Note the deliberate ordering of the tiles: data volume, then risk, then **data
quality last but not least prominent**. A dashboard that shows anomaly counts
without showing whether the underlying data can be trusted invites people to act
on numbers whose provenance nobody checked.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import (
    active_model,
    configure_page,
    load_gold,
    load_pipeline_runs,
    metric_row,
    money,
    number,
    page_header,
    percent,
    require_lake,
    sidebar_footer,
    status_badge,
)

configure_page("Overview")


def main() -> None:
    page_header(
        "Financial Intelligence Lakehouse",
        "Enterprise data platform for financial oversight, anomaly detection and data quality",
    )

    if not require_lake():
        sidebar_footer()
        return

    daily = load_gold("gold_daily_financial_summary")
    accounts = load_gold("gold_account_risk")
    vendors = load_gold("gold_vendor_risk")
    alerts = load_gold("gold_risk_alerts")
    quality = load_gold("gold_data_quality_summary")
    runs = load_pipeline_runs(limit=10)

    _headline_metrics(daily, accounts, vendors, alerts, quality)
    st.divider()

    left, right = st.columns([2, 1])
    with left:
        _spending_chart(daily)
    with right:
        _risk_breakdown(alerts, accounts, vendors)

    st.divider()
    lower_left, lower_right = st.columns(2)
    with lower_left:
        _quality_panel(quality)
    with lower_right:
        _pipeline_panel(runs)

    st.divider()
    _platform_summary()
    sidebar_footer()


def _headline_metrics(
    daily: pd.DataFrame,
    accounts: pd.DataFrame,
    vendors: pd.DataFrame,
    alerts: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    total_transactions = int(daily["transaction_count"].sum()) if not daily.empty else 0
    total_value = float(daily["total_amount_usd"].sum()) if not daily.empty else 0.0
    anomalies = int(daily["anomaly_count"].sum()) if not daily.empty else 0
    high_risk_accounts = (
        int(accounts["risk_level"].isin(["HIGH", "CRITICAL"]).sum()) if not accounts.empty else 0
    )
    high_risk_vendors = (
        int(vendors["risk_level"].isin(["HIGH", "CRITICAL"]).sum()) if not vendors.empty else 0
    )

    if not quality.empty:
        weights = (
            pd.to_numeric(quality.get("records_evaluated"), errors="coerce").fillna(1).clip(lower=1)
        )
        scores = pd.to_numeric(quality["overall_score"], errors="coerce").fillna(0)
        quality_score = float((scores * weights).sum() / weights.sum())
    else:
        quality_score = 0.0

    anomaly_rate = (anomalies / total_transactions * 100) if total_transactions else 0.0

    metric_row(
        [
            ("Transactions", number(total_transactions), None),
            ("Financial volume", money(total_value), None),
            ("Anomalies", number(anomalies), percent(anomaly_rate, 2)),
            ("High-risk accounts", number(high_risk_accounts), None),
            ("High-risk vendors", number(high_risk_vendors), None),
            ("Data quality", percent(quality_score, 2), None),
        ]
    )

    if not alerts.empty:
        open_alerts = int((alerts["status"] == "OPEN").sum())
        critical = int((alerts["risk_level"] == "CRITICAL").sum())
        if critical:
            st.error(
                f"{critical} critical alert(s) awaiting review, {open_alerts} open in total. "
                "See the Risk Analytics page."
            )
        elif open_alerts:
            st.warning(f"{open_alerts} open alert(s) awaiting review.")


def _spending_chart(daily: pd.DataFrame) -> None:
    st.subheader("Spending over time")
    if daily.empty:
        st.info("No daily summary available.")
        return

    frame = daily.copy()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="coerce")
    frame = frame.dropna(subset=["transaction_date"]).sort_values("transaction_date")

    figure = px.area(
        frame,
        x="transaction_date",
        y="total_amount_usd",
        labels={"transaction_date": "", "total_amount_usd": "Daily value (USD)"},
    )
    if "rolling_7d_avg_amount_usd" in frame.columns:
        figure.add_scatter(
            x=frame["transaction_date"],
            y=frame["rolling_7d_avg_amount_usd"],
            mode="lines",
            name="7-day average",
            line={"dash": "dot"},
        )
    figure.update_layout(height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0}, showlegend=True)
    st.plotly_chart(figure, width="stretch")


def _risk_breakdown(alerts: pd.DataFrame, accounts: pd.DataFrame, vendors: pd.DataFrame) -> None:
    st.subheader("Risk distribution")
    if accounts.empty:
        st.info("No risk profiles available.")
        return

    counts = (
        accounts["risk_level"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
    )
    figure = px.bar(
        x=counts.index,
        y=counts.to_numpy(),
        labels={"x": "", "y": "Accounts"},
        color=counts.index,
        color_discrete_map={
            "LOW": "#15803d",
            "MEDIUM": "#ca8a04",
            "HIGH": "#ea580c",
            "CRITICAL": "#b91c1c",
        },
    )
    figure.update_layout(height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0}, showlegend=False)
    st.plotly_chart(figure, width="stretch")


def _quality_panel(quality: pd.DataFrame) -> None:
    st.subheader("Data quality by dataset")
    if quality.empty:
        st.info("No quality results yet.")
        return

    columns = [
        c
        for c in [
            "dataset",
            "overall_score",
            "completeness",
            "validity",
            "uniqueness",
            "integrity",
            "status",
            "quarantined_records",
        ]
        if c in quality.columns
    ]
    frame = quality[columns].copy().sort_values("overall_score")
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "overall_score": st.column_config.ProgressColumn(
                "Overall", format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )


def _pipeline_panel(runs: pd.DataFrame) -> None:
    st.subheader("Recent pipeline runs")
    if runs.empty:
        st.info("No pipeline runs recorded yet.")
        return

    for _, run in runs.head(5).iterrows():
        columns = st.columns([3, 2, 2, 2])
        columns[0].markdown(f"`{run['run_id']}`")
        columns[1].markdown(status_badge(str(run["status"])), unsafe_allow_html=True)
        columns[2].caption(f"{float(run.get('duration_seconds') or 0):.1f}s")
        columns[3].caption(f"{int(run.get('records_written') or 0):,} written")


def _platform_summary() -> None:
    st.subheader("Platform")
    model = active_model()
    columns = st.columns(4)

    columns[0].markdown("**Architecture**")
    columns[0].caption(
        "Bronze -> Silver -> Gold lakehouse with quarantine, lineage and a metadata control plane."
    )

    columns[1].markdown("**Sources**")
    columns[1].caption(
        "CSV, JSON, XML, Excel, REST API and a file-based event stream, plus USAspending.gov."
    )

    columns[2].markdown("**Detection**")
    columns[2].caption(
        f"Isolation Forest ({model.get('model_version', 'not trained')}) combined with "
        "explainable deterministic rules."
    )

    columns[3].markdown("**Governance**")
    columns[3].caption(
        "Data contracts, six-dimension quality scoring, per-run lineage and a dataset catalogue."
    )


main()
