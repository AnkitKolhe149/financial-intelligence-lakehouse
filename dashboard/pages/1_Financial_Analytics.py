"""Financial analytics: where the money goes."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import (
    configure_page,
    empty_state,
    load_gold,
    metric_row,
    money,
    number,
    page_header,
    percent,
    require_lake,
    sidebar_footer,
)

configure_page("Financial Analytics")


def main() -> None:
    page_header("Financial Analytics", "Spending patterns by time, agency, department and vendor")
    if not require_lake():
        sidebar_footer()
        return

    daily = load_gold("gold_daily_financial_summary")
    agencies = load_gold("gold_agency_spending")
    departments = load_gold("gold_department_spending")
    vendors = load_gold("gold_vendor_risk")

    _filters_and_headline(daily)
    st.divider()

    tabs = st.tabs(
        ["Over time", "By agency", "By department", "Vendor concentration", "Distribution"]
    )
    with tabs[0]:
        _over_time(daily)
    with tabs[1]:
        _by_agency(agencies)
    with tabs[2]:
        _by_department(departments)
    with tabs[3]:
        _vendor_concentration(vendors)
    with tabs[4]:
        _distribution()

    sidebar_footer()


def _filters_and_headline(daily: pd.DataFrame) -> None:
    if daily.empty:
        empty_state("No daily summary available.")
        return

    frame = daily.copy()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="coerce")
    frame = frame.dropna(subset=["transaction_date"])

    minimum = frame["transaction_date"].min().date()
    maximum = frame["transaction_date"].max().date()

    with st.sidebar:
        st.subheader("Filters")
        window = st.date_input(
            "Date range", value=(minimum, maximum), min_value=minimum, max_value=maximum
        )

    if isinstance(window, tuple) and len(window) == 2:
        start, end = window
        frame = frame[
            (frame["transaction_date"].dt.date >= start)
            & (frame["transaction_date"].dt.date <= end)
        ]

    st.session_state["financial_frame"] = frame

    metric_row(
        [
            ("Days", number(len(frame)), None),
            ("Transactions", number(frame["transaction_count"].sum()), None),
            ("Total value", money(frame["total_amount_usd"].sum()), None),
            ("Mean per day", money(frame["total_amount_usd"].mean()), None),
            ("Typical transaction", money(frame["median_transaction_usd"].median(), 2), None),
        ]
    )


def _over_time(daily: pd.DataFrame) -> None:
    frame = st.session_state.get("financial_frame", daily)
    if frame.empty:
        empty_state("No data in the selected range.")
        return

    granularity = st.radio(
        "Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True, key="granularity"
    )
    working = frame.copy().set_index("transaction_date")
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[granularity]

    resampled = (
        working.resample(rule)
        .agg({"transaction_count": "sum", "total_amount_usd": "sum", "anomaly_count": "sum"})
        .reset_index()
    )

    figure = px.line(
        resampled,
        x="transaction_date",
        y="total_amount_usd",
        markers=granularity != "Daily",
        labels={"transaction_date": "", "total_amount_usd": "Value (USD)"},
    )
    figure.update_layout(height=340, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(figure, width="stretch")

    volume = px.bar(
        resampled,
        x="transaction_date",
        y="transaction_count",
        labels={"transaction_date": "", "transaction_count": "Transactions"},
    )
    volume.update_layout(height=260, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(volume, width="stretch")


def _by_agency(agencies: pd.DataFrame) -> None:
    if agencies.empty:
        empty_state("gold_agency_spending has not been built.")
        return

    top = agencies.nlargest(15, "total_amount_usd")
    label = "agency_name" if "agency_name" in top.columns else "agency_id"

    figure = px.bar(
        top,
        x="total_amount_usd",
        y=label,
        orientation="h",
        color="anomaly_rate_pct",
        color_continuous_scale="Reds",
        labels={"total_amount_usd": "Total (USD)", label: "", "anomaly_rate_pct": "Anomaly %"},
    )
    figure.update_layout(
        height=460,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(figure, width="stretch")

    columns = [
        c
        for c in [
            label,
            "fiscal_year",
            "transaction_count",
            "total_amount_usd",
            "unique_vendors",
            "budget_utilisation_pct",
            "top_vendor_share_pct",
            "anomaly_rate_pct",
        ]
        if c in agencies.columns
    ]
    st.dataframe(
        agencies[columns].sort_values("total_amount_usd", ascending=False),
        width="stretch",
        hide_index=True,
    )


def _by_department(departments: pd.DataFrame) -> None:
    if departments.empty:
        empty_state("gold_department_spending has not been built.")
        return

    over_budget = departments[departments.get("is_over_budget", False).fillna(False)]
    if not over_budget.empty:
        st.warning(
            f"{len(over_budget)} department-year(s) exceeded their allocation "
            f"by {money(float((over_budget['total_amount_usd'] - over_budget['allocated_amount_usd']).sum()))} in total."
        )

    columns = [
        c
        for c in [
            "department_name",
            "department_id",
            "agency_id",
            "fiscal_year",
            "total_amount_usd",
            "allocated_amount_usd",
            "budget_utilisation_pct",
            "is_over_budget",
            "transaction_count",
            "anomaly_rate_pct",
        ]
        if c in departments.columns
    ]
    frame = departments[columns].sort_values("budget_utilisation_pct", ascending=False)
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "budget_utilisation_pct": st.column_config.ProgressColumn(
                "Budget used",
                format="%.1f%%",
                min_value=0,
                max_value=float(max(frame["budget_utilisation_pct"].max(), 100)),
            )
        },
    )


def _vendor_concentration(vendors: pd.DataFrame) -> None:
    if vendors.empty:
        empty_state("gold_vendor_risk has not been built.")
        return

    top = vendors.nlargest(20, "total_amount_usd").copy()
    total = float(vendors["total_amount_usd"].sum())
    top["cumulative_share"] = top["total_amount_usd"].cumsum() / total * 100

    st.caption(
        f"The top 20 vendors receive {percent(top['total_amount_usd'].sum() / total * 100)} "
        f"of all spending across {number(len(vendors))} vendors."
    )

    figure = px.bar(
        top,
        x="vendor_name" if "vendor_name" in top.columns else "vendor_id",
        y="total_amount_usd",
        color="risk_level",
        color_discrete_map={
            "LOW": "#15803d",
            "MEDIUM": "#ca8a04",
            "HIGH": "#ea580c",
            "CRITICAL": "#b91c1c",
        },
        labels={"total_amount_usd": "Total paid (USD)", "vendor_name": "", "vendor_id": ""},
    )
    figure.add_scatter(
        x=top["vendor_name"] if "vendor_name" in top.columns else top["vendor_id"],
        y=top["cumulative_share"] * total / 100 / 100,
        mode="lines+markers",
        name="Cumulative share",
        yaxis="y2",
    )
    figure.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        xaxis={"tickangle": -40},
        yaxis2={"overlaying": "y", "side": "right", "showgrid": False, "title": "Cumulative %"},
    )
    st.plotly_chart(figure, width="stretch")


def _distribution() -> None:
    from dashboard.common import load_silver

    transactions = load_silver(
        "silver_transactions", columns=["amount_usd", "currency", "payment_method"]
    )
    if transactions.empty:
        empty_state("silver_transactions is not available.")
        return

    amounts = pd.to_numeric(transactions["amount_usd"], errors="coerce").dropna()
    st.caption(
        "Transaction amounts are heavy-tailed, so the log scale is the honest view - "
        "a linear histogram would show one bar."
    )

    figure = px.histogram(amounts, nbins=60, log_y=True, labels={"value": "Amount (USD)"})
    figure.update_layout(height=340, margin={"l": 0, "r": 0, "t": 20, "b": 0}, showlegend=False)
    st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**By currency**")
        st.dataframe(
            transactions.groupby("currency")["amount_usd"]
            .agg(["count", "sum", "mean", "median"])
            .round(2)
            .reset_index(),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.markdown("**By payment method**")
        st.dataframe(
            transactions.groupby("payment_method")["amount_usd"]
            .agg(["count", "sum", "mean"])
            .round(2)
            .reset_index(),
            width="stretch",
            hide_index=True,
        )


main()
