"""Risk analytics: anomalies, risk bands, alerts and rule effectiveness."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.common import (
    active_model,
    configure_page,
    empty_state,
    load_gold,
    metric_row,
    money,
    number,
    page_header,
    parse_reasons,
    percent,
    require_lake,
    risk_badge,
    sidebar_footer,
)

configure_page("Risk Analytics")


def main() -> None:
    page_header("Risk Analytics", "Anomaly detection, risk scoring and the alert queue")
    if not require_lake():
        sidebar_footer()
        return

    anomalies = load_gold("gold_anomaly_transactions")
    alerts = load_gold("gold_risk_alerts")
    accounts = load_gold("gold_account_risk")
    vendors = load_gold("gold_vendor_risk")
    daily = load_gold("gold_daily_financial_summary")

    _headline(anomalies, alerts, accounts, vendors)
    st.divider()

    tabs = st.tabs(
        [
            "Alert queue",
            "Anomaly trend",
            "High-risk accounts",
            "High-risk vendors",
            "Rule effectiveness",
            "Model",
        ]
    )
    with tabs[0]:
        _alert_queue(alerts, anomalies)
    with tabs[1]:
        _anomaly_trend(daily, anomalies)
    with tabs[2]:
        _high_risk_accounts(accounts)
    with tabs[3]:
        _high_risk_vendors(vendors)
    with tabs[4]:
        _rule_effectiveness(anomalies)
    with tabs[5]:
        _model_panel(anomalies)

    sidebar_footer()


def _headline(
    anomalies: pd.DataFrame, alerts: pd.DataFrame, accounts: pd.DataFrame, vendors: pd.DataFrame
) -> None:
    critical = int((anomalies["risk_level"] == "CRITICAL").sum()) if not anomalies.empty else 0
    high = int((anomalies["risk_level"] == "HIGH").sum()) if not anomalies.empty else 0
    ml_flagged = int(anomalies["is_anomaly"].fillna(False).sum()) if not anomalies.empty else 0
    flagged_value = float(anomalies["amount_usd"].sum()) if not anomalies.empty else 0.0

    metric_row(
        [
            ("Flagged transactions", number(len(anomalies)), None),
            ("ML anomalies", number(ml_flagged), None),
            ("Critical", number(critical), None),
            ("High", number(high), None),
            (
                "Open alerts",
                number(int((alerts["status"] == "OPEN").sum()) if not alerts.empty else 0),
                None,
            ),
            ("Value under review", money(flagged_value), None),
        ]
    )


def _alert_queue(alerts: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    if alerts.empty:
        empty_state(
            "No alerts have been raised.", "Alerts are created above the configured risk threshold."
        )
        return

    left, right, third = st.columns(3)
    level = left.multiselect(
        "Risk level", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH"]
    )
    status = right.multiselect(
        "Status", ["OPEN", "REVIEWING", "RESOLVED", "FALSE_POSITIVE"], default=["OPEN"]
    )
    minimum = third.slider("Minimum risk score", 0, 100, 50)

    frame = alerts.copy()
    if level:
        frame = frame[frame["risk_level"].isin(level)]
    if status:
        frame = frame[frame["status"].isin(status)]
    frame = frame[pd.to_numeric(frame["risk_score"], errors="coerce").fillna(0) >= minimum]

    st.caption(f"{len(frame):,} alert(s) match the filter")
    columns = [
        c
        for c in [
            "alert_id",
            "transaction_id",
            "account_id",
            "vendor_id",
            "amount_usd",
            "risk_score",
            "risk_level",
            "reason_count",
            "status",
            "reason",
        ]
        if c in frame.columns
    ]
    st.dataframe(
        frame[columns].sort_values("risk_score", ascending=False).head(200),
        width="stretch",
        hide_index=True,
    )

    if not frame.empty:
        st.markdown("**Top case detail**")
        top = frame.nlargest(1, "risk_score").iloc[0]
        detail = anomalies[anomalies["transaction_id"] == top["transaction_id"]]
        st.markdown(
            f"Transaction `{top['transaction_id']}` - {money(top['amount_usd'], 2)} - "
            f"score {int(top['risk_score'])} {risk_badge(str(top['risk_level']))}",
            unsafe_allow_html=True,
        )
        if not detail.empty:
            for reason in parse_reasons(detail.iloc[0].get("risk_reasons")):
                st.markdown(f"- {reason}")


def _anomaly_trend(daily: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    if daily.empty:
        empty_state("No daily summary available.")
        return

    frame = daily.copy()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="coerce")
    frame = frame.dropna(subset=["transaction_date"]).sort_values("transaction_date")

    figure = px.line(
        frame,
        x="transaction_date",
        y="anomaly_rate_pct",
        labels={"transaction_date": "", "anomaly_rate_pct": "Anomaly rate (%)"},
    )
    figure.update_layout(height=320, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(figure, width="stretch")

    if not anomalies.empty:
        st.markdown("**Amount distribution of flagged transactions**")
        amounts = pd.to_numeric(anomalies["amount_usd"], errors="coerce").dropna()
        histogram = px.histogram(amounts, nbins=50, log_y=True, labels={"value": "Amount (USD)"})
        histogram.update_layout(
            height=280, margin={"l": 0, "r": 0, "t": 20, "b": 0}, showlegend=False
        )
        st.plotly_chart(histogram, width="stretch")


def _high_risk_accounts(accounts: pd.DataFrame) -> None:
    if accounts.empty:
        empty_state("gold_account_risk has not been built.")
        return

    risky = accounts[accounts["risk_level"].isin(["HIGH", "CRITICAL"])]
    st.caption(
        f"{len(risky):,} of {len(accounts):,} accounts are high or critical risk "
        f"({percent(len(risky) / len(accounts) * 100 if len(accounts) else 0)})"
    )
    columns = [
        c
        for c in [
            "account_id",
            "account_type",
            "region",
            "transaction_count",
            "total_amount_usd",
            "unique_vendors",
            "unique_locations",
            "anomaly_count",
            "high_risk_count",
            "risk_score",
            "risk_level",
        ]
        if c in accounts.columns
    ]
    st.dataframe(
        (risky if not risky.empty else accounts)[columns]
        .sort_values("risk_score", ascending=False)
        .head(200),
        width="stretch",
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk", format="%d", min_value=0, max_value=100
            )
        },
    )


def _high_risk_vendors(vendors: pd.DataFrame) -> None:
    if vendors.empty:
        empty_state("gold_vendor_risk has not been built.")
        return

    columns = [
        c
        for c in [
            "vendor_id",
            "vendor_name",
            "transaction_count",
            "total_amount_usd",
            "unique_accounts",
            "unique_agencies",
            "contract_count",
            "uncontracted_payment_pct",
            "growth_ratio",
            "shares_tax_id_with",
            "risk_score",
            "risk_level",
        ]
        if c in vendors.columns
    ]
    st.dataframe(
        vendors[columns].sort_values("risk_score", ascending=False).head(200),
        width="stretch",
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk", format="%d", min_value=0, max_value=100
            )
        },
    )

    duplicates = (
        vendors[vendors.get("shares_tax_id_with", 0) > 0]
        if "shares_tax_id_with" in vendors.columns
        else pd.DataFrame()
    )
    if not duplicates.empty:
        st.warning(
            f"{len(duplicates)} vendor record(s) share a tax id with another registration - "
            "possible duplicate master data. Surfaced for review, never merged automatically."
        )


def _rule_effectiveness(anomalies: pd.DataFrame) -> None:
    if anomalies.empty or "risk_rule_hits" not in anomalies.columns:
        empty_state("No scored transactions available.")
        return

    counts: dict[str, int] = {}
    value: dict[str, float] = {}
    for _, row in anomalies.iterrows():
        for rule in parse_reasons(row.get("risk_rule_hits")):
            counts[rule] = counts.get(rule, 0) + 1
            value[rule] = value.get(rule, 0.0) + float(row.get("amount_usd") or 0)

    if not counts:
        empty_state("No rules fired on the flagged population.")
        return

    frame = pd.DataFrame(
        {
            "rule": list(counts),
            "transactions": list(counts.values()),
            "value_usd": [round(value[r], 2) for r in counts],
        }
    ).sort_values("transactions", ascending=False)

    figure = px.bar(
        frame,
        x="transactions",
        y="rule",
        orientation="h",
        labels={"transactions": "Transactions flagged", "rule": ""},
    )
    figure.update_layout(
        height=340,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(figure, width="stretch")
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption(
        "A rule that never fires is dead configuration; one that fires on everything is noise. "
        "Both are visible here."
    )


def _model_panel(anomalies: pd.DataFrame) -> None:
    model = active_model()
    if not model:
        empty_state("No model has been registered yet.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Active model**")
        st.write(
            {
                "version": model.get("model_version"),
                "algorithm": model.get("algorithm"),
                "trained_at": str(model.get("trained_at")),
                "training_rows": model.get("training_rows"),
                "features": len(model.get("feature_list") or []),
            }
        )
    with right:
        st.markdown("**Held-out evaluation**")
        evaluation = model.get("evaluation") or {}
        if isinstance(evaluation, dict) and evaluation:
            st.write(
                {
                    "records_scored": evaluation.get("records_scored"),
                    "known_fraud_in_scope": evaluation.get("known_fraud_in_scope"),
                    "pr_auc": evaluation.get("pr_auc"),
                    "roc_auc": evaluation.get("roc_auc"),
                    "precision_at_k": evaluation.get("precision_at_k"),
                    "lift_at_k": evaluation.get("lift_at_k"),
                }
            )
            st.caption(
                evaluation.get("metric_note")
                or "Accuracy is not reported: on a sub-1% base rate it is meaningless."
            )
        else:
            st.caption("No evaluation recorded for this model version.")

    if not anomalies.empty and "anomaly_score" in anomalies.columns:
        st.markdown("**Anomaly score distribution (flagged population)**")
        scores = pd.to_numeric(anomalies["anomaly_score"], errors="coerce").dropna()
        figure = px.histogram(scores, nbins=40, labels={"value": "Anomaly score"})
        figure.update_layout(height=260, margin={"l": 0, "r": 0, "t": 20, "b": 0}, showlegend=False)
        st.plotly_chart(figure, width="stretch")


main()
