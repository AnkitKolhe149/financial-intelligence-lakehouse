"""Transaction investigation: find a transaction and see why it was scored."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.common import (
    configure_page,
    empty_state,
    load_gold,
    load_lineage,
    load_quarantine,
    load_silver,
    money,
    page_header,
    parse_reasons,
    require_lake,
    risk_badge,
    sidebar_footer,
)

configure_page("Transaction Investigation")


def main() -> None:
    page_header(
        "Transaction Investigation",
        "Search the fact table, inspect risk explanations, and trace a record to its source",
    )
    if not require_lake():
        sidebar_footer()
        return

    transactions = load_silver("silver_transactions")
    anomalies = load_gold("gold_anomaly_transactions")

    if transactions.empty:
        empty_state("silver_transactions is not available.")
        sidebar_footer()
        return

    tabs = st.tabs(["Search", "Single transaction", "Quarantined records"])
    with tabs[0]:
        _search(transactions, anomalies)
    with tabs[1]:
        _detail(transactions, anomalies)
    with tabs[2]:
        _quarantine_lookup()

    sidebar_footer()


def _search(transactions: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    st.subheader("Filter transactions")

    row1 = st.columns(4)
    account = row1[0].text_input("Account ID", placeholder="ACC-0000123")
    vendor = row1[1].text_input("Vendor ID", placeholder="VEN-000123")
    agency = row1[2].text_input("Agency ID", placeholder="AGY-0001")
    transaction_id = row1[3].text_input("Transaction ID", placeholder="TXN-000000000001")

    row2 = st.columns(4)
    levels = row2[0].multiselect("Risk level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    minimum_amount = row2[1].number_input(
        "Minimum amount (USD)", min_value=0.0, value=0.0, step=100.0
    )
    currency = row2[2].selectbox(
        "Currency", ["(any)", *sorted(transactions["currency"].dropna().unique().tolist())]
    )
    only_flagged = row2[3].checkbox("Flagged only", value=False)

    frame = transactions.copy()
    if account:
        frame = frame[frame["account_id"].astype(str).str.contains(account, case=False, na=False)]
    if vendor:
        frame = frame[frame["vendor_id"].astype(str).str.contains(vendor, case=False, na=False)]
    if agency:
        frame = frame[frame["agency_id"].astype(str).str.contains(agency, case=False, na=False)]
    if transaction_id:
        frame = frame[
            frame["transaction_id"].astype(str).str.contains(transaction_id, case=False, na=False)
        ]
    if minimum_amount > 0:
        frame = frame[pd.to_numeric(frame["amount_usd"], errors="coerce") >= minimum_amount]
    if currency != "(any)":
        frame = frame[frame["currency"] == currency]

    if not anomalies.empty:
        risk = anomalies[["transaction_id", "risk_score", "risk_level", "is_anomaly"]]
        frame = frame.merge(risk, on="transaction_id", how="left")
        frame["risk_score"] = frame["risk_score"].fillna(0).astype(int)
        frame["risk_level"] = frame["risk_level"].fillna("LOW")
        if levels:
            frame = frame[frame["risk_level"].isin(levels)]
        if only_flagged:
            frame = frame[frame["risk_score"] > 0]

    st.caption(f"{len(frame):,} matching transaction(s)")
    columns = [
        c
        for c in [
            "transaction_id",
            "transaction_timestamp",
            "account_id",
            "vendor_id",
            "agency_id",
            "amount",
            "currency",
            "amount_usd",
            "payment_method",
            "merchant_category",
            "risk_score",
            "risk_level",
            "_source_system",
        ]
        if c in frame.columns
    ]
    st.dataframe(
        frame[columns]
        .sort_values(
            "risk_score" if "risk_score" in frame.columns else "amount_usd", ascending=False
        )
        .head(300),
        width="stretch",
        hide_index=True,
    )


def _detail(transactions: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    st.subheader("Transaction detail")

    default = ""
    if not anomalies.empty:
        default = str(anomalies.nlargest(1, "risk_score").iloc[0]["transaction_id"])

    transaction_id = st.text_input(
        "Transaction ID", value=default, help="Defaults to the highest-scoring transaction"
    )
    if not transaction_id:
        return

    match = transactions[transactions["transaction_id"].astype(str) == transaction_id]
    if match.empty:
        st.error(f"Transaction {transaction_id} not found in Silver.")
        st.caption("It may have been quarantined - check the Quarantined records tab.")
        return

    record = match.iloc[0]
    scored = anomalies[anomalies["transaction_id"].astype(str) == transaction_id]

    header = st.columns([2, 1, 1, 1])
    header[0].markdown(f"### `{transaction_id}`")
    header[1].metric("Amount", money(record.get("amount_usd"), 2))
    if not scored.empty:
        detail = scored.iloc[0]
        header[2].metric("Risk score", int(detail.get("risk_score") or 0))
        header[3].markdown(
            f"<div style='padding-top:1.6rem'>{risk_badge(str(detail.get('risk_level', 'LOW')))}</div>",
            unsafe_allow_html=True,
        )
    else:
        header[2].metric("Risk score", 0)
        header[3].markdown(
            f"<div style='padding-top:1.6rem'>{risk_badge('LOW')}</div>", unsafe_allow_html=True
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Transaction**")
        st.write(
            {
                "timestamp": str(record.get("transaction_timestamp")),
                "posted": str(record.get("posted_timestamp")),
                "account": record.get("account_id"),
                "vendor": record.get("vendor_id"),
                "contract": record.get("contract_id"),
                "agency": record.get("agency_id"),
                "department": record.get("department_id"),
                "amount": f"{record.get('amount')} {record.get('currency')}",
                "amount_usd": record.get("amount_usd"),
                "payment_method": record.get("payment_method"),
                "merchant_category": record.get("merchant_category"),
                "location": record.get("location_id"),
            }
        )
    with right:
        st.markdown("**Provenance**")
        st.write(
            {
                "source_system": record.get("_source_system"),
                "record_hash": str(record.get("_record_hash"))[:24] + "...",
                "ingestion_run": record.get("_ingestion_run_id"),
                "silver_run": record.get("_silver_run_id"),
                "schema_version": record.get("_schema_version"),
                "late_arriving": bool(record.get("_is_late_arriving", False)),
                "duplicate_count": record.get("_duplicate_count"),
            }
        )

    st.markdown("**Why this score**")
    if scored.empty:
        st.info("No rules fired and the model did not flag this transaction.")
    else:
        detail = scored.iloc[0]
        reasons = parse_reasons(detail.get("risk_reasons"))
        rules = parse_reasons(detail.get("risk_rule_hits"))
        if reasons:
            for reason, rule in zip(reasons, rules + [""] * len(reasons), strict=False):
                st.markdown(
                    f"- {reason}  \n  <small><code>{rule}</code></small>", unsafe_allow_html=True
                )
        else:
            st.caption("No rule explanations recorded.")

        st.caption(
            f"Anomaly score {float(detail.get('anomaly_score') or 0):.4f} "
            f"from model `{detail.get('model_version')}`"
        )

    st.markdown("**Lineage**")
    edges = load_lineage()
    if edges.empty:
        st.caption("No lineage recorded.")
    else:
        chain = edges[edges["target_dataset"] == "silver_transactions"]
        sources = sorted(set(chain["source_dataset"]))
        st.markdown(
            " -> ".join(["`" + s + "`" for s in sources[:3]])
            + " -> `silver_transactions` -> `gold_anomaly_transactions`"
        )


def _quarantine_lookup() -> None:
    quarantine = load_quarantine()
    if quarantine.empty:
        st.success("Nothing has been quarantined.")
        return

    search = st.text_input("Search the raw payload", placeholder="TXN-000000000055")
    frame = quarantine
    if search:
        frame = quarantine[quarantine["raw_record"].astype(str).str.contains(search, na=False)]

    st.caption(f"{len(frame):,} quarantined record(s)")
    columns = [
        c
        for c in [
            "error_code",
            "error_message",
            "error_field",
            "source_system",
            "source_file",
            "detected_at",
        ]
        if c in frame.columns
    ]
    st.dataframe(frame[columns].head(200), width="stretch", hide_index=True)

    if not frame.empty:
        with st.expander("Raw record as received"):
            st.code(str(frame.iloc[0]["raw_record"])[:2000], language="json")
        st.caption(
            "The record is preserved exactly as the source sent it, so it can be "
            "replayed once the underlying problem is fixed."
        )


main()
