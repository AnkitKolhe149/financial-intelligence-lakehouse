"""Graph investigation: relationship analysis, with a working fallback.

If Neo4j is running the page uses Cypher. If it is not, the same questions are
answered in-process from Parquet and the page says so plainly rather than
showing an error. The graph is an enrichment layer, and the dashboard is
expected to work without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.common import (
    configure_page,
    empty_state,
    load_gold,
    money,
    number,
    page_header,
    require_lake,
    sidebar_footer,
)
from src.core.config import get_settings

configure_page("Graph Investigation")


def main() -> None:
    page_header(
        "Graph Investigation",
        "Relationship analysis across accounts, vendors, contracts and agencies",
    )
    if not require_lake():
        sidebar_footer()
        return

    settings = get_settings()
    _backend_banner(settings)

    tabs = st.tabs(
        [
            "Vendor neighbourhood",
            "Shared vendors",
            "High-risk networks",
            "Award loops",
            "Duplicate registrations",
        ]
    )
    with tabs[0]:
        _vendor_neighbourhood(settings)
    with tabs[1]:
        _shared_vendors(settings)
    with tabs[2]:
        _high_risk_networks(settings)
    with tabs[3]:
        _award_loops(settings)
    with tabs[4]:
        _duplicates(settings)

    sidebar_footer()


def _backend_banner(settings) -> None:
    from src.graph.graph_queries import graph_statistics

    stats = graph_statistics(settings)
    if stats.get("available"):
        nodes = stats.get("nodes", {})
        relationships = stats.get("relationships", {})
        st.success(
            f"Neo4j connected - {sum(nodes.values()):,} nodes, "
            f"{sum(relationships.values()):,} relationships"
        )
        columns = st.columns(2)
        with columns[0]:
            st.dataframe(
                pd.DataFrame(nodes.items(), columns=["Node label", "Count"]),
                width="stretch",
                hide_index=True,
            )
        with columns[1]:
            st.dataframe(
                pd.DataFrame(relationships.items(), columns=["Relationship", "Count"]),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info(
            "Neo4j is not connected, so these questions are being answered in-process "
            "from the Parquet lake. The results are the same; the Cypher implementation "
            "is used automatically when Neo4j is available.\n\n"
            "Start it with `docker compose up -d neo4j`, then `make load-graph`."
        )


def _vendor_neighbourhood(settings) -> None:
    from src.graph.graph_queries import vendor_neighbourhood

    vendors = load_gold("gold_vendor_risk")
    if vendors.empty:
        empty_state("gold_vendor_risk has not been built.")
        return

    risky = vendors.nlargest(50, "risk_score")
    labels = {
        f"{row['vendor_id']} - {row.get('vendor_name', '')} "
        f"({row['risk_level']}, score {int(row['risk_score'])})": row["vendor_id"]
        for _, row in risky.iterrows()
    }

    left, right = st.columns([3, 1])
    choice = left.selectbox("Vendor", list(labels))
    depth = right.slider("Traversal depth", 1, 3, 2)
    vendor_id = labels[choice]

    result = vendor_neighbourhood(vendor_id, depth=depth, settings=settings)
    nodes = result.get("nodes", [])

    st.caption(
        f"Backend: `{result.get('backend', 'unknown')}` - "
        f"{result.get('node_count', 0)} connected node(s) within {depth} hop(s)"
    )
    if not nodes:
        empty_state(result.get("reason", "No connected nodes found."))
        return

    frame = pd.DataFrame(nodes)
    _render_network(vendor_id, frame)

    st.dataframe(frame, width="stretch", hide_index=True)
    summary = frame.groupby("node_type").size().reset_index(name="nodes")
    st.dataframe(summary, width="stretch", hide_index=True)


def _render_network(centre: str, frame: pd.DataFrame) -> None:
    """Draw a simple radial network: the centre vendor plus its neighbours.

    A force-directed layout would need an extra dependency; a radial layout by
    hop distance shows the same structure and keeps the page self-contained.
    """
    import math

    colours = {
        "Vendor": "#7c3aed",
        "Account": "#0284c7",
        "Transaction": "#64748b",
        "Agency": "#059669",
        "Contract": "#ca8a04",
        "Department": "#db2777",
        "Location": "#78716c",
    }

    display = frame.head(60).copy()
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            text=[centre],
            textposition="middle center",
            marker={"size": 34, "color": "#7c3aed"},
            name="Vendor (centre)",
            hoverinfo="text",
        )
    )

    for hop in sorted(display["hops"].unique()):
        ring = display[display["hops"] == hop]
        count = len(ring)
        radius = float(hop)
        xs, ys, texts, marker_colours = [], [], [], []
        for index, (_, node) in enumerate(ring.iterrows()):
            angle = 2 * math.pi * index / max(count, 1)
            xs.append(radius * math.cos(angle))
            ys.append(radius * math.sin(angle))
            texts.append(f"{node['node_type']}: {node['node_id']}")
            marker_colours.append(colours.get(str(node["node_type"]), "#94a3b8"))

        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                text=texts,
                hoverinfo="text",
                marker={"size": 14, "color": marker_colours},
                name=f"{hop} hop(s)",
            )
        )
        for x, y in zip(xs, ys, strict=True):
            figure.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=x,
                y1=y,
                line={"color": "rgba(148,163,184,0.25)", "width": 1},
            )

    figure.update_layout(
        height=460,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=True,
        hovermode="closest",
    )
    st.plotly_chart(figure, width="stretch")


def _shared_vendors(settings) -> None:
    from src.graph.graph_queries import shared_vendor_accounts

    limit = st.slider("Results", 5, 100, 25, key="shared_limit")
    result = shared_vendor_accounts(limit=limit, settings=settings)
    st.caption(f"Backend: `{result.get('backend')}`")

    rows = result.get("results", [])
    if not rows:
        empty_state(result.get("reason", "No account pairs share a vendor."))
        return

    st.markdown(
        "Accounts connected through a common supplier. Not suspicious in itself - "
        "shared suppliers are normal - but it is the structure that fraud rings use, "
        "and the starting point for expanding an investigation."
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _high_risk_networks(settings) -> None:
    from src.graph.graph_queries import high_risk_vendor_networks

    columns = st.columns(2)
    limit = columns[0].slider("Results", 5, 100, 25, key="network_limit")
    minimum = columns[1].slider("Minimum high-risk accounts", 2, 10, 2)

    result = high_risk_vendor_networks(limit=limit, min_accounts=minimum, settings=settings)
    st.caption(f"Backend: `{result.get('backend')}`")

    rows = result.get("results", [])
    if not rows:
        empty_state(result.get("reason", "No vendors connect multiple high-risk accounts."))
        return

    frame = pd.DataFrame(rows)
    st.markdown(
        "**The headline fraud-network question**: vendors receiving payments from several "
        "accounts that were *independently* scored as high risk."
    )
    if "total_usd" in frame.columns:
        st.metric("Value flowing through these vendors", money(float(frame["total_usd"].sum())))
    st.dataframe(frame, width="stretch", hide_index=True)


def _award_loops(settings) -> None:
    from src.graph.graph_queries import circular_payment_paths

    limit = st.slider("Results", 5, 100, 25, key="loop_limit")
    result = circular_payment_paths(limit=limit, settings=settings)
    st.caption(f"Backend: `{result.get('backend')}`")

    rows = result.get("results", [])
    if not rows:
        empty_state(result.get("reason", "No award loops with flagged payments found."))
        return

    st.markdown(
        "An agency awards a contract to a vendor, and one of that agency's own accounts "
        "makes flagged payments to the same vendor. Proof of nothing on its own - "
        "it is the shape that warrants a look, especially where competition was limited."
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _duplicates(settings) -> None:
    from src.graph.graph_queries import shared_tax_id_vendors

    limit = st.slider("Results", 5, 100, 25, key="dup_limit")
    result = shared_tax_id_vendors(limit=limit, settings=settings)
    st.caption(f"Backend: `{result.get('backend')}`")

    rows = result.get("results", [])
    if not rows:
        st.success("No vendors share a tax identifier.")
        return

    frame = pd.DataFrame(rows)
    st.markdown(
        "The same legal entity registered under multiple vendor ids. The platform "
        "**reports** these rather than merging them: automatically collapsing master "
        "data on a fuzzy signal destroys the audit trail, so adjudication stays with a "
        "data steward."
    )
    st.metric("Duplicate registrations found", number(len(frame)))
    st.dataframe(frame, width="stretch", hide_index=True)


main()
