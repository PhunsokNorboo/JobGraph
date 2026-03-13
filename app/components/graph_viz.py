"""Interactive knowledge graph visualization for search results."""
import streamlit as st
import plotly.graph_objects as go
from extraction.schema import SearchResult


def render_graph_viz(results: list[SearchResult], query_skills: list[str]):
    """Render an interactive subgraph showing query skills -> jobs -> required skills."""
    if not results:
        return

    # Build nodes and edges
    nodes = {}  # id -> {label, type, color, size}
    edges = []  # (src_id, dst_id)

    # Add a center "You" node
    nodes["you"] = {"label": "You", "type": "query", "color": "#E91E63", "size": 30}

    # Add query skill nodes
    for skill in query_skills[:10]:  # Limit to top 10
        sid = f"qs_{skill}"
        nodes[sid] = {"label": skill, "type": "your_skill", "color": "#4CAF50", "size": 15}
        edges.append(("you", sid))

    # Add job nodes and their skill connections
    for i, r in enumerate(results[:8]):  # Limit to top 8 jobs
        jid = f"job_{i}"
        title = r.job.title[:30] + "..." if len(r.job.title) > 30 else r.job.title
        nodes[jid] = {"label": title, "type": "job", "color": "#2196F3", "size": 20}

        # Connect matched skills to this job
        for skill in r.matched_skills:
            sid = f"qs_{skill}"
            if sid in nodes:
                edges.append((sid, jid))

        # Add top missing skills for this job
        for skill in r.missing_skills[:3]:
            sid = f"ms_{skill}"
            if sid not in nodes:
                nodes[sid] = {"label": skill, "type": "gap_skill", "color": "#FF9800", "size": 12}
            edges.append((jid, sid))

    # Layout: force-directed using simple positioning
    import math
    node_ids = list(nodes.keys())
    n = len(node_ids)
    positions = {}

    # Place "you" at center
    positions["you"] = (0, 0)

    # Place query skills in inner ring
    qs_nodes = [nid for nid in node_ids if nodes[nid]["type"] == "your_skill"]
    for i, nid in enumerate(qs_nodes):
        angle = 2 * math.pi * i / max(len(qs_nodes), 1)
        positions[nid] = (1.5 * math.cos(angle), 1.5 * math.sin(angle))

    # Place jobs in middle ring
    job_nodes = [nid for nid in node_ids if nodes[nid]["type"] == "job"]
    for i, nid in enumerate(job_nodes):
        angle = 2 * math.pi * i / max(len(job_nodes), 1) + 0.3
        positions[nid] = (3.0 * math.cos(angle), 3.0 * math.sin(angle))

    # Place gap skills in outer ring
    ms_nodes = [nid for nid in node_ids if nodes[nid]["type"] == "gap_skill"]
    for i, nid in enumerate(ms_nodes):
        angle = 2 * math.pi * i / max(len(ms_nodes), 1) + 0.15
        positions[nid] = (4.5 * math.cos(angle), 4.5 * math.sin(angle))

    # Build edge traces
    edge_x, edge_y = [], []
    for src, dst in edges:
        if src in positions and dst in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(150,150,150,0.4)"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Group nodes by type for legend
    type_config = {
        "query": ("You", "#E91E63"),
        "your_skill": ("Your Skills", "#4CAF50"),
        "job": ("Matched Jobs", "#2196F3"),
        "gap_skill": ("Skills to Learn", "#FF9800"),
    }

    for ntype, (legend_name, color) in type_config.items():
        typed_nodes = [nid for nid in node_ids if nodes[nid]["type"] == ntype]
        if not typed_nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[positions[nid][0] for nid in typed_nodes],
            y=[positions[nid][1] for nid in typed_nodes],
            mode="markers+text",
            marker=dict(
                size=[nodes[nid]["size"] for nid in typed_nodes],
                color=color,
                line=dict(width=1, color="white"),
            ),
            text=[nodes[nid]["label"] for nid in typed_nodes],
            textposition="top center",
            textfont=dict(size=10),
            name=legend_name,
            hovertext=[nodes[nid]["label"] for nid in typed_nodes],
            hoverinfo="text",
        ))

    fig.update_layout(
        title="Knowledge Graph — Your Skills to Matching Jobs",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(t=60, b=20, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)
