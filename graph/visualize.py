"""Graph visualization utilities for the JobGraph."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- safe for servers
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import torch

logger = logging.getLogger(__name__)

# Colour palette per node type
_NODE_COLOURS = {
    "company": "#FF6B6B",
    "job": "#4ECDC4",
    "skill": "#45B7D1",
}
_NODE_SIZES = {
    "company": 800,
    "job": 400,
    "skill": 200,
}


def visualize_company_subgraph(
    hetero_data,
    mappings: dict,
    company_name: str,
    max_jobs: int = 5,
    output_path: Optional[Path] = None,
) -> None:
    """Render a company's jobs and required skills as a network diagram.

    Extracts the local neighbourhood of a given company from the full
    HeteroData graph and draws it with networkx + matplotlib.

    Args:
        hetero_data: PyG HeteroData object (as returned by builder.build_graph).
        mappings:    dict with keys ``companies``, ``skills``, ``job_ids``.
        company_name: exact company name to visualise.
        max_jobs:    maximum number of job nodes to include (keeps the
                     diagram readable).
        output_path: if provided, save as PNG; otherwise call plt.show().
    """
    companies: list[str] = mappings["companies"]
    skills: list[str] = mappings["skills"]
    job_ids: list[str] = mappings.get("job_ids", [])

    if company_name not in companies:
        logger.error("Company '%s' not found in graph", company_name)
        return

    company_idx = companies.index(company_name)

    # ── Find jobs at this company ───────────────────────────
    job_at_company = hetero_data["job", "at", "company"].edge_index
    job_mask = job_at_company[1] == company_idx
    job_indices: list[int] = job_at_company[0][job_mask].tolist()[:max_jobs]

    if not job_indices:
        logger.warning("No jobs found for company '%s'", company_name)
        return

    # ── Find skills required by those jobs ──────────────────
    job_requires_skill = hetero_data["job", "requires", "skill"].edge_index
    skill_indices: set[int] = set()
    job_skill_edges: list[tuple[str, str]] = []

    for job_idx in job_indices:
        mask = job_requires_skill[0] == job_idx
        s_indices = job_requires_skill[1][mask].tolist()
        for s_idx in s_indices:
            skill_indices.add(s_idx)
            job_skill_edges.append((f"job_{job_idx}", f"skill_{s_idx}"))

    # ── Find skill-skill co-occurrence edges in this subgraph
    skill_cooccurs = hetero_data["skill", "cooccurs", "skill"].edge_index
    skill_skill_edges: list[tuple[str, str]] = []
    for i in range(skill_cooccurs.shape[1]):
        s1 = skill_cooccurs[0][i].item()
        s2 = skill_cooccurs[1][i].item()
        if s1 in skill_indices and s2 in skill_indices and s1 < s2:
            skill_skill_edges.append((f"skill_{s1}", f"skill_{s2}"))

    # ── Build networkx graph ────────────────────────────────
    G = nx.Graph()

    # Company hub node
    G.add_node(company_name, node_type="company", label=company_name)

    # Job nodes + edges to company
    for j_idx in job_indices:
        label = (
            job_ids[j_idx][:20] if j_idx < len(job_ids) else f"Job {j_idx}"
        )
        G.add_node(f"job_{j_idx}", node_type="job", label=label)
        G.add_edge(company_name, f"job_{j_idx}")

    # Skill nodes
    for s_idx in skill_indices:
        label = skills[s_idx] if s_idx < len(skills) else f"Skill {s_idx}"
        G.add_node(f"skill_{s_idx}", node_type="skill", label=label)

    # Job-skill and skill-skill edges
    G.add_edges_from(job_skill_edges)
    G.add_edges_from(skill_skill_edges)

    # ── Draw ────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    for node_type, colour in _NODE_COLOURS.items():
        nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("node_type") == node_type
        ]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=colour,
            node_size=_NODE_SIZES[node_type],
            alpha=0.9,
            ax=ax,
        )

    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title(
        f"JobGraph Subgraph: {company_name}",
        fontsize=14,
        fontweight="bold",
    )
    legend_handles = [
        Patch(color=c, label=t.title()) for t, c in _NODE_COLOURS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper left")
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved visualization to %s", output_path)
    else:
        plt.show()
    plt.close(fig)
