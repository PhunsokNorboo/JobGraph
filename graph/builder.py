"""Constructs PyG HeteroData graph from structured job data."""

import ast
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from extraction.schema import NODE_TYPES, EDGE_TYPES, EMBEDDING_DIM
from graph.features import (
    build_job_features,
    build_skill_features,
    build_company_features,
    get_text_encoder,
)

logger = logging.getLogger(__name__)


def _parse_skill_list(val):
    """Safely convert a stored skill list back to a Python list.

    Parquet may store the column as native lists or as string
    representations -- handle both.
    """
    if isinstance(val, list):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def build_graph(
    jobs_parquet: Path,
    output_dir: Path,
    cooccurrence_threshold: int = 5,
) -> HeteroData:
    """Build a heterogeneous graph from jobs.parquet.

    Node types: job, skill, company
    Edge types: (job, requires, skill), (job, at, company),
                (skill, cooccurs, skill)

    Args:
        jobs_parquet: path to the processed jobs.parquet file.
        output_dir:   directory for saved artefacts (hetero_data.pt,
                      mappings.json).
        cooccurrence_threshold: minimum co-occurrence count to create a
                                skill-skill edge.

    Returns:
        Fully populated PyG HeteroData object.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare DataFrame ────────────────────────────
    df = pd.read_parquet(jobs_parquet)
    # Reset index so integer row position == node id
    df = df.reset_index(drop=True)
    logger.info("Loaded %d jobs from %s", len(df), jobs_parquet)

    df["required_skills"] = df["required_skills"].apply(_parse_skill_list)

    # ── Node index maps ─────────────────────────────────────
    companies = sorted(df["company"].dropna().unique().tolist())
    company_to_idx: dict[str, int] = {c: i for i, c in enumerate(companies)}

    all_skills_flat: list[str] = []
    for skills in df["required_skills"]:
        all_skills_flat.extend(skills)
    skill_counts = Counter(all_skills_flat)
    skills = sorted(skill_counts.keys())
    skill_to_idx: dict[str, int] = {s: i for i, s in enumerate(skills)}
    skill_freqs: list[int] = [skill_counts[s] for s in skills]

    logger.info(
        "Nodes: %d jobs, %d skills, %d companies",
        len(df),
        len(skills),
        len(companies),
    )

    # ── Node features ───────────────────────────────────────
    encoder = get_text_encoder()
    x_job = build_job_features(df, encoder)
    x_skill = build_skill_features(skills, skill_freqs, encoder)
    x_company = build_company_features(companies, encoder)

    # ── Edges: job  ── requires ──▸ skill ───────────────────
    job_skill_src: list[int] = []
    job_skill_dst: list[int] = []
    for row_pos, row in df.iterrows():
        for skill in row["required_skills"]:
            if skill in skill_to_idx:
                job_skill_src.append(int(row_pos))
                job_skill_dst.append(skill_to_idx[skill])

    # ── Edges: job  ── at ──▸ company ───────────────────────
    job_company_src: list[int] = []
    job_company_dst: list[int] = []
    for row_pos, row in df.iterrows():
        company = row.get("company")
        if company and company in company_to_idx:
            job_company_src.append(int(row_pos))
            job_company_dst.append(company_to_idx[company])

    # ── Edges: skill ── cooccurs ── skill (undirected) ──────
    skill_cooccur: Counter = Counter()
    for skills_list in df["required_skills"]:
        idxs = sorted(
            {skill_to_idx[s] for s in skills_list if s in skill_to_idx}
        )
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                skill_cooccur[(idxs[i], idxs[j])] += 1

    cooccur_src: list[int] = []
    cooccur_dst: list[int] = []
    for (s1, s2), count in skill_cooccur.items():
        if count >= cooccurrence_threshold:
            # Store both directions for undirected edge
            cooccur_src.extend([s1, s2])
            cooccur_dst.extend([s2, s1])

    # ── Assemble HeteroData ─────────────────────────────────
    data = HeteroData()

    data["job"].x = x_job
    data["skill"].x = x_skill
    data["company"].x = x_company

    data["job", "requires", "skill"].edge_index = torch.tensor(
        [job_skill_src, job_skill_dst], dtype=torch.long
    )
    data["job", "at", "company"].edge_index = torch.tensor(
        [job_company_src, job_company_dst], dtype=torch.long
    )
    data["skill", "cooccurs", "skill"].edge_index = torch.tensor(
        [cooccur_src, cooccur_dst], dtype=torch.long
    )

    # ── Persist artefacts ───────────────────────────────────
    graph_path = output_dir / "hetero_data.pt"
    torch.save(data, graph_path)
    logger.info("Saved graph to %s", graph_path)

    mappings = {
        "companies": companies,
        "skills": skills,
        "skill_freqs": skill_freqs,
        "job_ids": df["job_id"].tolist(),
    }
    mappings_path = output_dir / "mappings.json"
    with open(mappings_path, "w") as f:
        json.dump(mappings, f, indent=2)
    logger.info("Saved mappings to %s", mappings_path)

    _print_graph_stats(data)

    return data


def _print_graph_stats(data: HeteroData) -> None:
    """Log node/edge counts and feature dimensions."""
    logger.info("=== Graph Statistics ===")
    for node_type in data.node_types:
        shape = data[node_type].x.shape
        logger.info("  %s: %d nodes, %d features", node_type, shape[0], shape[1])
    for edge_type in data.edge_types:
        n_edges = data[edge_type].edge_index.shape[1]
        logger.info("  %s: %d edges", edge_type, n_edges)


# ── CLI entry point ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parent.parent
    parquet_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else base / "data" / "processed" / "jobs.parquet"
    )
    build_graph(
        jobs_parquet=parquet_path,
        output_dir=base / "data" / "graph",
    )
