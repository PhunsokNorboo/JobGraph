"""Generate and save node embeddings from trained model."""
import json
import logging
from pathlib import Path

import numpy as np
import torch

from model.hgt import JobGraphHGT

logger = logging.getLogger(__name__)


def generate_embeddings(
    graph_path: Path,
    model_path: Path,
    output_dir: Path,
    device: str = "auto",
):
    """Generate embeddings for all nodes using trained model.

    Saves:
        - job_embeddings.npy (N_jobs x 256)
        - skill_embeddings.npy (N_skills x 256)
        - company_embeddings.npy (N_companies x 256)
        - job_index.json (mapping job_idx -> job_id)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graph and model
    data = torch.load(graph_path, weights_only=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    model = JobGraphHGT(
        metadata=checkpoint["metadata"],
        hidden_channels=checkpoint["hidden_channels"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Forward pass
    with torch.no_grad():
        d = data.to(device)
        z_dict = model(d.x_dict, d.edge_index_dict)

    # Save embeddings
    for node_type in z_dict:
        emb = z_dict[node_type].cpu().numpy()
        np.save(output_dir / f"{node_type}_embeddings.npy", emb)
        logger.info(f"Saved {node_type} embeddings: {emb.shape}")

    # Save job index mapping
    mappings_path = output_dir / "mappings.json"
    if mappings_path.exists():
        with open(mappings_path) as f:
            mappings = json.load(f)
        job_index = {str(i): jid for i, jid in enumerate(mappings.get("job_ids", []))}
        with open(output_dir / "job_index.json", "w") as f:
            json.dump(job_index, f, indent=2)
        logger.info(f"Saved job index with {len(job_index)} entries")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parent.parent
    generate_embeddings(
        graph_path=base / "data" / "graph" / "hetero_data.pt",
        model_path=base / "data" / "graph" / "best_model.pt",
        output_dir=base / "data" / "graph",
    )
