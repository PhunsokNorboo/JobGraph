"""Build and manage FAISS index for job embeddings."""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_index(embeddings_path: Path, output_path: Path) -> None:
    """Build FAISS index from saved embeddings.

    Args:
        embeddings_path: Path to job_embeddings.npy
        output_path: Path to save faiss.index
    """
    import faiss

    embeddings = np.load(embeddings_path).astype("float32")
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Inner product = cosine similarity after normalization
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    logger.info(f"FAISS index saved to {output_path} ({index.ntotal} vectors)")


def load_index(index_path: Path):
    """Load FAISS index from disk."""
    import faiss

    return faiss.read_index(str(index_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parent.parent
    build_index(
        embeddings_path=base / "data" / "graph" / "job_embeddings.npy",
        output_path=base / "data" / "graph" / "faiss.index",
    )
