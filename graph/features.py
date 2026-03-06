"""Node feature engineering for the JobGraph."""

import logging
from pathlib import Path

import numpy as np
import torch

from extraction.schema import Seniority, RoleFamily, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Seniority ordinal encoding (entry=0, mid=1, ..., manager=5)
SENIORITY_ORD: dict[str, int] = {s.value: i for i, s in enumerate(Seniority)}

# Role family one-hot dimension
ROLE_FAMILY_DIM: int = len(RoleFamily)

# Lazy-loaded encoder singleton (avoid module-level init)
_encoder = None


def get_text_encoder():
    """Lazy-load sentence-transformers model.

    Returns the same instance on subsequent calls to avoid redundant loads.
    Model: all-MiniLM-L6-v2 (384-dim, fast inference).
    """
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def encode_texts(texts: list[str], encoder=None) -> np.ndarray:
    """Encode list of strings to dense embeddings.

    Args:
        texts: list of strings to encode.
        encoder: optional pre-loaded SentenceTransformer. Falls back to
                 lazy-loaded singleton.

    Returns:
        np.ndarray of shape (len(texts), 384).
    """
    if encoder is None:
        encoder = get_text_encoder()
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    return encoder.encode(texts, show_progress_bar=True, batch_size=64)


def build_job_features(jobs_df, encoder=None) -> torch.Tensor:
    """Build feature matrix for job nodes.

    Features concatenated per row:
        title_embedding  (384)
        seniority_norm   (1)   -- ordinal, normalised to [0, 1]
        salary_mid_norm  (1)   -- midpoint of salary range, normalised
        role_family_ohe  (8)   -- one-hot encoded

    Total: 394 dims.

    Args:
        jobs_df: DataFrame with columns: title, seniority, salary_min,
                 salary_max, role_family.
        encoder: optional SentenceTransformer instance.

    Returns:
        Tensor of shape (n_jobs, 394).
    """
    n = len(jobs_df)
    if n == 0:
        return torch.empty((0, 394), dtype=torch.float32)

    # Title embeddings
    title_embs = encode_texts(jobs_df["title"].tolist(), encoder)

    # Seniority ordinal (normalised)
    max_sen = max(SENIORITY_ORD.values()) if SENIORITY_ORD else 1
    seniority = (
        jobs_df["seniority"]
        .map(SENIORITY_ORD)
        .fillna(0)
        .values.astype(float)
        .reshape(-1, 1)
    )
    seniority = seniority / max(max_sen, 1)

    # Salary midpoint (normalised)
    salary_min = jobs_df["salary_min"].fillna(0).astype(float)
    salary_max = jobs_df["salary_max"].fillna(0).astype(float)
    salary_mid = ((salary_min + salary_max) / 2).values.reshape(-1, 1)
    sal_max_val = salary_mid.max()
    salary_mid = salary_mid / max(sal_max_val, 1)

    # Role family one-hot
    role_map: dict[str, int] = {r.value: i for i, r in enumerate(RoleFamily)}
    role_indices = (
        jobs_df["role_family"]
        .map(role_map)
        .fillna(len(RoleFamily) - 1)
        .astype(int)
        .values
    )
    role_onehot = np.zeros((n, ROLE_FAMILY_DIM), dtype=np.float32)
    role_onehot[np.arange(n), role_indices] = 1.0

    features = np.hstack([title_embs, seniority, salary_mid, role_onehot])
    return torch.tensor(features, dtype=torch.float32)


def build_skill_features(
    skill_names: list[str], skill_freqs: list[int], encoder=None
) -> torch.Tensor:
    """Build feature matrix for skill nodes.

    Features concatenated per row:
        name_embedding    (384)
        frequency_norm    (1)   -- normalised to [0, 1]

    Total: 385 dims.

    Args:
        skill_names: canonical skill names.
        skill_freqs: how often each skill appears across all jobs.
        encoder: optional SentenceTransformer instance.

    Returns:
        Tensor of shape (n_skills, 385).
    """
    if not skill_names:
        return torch.empty((0, 385), dtype=torch.float32)

    name_embs = encode_texts(skill_names, encoder)
    freq_arr = np.array(skill_freqs, dtype=np.float32).reshape(-1, 1)
    freq_arr = freq_arr / max(freq_arr.max(), 1)

    features = np.hstack([name_embs, freq_arr])
    return torch.tensor(features, dtype=torch.float32)


def build_company_features(
    company_names: list[str], encoder=None
) -> torch.Tensor:
    """Build feature matrix for company nodes.

    Features: name_embedding (384 dims).

    Args:
        company_names: list of company name strings.
        encoder: optional SentenceTransformer instance.

    Returns:
        Tensor of shape (n_companies, 384).
    """
    if not company_names:
        return torch.empty((0, 384), dtype=torch.float32)

    name_embs = encode_texts(company_names, encoder)
    return torch.tensor(name_embs, dtype=torch.float32)
