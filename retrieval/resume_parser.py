"""Parse resume PDFs and free-text queries into search vectors."""
import logging
from pathlib import Path

import numpy as np

from extraction.schema import SearchQuery, RoleFamily, Seniority

logger = logging.getLogger(__name__)


def parse_pdf(file_path_or_bytes) -> str:
    """Extract text from a PDF file or bytes.

    Args:
        file_path_or_bytes: Path to PDF or bytes-like object (from Streamlit upload)

    Returns:
        Extracted text string
    """
    from pdfminer.high_level import extract_text
    from io import BytesIO

    if isinstance(file_path_or_bytes, (str, Path)):
        return extract_text(str(file_path_or_bytes))
    else:
        return extract_text(BytesIO(file_path_or_bytes.read()))


def extract_skills_from_text(text: str) -> list[str]:
    """Extract technical skills from free text using keyword matching.

    This is a fast heuristic -- no LLM needed. For higher quality,
    use the LLM extraction path.
    """
    # Common technical skills to detect
    KNOWN_SKILLS = {
        "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
        "ruby", "php", "scala", "kotlin", "swift", "r",
        "react", "angular", "vue", "next.js", "node.js", "django", "flask", "fastapi",
        "spring", "express", "rails",
        "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "scipy",
        "keras", "hugging face", "transformers", "langchain",
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "kafka", "rabbitmq", "spark", "hadoop", "airflow",
        "git", "linux", "ci/cd", "jenkins", "github actions",
        "graphql", "rest", "grpc", "sql", "nosql",
        "machine learning", "deep learning", "nlp", "computer vision", "llm",
        "streamlit", "tableau", "power bi",
        "figma", "sketch",
    }

    text_lower = text.lower()
    found = []
    for skill in sorted(KNOWN_SKILLS):
        if skill in text_lower:
            found.append(skill)
    return found


def encode_query(
    query_text: str,
    encoder=None,
) -> np.ndarray:
    """Encode query text into embedding vector for FAISS search.

    Uses sentence-transformers (all-MiniLM-L6-v2) directly — same space
    as the FAISS index.

    Args:
        query_text: Resume text or free-text query
        encoder: Optional pre-loaded SentenceTransformer

    Returns:
        Normalized embedding vector (1, 384)
    """
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

    embedding = encoder.encode([query_text]).astype("float32")  # (1, 384)

    # Normalize for cosine similarity
    import faiss
    faiss.normalize_L2(embedding)
    return embedding


def build_search_query(text: str) -> SearchQuery:
    """Build a SearchQuery from raw text."""
    skills = extract_skills_from_text(text)
    return SearchQuery(
        raw_text=text,
        extracted_skills=skills,
    )
