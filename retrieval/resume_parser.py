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


_SENIORITY_RANK = {
    Seniority.ENTRY: 0,
    Seniority.MID: 1,
    Seniority.SENIOR: 2,
    Seniority.STAFF: 3,
    Seniority.PRINCIPAL: 4,
    Seniority.MANAGER: 4,
}

_SENIORITY_KEYWORDS: list[tuple[str, Seniority]] = [
    # Check multi-word phrases first (most specific)
    ("entry level", Seniority.ENTRY),
    ("entry-level", Seniority.ENTRY),
    ("just graduated", Seniority.ENTRY),
    ("new grad", Seniority.ENTRY),
    ("recent graduate", Seniority.ENTRY),
    ("recently graduated", Seniority.ENTRY),
    ("first job", Seniority.ENTRY),
    ("fresher", Seniority.ENTRY),
    ("junior", Seniority.ENTRY),
    ("intern ", Seniority.ENTRY),
    ("internship", Seniority.ENTRY),
    ("mid level", Seniority.MID),
    ("mid-level", Seniority.MID),
    ("some experience", Seniority.MID),
    ("lead developer", Seniority.SENIOR),
    ("senior", Seniority.SENIOR),
    ("experienced", Seniority.SENIOR),
    ("staff", Seniority.STAFF),
    ("principal", Seniority.STAFF),
    ("architect", Seniority.STAFF),
    ("distinguished", Seniority.STAFF),
    ("engineering manager", Seniority.MANAGER),
    ("head of", Seniority.MANAGER),
    ("director", Seniority.MANAGER),
    ("vp of", Seniority.MANAGER),
    ("manager", Seniority.MANAGER),
]


def detect_seniority(text: str) -> Seniority | None:
    """Detect preferred seniority level from query text.

    Returns the first matching seniority, or None if no signal found.
    Keywords are checked in priority order (most specific first).
    """
    text_lower = text.lower()
    for keyword, seniority in _SENIORITY_KEYWORDS:
        if keyword in text_lower:
            return seniority
    return None


def get_allowed_seniorities(preferred: Seniority) -> set[str]:
    """Return seniority values within ±1 level of the preferred level."""
    rank = _SENIORITY_RANK.get(preferred, 1)
    allowed = set()
    for sen, r in _SENIORITY_RANK.items():
        if abs(r - rank) <= 1:
            allowed.add(sen.value)
    return allowed


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
