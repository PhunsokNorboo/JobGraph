"""Search pipeline: encode query -> ANN search -> BM25 re-rank -> enrich results."""
import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from extraction.schema import JobRecord, SearchResult, SearchQuery, JOBS_PARQUET_COLUMNS
from retrieval.index import load_index
from retrieval.resume_parser import (
    encode_query, build_search_query, extract_skills_from_text,
    detect_seniority, get_allowed_seniorities,
)

logger = logging.getLogger(__name__)


def _parse_skill_list(val) -> list[str]:
    """Safely parse a skill list from parquet (may be list, ndarray, or string)."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return []
    return []


class JobSearchEngine:
    """Hybrid search engine: GNN embeddings + BM25 re-ranking."""

    def __init__(self, data_dir: Path | None = None):
        """Lazy initialization -- nothing loaded until first search."""
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = data_dir
        self._index = None
        self._jobs_df = None
        self._encoder = None
        self._bm25 = None
        self._job_index = None

    @property
    def index(self):
        if self._index is None:
            self._index = load_index(self.data_dir / "graph" / "faiss.index")
            logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
        return self._index

    @property
    def jobs_df(self) -> pd.DataFrame:
        if self._jobs_df is None:
            self._jobs_df = pd.read_parquet(self.data_dir / "processed" / "jobs.parquet")
            logger.info(f"Loaded {len(self._jobs_df)} jobs")
        return self._jobs_df

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            # Tokenize job titles + descriptions for BM25
            titles = self.jobs_df["title"].fillna("")
            descs = self.jobs_df["description_summary"].fillna("")
            corpus = (titles + " " + descs).tolist()
            tokenized = [doc.lower().split() for doc in corpus]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        seniority_filter: str | None = None,
        role_family_filter: str | None = None,
        location_filter: str | None = None,
    ) -> list[SearchResult]:
        """Full search pipeline.

        Args:
            query_text: Resume text or free-text job query
            top_k: Number of results to return
            seniority_filter: Explicit seniority filter (e.g. "entry", "senior").
                              If None, auto-detects from query text.
            role_family_filter: Filter by role family (e.g. "swe", "ml", "data").
            location_filter: Filter by location type (e.g. "remote", "hybrid", "onsite").

        Returns:
            List of SearchResult with job, similarity, skill overlap
        """
        # Step 1: Encode query (same 384-dim space as FAISS index)
        query_vec = encode_query(query_text, self.encoder)

        # Determine seniority filter
        allowed_seniorities = None
        if seniority_filter:
            from extraction.schema import Seniority
            if seniority_filter == "staff+":
                # Staff+ means staff, principal, manager
                allowed_seniorities = {"staff", "principal", "manager"}
            else:
                try:
                    preferred = Seniority(seniority_filter)
                    # Explicit dropdown selection = strict match (exact level only)
                    allowed_seniorities = {preferred.value}
                except ValueError:
                    pass
        else:
            detected = detect_seniority(query_text)
            if detected:
                # Auto-detect from query = allow ±1 level for flexibility
                allowed_seniorities = get_allowed_seniorities(detected)
                logger.info(f"Auto-detected seniority: {detected.value}, allowed: {allowed_seniorities}")

        # Step 2: ANN search — search all vectors to ensure filtered results aren't missed
        n_candidates = self.index.ntotal
        distances, indices = self.index.search(query_vec, n_candidates)

        # Step 3: BM25 re-ranking
        query_tokens = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Combine scores: 0.7 * cosine + 0.3 * normalized_bm25
        candidates = []
        bm25_max = max(bm25_scores.max(), 1e-10)

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.jobs_df):
                continue

            row_data = self.jobs_df.iloc[idx]

            # Apply seniority filter
            if allowed_seniorities is not None:
                if row_data.get("seniority", "") not in allowed_seniorities:
                    continue

            # Apply role family filter
            if role_family_filter:
                if row_data.get("role_family", "") != role_family_filter:
                    continue

            # Apply location filter
            if location_filter:
                if row_data.get("location_type", "") != location_filter:
                    continue

            combined = 0.7 * float(dist) + 0.3 * (bm25_scores[idx] / bm25_max)
            candidates.append((idx, combined, float(dist)))

            # Stop early once we have enough candidates
            if len(candidates) >= top_k * 3:
                break

        # Sort by combined score, take top_k
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:top_k]

        # Step 4: Enrich with skill overlap
        user_skills = set(s.lower() for s in extract_skills_from_text(query_text))

        results = []
        for idx, combined_score, cosine_sim in candidates:
            row = self.jobs_df.iloc[idx]

            # Parse skills (handles list, ndarray, or string repr)
            required = _parse_skill_list(row["required_skills"])
            nice_to_have = _parse_skill_list(row.get("nice_to_have_skills", []))

            required_lower = {s.lower(): s for s in required}

            matched = [required_lower[s] for s in user_skills if s in required_lower]
            missing = [s for s_lower, s in required_lower.items() if s_lower not in user_skills]

            job = JobRecord(
                job_id=row.get("job_id", ""),
                title=row["title"] if row["title"] else "Untitled Position",
                company=row["company"],
                seniority=row["seniority"],
                role_family=row["role_family"],
                required_skills=required,
                nice_to_have_skills=nice_to_have,
                salary_min=int(row["salary_min"]) if pd.notna(row.get("salary_min")) else None,
                salary_max=int(row["salary_max"]) if pd.notna(row.get("salary_max")) else None,
                location_type=row["location_type"],
                location_city=row.get("location_city"),
                description_summary=row.get("description_summary", ""),
                raw_url=row["raw_url"],
            )

            results.append(SearchResult(
                job=job,
                similarity_score=cosine_sim,
                matched_skills=matched,
                missing_skills=missing,
            ))

        return results


# Module-level convenience function
_engine = None


def search(
    query_text: str,
    top_k: int = 10,
    data_dir: Path | None = None,
    seniority_filter: str | None = None,
    role_family_filter: str | None = None,
    location_filter: str | None = None,
) -> list[SearchResult]:
    """Convenience function for search."""
    global _engine
    if _engine is None:
        _engine = JobSearchEngine(data_dir)
    return _engine.search(
        query_text, top_k,
        seniority_filter=seniority_filter,
        role_family_filter=role_family_filter,
        location_filter=location_filter,
    )
