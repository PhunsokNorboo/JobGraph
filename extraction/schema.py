"""
JobGraph Shared Schemas — Single Source of Truth.

Every module in the project imports its data contracts from here.
Do NOT define data models anywhere else.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ─── Enums ───────────────────────────────────────────────

class Seniority(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    MANAGER = "manager"


class RoleFamily(str, Enum):
    SWE = "swe"
    ML = "ml"
    DATA = "data"
    DESIGN = "design"
    PM = "pm"
    DEVOPS = "devops"
    SALES = "sales"
    OTHER = "other"


class LocationType(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"


class CompanyStage(str, Enum):
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    PUBLIC = "public"
    UNKNOWN = "unknown"


class CompanySize(str, Enum):
    TINY = "1-50"
    SMALL = "51-200"
    MEDIUM = "201-1000"
    LARGE = "1000+"


class SkillCategory(str, Enum):
    LANGUAGE = "language"
    FRAMEWORK = "framework"
    DATABASE = "database"
    CLOUD = "cloud"
    TOOL = "tool"
    ML_FRAMEWORK = "ml_framework"
    OTHER = "other"


# ─── Pydantic Models ────────────────────────────────────

class RawJobData(BaseModel):
    """Raw scraped job data before LLM extraction."""
    company: str
    url: str
    html: str
    scraped_at: str


class JobRecord(BaseModel):
    """Structured job record after LLM extraction + validation."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    company: str
    seniority: Seniority
    role_family: RoleFamily
    required_skills: list[str]
    nice_to_have_skills: list[str] = Field(default_factory=list)
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    location_type: LocationType
    location_city: Optional[str] = None
    description_summary: str
    raw_url: str
    scraped_at: Optional[str] = None


class CompanyRecord(BaseModel):
    """Company metadata from seed CSV."""
    name: str
    domain: str
    careers_url: Optional[str] = None
    industry: str
    stage: CompanyStage = CompanyStage.UNKNOWN
    size: CompanySize = CompanySize.SMALL


class SkillVocab(BaseModel):
    """Canonical skill entry in the vocabulary."""
    skill_id: int
    name: str
    category: SkillCategory = SkillCategory.OTHER
    frequency: int = 0


class SearchResult(BaseModel):
    """A single search result returned to the UI."""
    job: JobRecord
    similarity_score: float
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    """User search input."""
    raw_text: str
    extracted_skills: list[str] = Field(default_factory=list)
    preferred_role: Optional[RoleFamily] = None
    preferred_seniority: Optional[Seniority] = None


# ─── Column Constants ───────────────────────────────────
# Use these when creating DataFrames to prevent typos

JOBS_PARQUET_COLUMNS = [
    "job_id", "company", "title", "seniority", "role_family",
    "required_skills", "nice_to_have_skills", "salary_min", "salary_max",
    "location_type", "location_city", "description_summary", "raw_url",
    "scraped_at",
]

COMPANIES_CSV_COLUMNS = [
    "name", "domain", "careers_url", "industry", "stage", "size",
]

SKILL_VOCAB_COLUMNS = [
    "skill_id", "name", "category", "frequency",
]

# ─── Graph Constants ────────────────────────────────────

NODE_TYPES = ["job", "skill", "company"]
EDGE_TYPES = [
    ("job", "requires", "skill"),
    ("job", "at", "company"),
    ("skill", "cooccurs", "skill"),
]

EMBEDDING_DIM = 256
