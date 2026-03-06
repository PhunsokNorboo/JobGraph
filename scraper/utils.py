"""Scraper utility functions.

Helpers for file I/O, robots.txt checking, and company CSV loading.
All path resolution uses Path(__file__) — no hardcoded absolute paths.
"""

import csv
import json
import re
import logging
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from extraction.schema import COMPANIES_CSV_COLUMNS

logger = logging.getLogger(__name__)

USER_AGENT = "JobGraph/1.0 (research project)"

# Cache parsed robots.txt per domain to avoid re-fetching
_robots_cache: dict[str, RobotFileParser | None] = {}


def get_base_dir() -> Path:
    """Return the project root directory (2 levels up from this file).

    scraper/utils.py -> scraper/ -> project root
    """
    return Path(__file__).resolve().parent.parent


def slugify(name: str) -> str:
    """Convert a company name to a filesystem-safe slug.

    Examples:
        slugify("Weights & Biases") -> "weights-biases"
        slugify("Palo Alto Networks") -> "palo-alto-networks"
        slugify("1Password") -> "1password"
    """
    slug = name.lower().strip()
    # Replace non-alphanumeric characters (except hyphens) with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    return slug


def load_companies(csv_path: Path) -> list[dict]:
    """Load companies.csv into a list of dicts.

    Each dict has keys matching COMPANIES_CSV_COLUMNS:
    name, domain, careers_url, industry, stage, size.
    """
    companies = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only keep the columns we care about, strip whitespace
            company = {col: row.get(col, "").strip() for col in COMPANIES_CSV_COLUMNS}
            companies.append(company)
    logger.info("Loaded %d companies from %s", len(companies), csv_path)
    return companies


def respect_robots(url: str) -> bool:
    """Check whether the given URL is allowed by the domain's robots.txt.

    Returns True if the URL is allowed (or if robots.txt cannot be fetched).
    Returns False only if robots.txt explicitly disallows the path.
    """
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    if domain not in _robots_cache:
        robots_url = f"{domain}/robots.txt"
        rp = RobotFileParser()
        try:
            # Fetch robots.txt with a short timeout
            resp = httpx.get(robots_url, timeout=5.0, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                _robots_cache[domain] = rp
            else:
                # No robots.txt or error -> allow everything
                _robots_cache[domain] = None
        except (httpx.HTTPError, Exception) as exc:
            logger.debug("Could not fetch robots.txt for %s: %s", domain, exc)
            _robots_cache[domain] = None

    rp = _robots_cache[domain]
    if rp is None:
        return True
    return rp.can_fetch(USER_AGENT, url)


def save_raw_job(company_slug: str, job_id: str, data: dict) -> Path:
    """Save a raw job dict as JSON to data/raw/{company_slug}/{job_id}.json.

    Creates parent directories if they don't exist. Returns the path
    to the saved file.
    """
    base = get_base_dir() / "data" / "raw" / company_slug
    base.mkdir(parents=True, exist_ok=True)
    filepath = base / f"{job_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug("Saved raw job: %s", filepath)
    return filepath
