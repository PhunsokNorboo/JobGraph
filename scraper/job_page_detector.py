"""Detects and classifies job listing pages from company domains.

Given a company domain, probes common career page URL patterns and
known ATS (Applicant Tracking System) providers to find the active
careers page. Uses HEAD requests with httpx to minimize bandwidth.
"""

import asyncio
import logging
from urllib.parse import urlparse

import httpx

from scraper.utils import USER_AGENT

logger = logging.getLogger(__name__)

# Common paths appended to a company's root domain
COMMON_PATHS = [
    "/careers",
    "/jobs",
    "/join",
    "/work-with-us",
    "/open-roles",
    "/join-us",
    "/careers/",
    "/jobs/",
]

# Known ATS board URL templates. {company_slug} is replaced at runtime.
ATS_PATTERNS: dict[str, str] = {
    "greenhouse": "https://boards.greenhouse.io/{company_slug}",
    "lever": "https://jobs.lever.co/{company_slug}",
    "ashby": "https://jobs.ashbyhq.com/{company_slug}",
    "workable": "https://apply.workable.com/{company_slug}",
}


def _slugify_for_ats(name: str) -> str:
    """Derive a plausible ATS slug from a company name.

    Lowercase, strip non-alphanumeric, collapse whitespace to nothing.
    Examples: "Weights & Biases" -> "weightsbiases",
              "Palo Alto Networks" -> "paloaltonetworks"
    """
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())


async def _check_url(client: httpx.AsyncClient, url: str) -> bool:
    """Return True if the URL responds with HTTP 200 (via HEAD, then GET)."""
    try:
        resp = await client.head(url, follow_redirects=True, timeout=10.0)
        if resp.status_code == 200:
            return True
        # Some servers reject HEAD; fall back to GET
        if resp.status_code in (405, 403):
            resp = await client.get(url, follow_redirects=True, timeout=10.0)
            return resp.status_code == 200
        return False
    except (httpx.HTTPError, Exception) as exc:
        logger.debug("URL check failed for %s: %s", url, exc)
        return False


def detect_ats_type(url: str) -> str | None:
    """Identify which ATS a careers URL belongs to, if any.

    Returns one of 'greenhouse', 'lever', 'ashby', 'workable', or None.
    """
    url_lower = url.lower()
    if "greenhouse.io" in url_lower:
        return "greenhouse"
    if "lever.co" in url_lower:
        return "lever"
    if "ashbyhq.com" in url_lower:
        return "ashby"
    if "workable.com" in url_lower:
        return "workable"
    return None


async def detect_careers_url(
    domain: str,
    known_url: str | None = None,
    company_name: str | None = None,
) -> str | None:
    """Detect the careers page URL for a company.

    Strategy:
    1. If known_url is provided and responds 200, use it.
    2. Try common path suffixes on the domain (e.g. domain.com/careers).
    3. Try known ATS board patterns using a slug derived from company_name or domain.

    Args:
        domain: Company domain without protocol, e.g. "anthropic.com".
        known_url: A previously known careers URL to verify first.
        company_name: Company display name, used to derive ATS slugs.

    Returns:
        The first valid careers URL found, or None.
    """
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(headers=headers) as client:
        # 1. Check the known URL first
        if known_url:
            if await _check_url(client, known_url):
                logger.info("Known careers URL valid: %s", known_url)
                return known_url
            logger.info("Known careers URL invalid (%s), probing alternatives", known_url)

        # Normalise domain
        if not domain.startswith("http"):
            base_url = f"https://{domain}"
        else:
            base_url = domain
        base_url = base_url.rstrip("/")

        # 2. Try common paths on the domain
        for path in COMMON_PATHS:
            candidate = f"{base_url}{path}"
            if await _check_url(client, candidate):
                logger.info("Found careers page at: %s", candidate)
                return candidate

        # 3. Try ATS board patterns
        slug = _slugify_for_ats(company_name) if company_name else domain.split(".")[0]
        for ats_name, pattern in ATS_PATTERNS.items():
            candidate = pattern.format(company_slug=slug)
            if await _check_url(client, candidate):
                logger.info("Found %s board at: %s", ats_name, candidate)
                return candidate

    logger.warning("No careers page found for domain: %s", domain)
    return None
