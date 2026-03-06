"""Playwright-based web crawler for company career pages.

Scrapes job postings from career pages — handles JS-rendered content
via Playwright and knows how to parse Greenhouse, Lever, and Ashby
board layouts. Respects robots.txt and rate-limits requests.
"""

import asyncio
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright, Page, Browser
from tqdm import tqdm

from extraction.schema import RawJobData
from scraper.utils import (
    USER_AGENT,
    get_base_dir,
    load_companies,
    respect_robots,
    save_raw_job,
    slugify,
)
from scraper.job_page_detector import detect_ats_type, detect_careers_url

logger = logging.getLogger(__name__)


def _job_hash(url: str) -> str:
    """Deterministic short hash for a job URL — used as the filename."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


async def _random_delay() -> None:
    """Sleep 1-2 seconds between requests to be polite."""
    await asyncio.sleep(random.uniform(1.0, 2.0))


# ─── ATS-specific parsers ──────────────────────────────────────


async def _scrape_greenhouse(page: Page, careers_url: str) -> list[dict]:
    """Parse a Greenhouse job board.

    Tries the JSON API first (/departments), falls back to HTML scraping.
    """
    jobs: list[dict] = []

    # Attempt 1: JSON API (most Greenhouse boards expose this)
    # The canonical API endpoint is the board URL + /departments
    # e.g. https://boards.greenhouse.io/anthropic -> https://boards-api.greenhouse.io/v1/boards/anthropic/departments
    # But there's also a simpler HTML approach: look for job links on the page.
    #
    # The public JSON endpoint pattern:
    # https://boards-api.greenhouse.io/v1/boards/{slug}/jobs
    slug = careers_url.rstrip("/").split("/")[-1]
    api_url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"

    try:
        resp = await page.context.request.get(api_url)
        if resp.status == 200:
            data = await resp.json()
            for job_entry in data.get("jobs", []):
                job_url = job_entry.get("absolute_url", "")
                title = job_entry.get("title", "")
                content = job_entry.get("content", "")
                if job_url:
                    jobs.append({
                        "url": job_url,
                        "title": title,
                        "html": content,
                    })
            if jobs:
                logger.info("Greenhouse API returned %d jobs for %s", len(jobs), slug)
                return jobs
    except Exception as exc:
        logger.debug("Greenhouse API failed for %s: %s", slug, exc)

    # Attempt 2: HTML scraping — navigate to the board page
    await page.goto(careers_url, wait_until="domcontentloaded", timeout=30000)
    await _random_delay()

    # Greenhouse uses <div class="opening"> with <a> tags
    links = await page.query_selector_all("div.opening a, a.job-post-link, a[data-job-id]")
    if not links:
        # Broader fallback: any link whose href contains /jobs/
        links = await page.query_selector_all("a[href*='/jobs/']")

    seen_urls: set[str] = set()
    for link in links:
        href = await link.get_attribute("href")
        if not href:
            continue
        # Make absolute
        if href.startswith("/"):
            href = f"https://boards.greenhouse.io{href}"
        if href in seen_urls:
            continue
        seen_urls.add(href)

        title_text = (await link.inner_text()).strip()
        jobs.append({"url": href, "title": title_text, "html": ""})

    logger.info("Greenhouse HTML scrape found %d jobs for %s", len(jobs), slug)
    return jobs


async def _scrape_lever(page: Page, careers_url: str) -> list[dict]:
    """Parse a Lever job board.

    Tries JSON mode first (?mode=json), falls back to HTML.
    """
    jobs: list[dict] = []

    # Attempt 1: JSON mode
    json_url = careers_url.rstrip("/") + "?mode=json"
    try:
        resp = await page.context.request.get(json_url)
        if resp.status == 200:
            data = await resp.json()
            for posting in data:
                job_url = posting.get("hostedUrl", "")
                title = posting.get("text", "")
                desc = posting.get("descriptionPlain", posting.get("description", ""))
                if job_url:
                    jobs.append({"url": job_url, "title": title, "html": desc})
            if jobs:
                logger.info("Lever JSON returned %d jobs", len(jobs))
                return jobs
    except Exception as exc:
        logger.debug("Lever JSON failed: %s", exc)

    # Attempt 2: HTML scraping
    await page.goto(careers_url, wait_until="domcontentloaded", timeout=30000)
    await _random_delay()

    links = await page.query_selector_all(".posting-title a, a.posting-btn-submit")
    if not links:
        links = await page.query_selector_all("a[href*='/jobs/']")

    seen_urls: set[str] = set()
    for link in links:
        href = await link.get_attribute("href")
        if not href or href in seen_urls:
            continue
        seen_urls.add(href)
        title_text = (await link.inner_text()).strip()
        jobs.append({"url": href, "title": title_text, "html": ""})

    logger.info("Lever HTML scrape found %d jobs", len(jobs))
    return jobs


async def _scrape_ashby(page: Page, careers_url: str) -> list[dict]:
    """Parse an Ashby job board.

    Ashby boards are heavily JS-rendered; navigate with Playwright
    and extract from the DOM.
    """
    jobs: list[dict] = []

    # Ashby also has an API, but it varies. Try HTML first.
    await page.goto(careers_url, wait_until="domcontentloaded", timeout=30000)
    await _random_delay()

    # Ashby renders job links inside the main content area
    links = await page.query_selector_all(
        "a[href*='/jobs/'], a[href*='/posting/'], a._company-link"
    )
    if not links:
        # Broader: any anchor that looks like a job link
        links = await page.query_selector_all("a[href]")

    seen_urls: set[str] = set()
    for link in links:
        href = await link.get_attribute("href")
        if not href:
            continue
        # Filter to only job-like paths
        if "/jobs/" not in href and "/posting/" not in href:
            continue
        # Make absolute
        if href.startswith("/"):
            href = f"https://jobs.ashbyhq.com{href}"
        if href in seen_urls:
            continue
        seen_urls.add(href)
        title_text = (await link.inner_text()).strip()
        jobs.append({"url": href, "title": title_text, "html": ""})

    logger.info("Ashby scrape found %d jobs", len(jobs))
    return jobs


async def _scrape_generic(page: Page, careers_url: str) -> list[dict]:
    """Scrape a generic careers page by extracting all job-like links."""
    jobs: list[dict] = []

    await page.goto(careers_url, wait_until="domcontentloaded", timeout=30000)
    await _random_delay()

    # Look for links that contain common job-related path segments
    all_links = await page.query_selector_all("a[href]")
    job_path_keywords = [
        "/job/", "/jobs/", "/position/", "/positions/",
        "/role/", "/roles/", "/opening/", "/openings/",
        "/career/", "/vacancy/", "/apply/",
    ]

    seen_urls: set[str] = set()
    for link in all_links:
        href = await link.get_attribute("href")
        if not href:
            continue
        # Resolve relative URLs
        if href.startswith("./") or (not href.startswith("http") and not href.startswith("//")):
            from urllib.parse import urljoin
            href = urljoin(careers_url, href)
        elif href.startswith("/"):
            from urllib.parse import urlparse
            parsed = urlparse(careers_url)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"
        href_lower = href.lower()
        # Skip non-HTTP links (mailto, javascript, pdf, etc.)
        if not href_lower.startswith("http") or href_lower.endswith(".pdf"):
            continue
        if any(kw in href_lower for kw in job_path_keywords):
            if href in seen_urls:
                continue
            seen_urls.add(href)
            title_text = (await link.inner_text()).strip()
            if title_text:  # skip empty links
                jobs.append({"url": href, "title": title_text, "html": ""})

    logger.info("Generic scrape found %d job links at %s", len(jobs), careers_url)
    return jobs


# ─── Per-job detail fetcher ────────────────────────────────────


async def _fetch_job_detail(page: Page, job: dict) -> dict | None:
    """Visit an individual job page and capture its full HTML.

    Returns an updated job dict with the 'html' field populated,
    or None if the page could not be loaded.
    """
    url = job["url"]

    # Check robots.txt before fetching
    if not respect_robots(url):
        logger.info("Blocked by robots.txt: %s", url)
        return None

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await _random_delay()
        html = await page.content()
        return {**job, "html": html}
    except Exception as exc:
        logger.warning("Failed to fetch job detail %s: %s", url, exc)
        return None


# ─── Main company scraper ──────────────────────────────────────


async def scrape_company(company: dict, output_dir: Path) -> list[dict]:
    """Scrape all job postings from a company's career page.

    Args:
        company: dict with keys from companies.csv
                 (name, domain, careers_url, industry, stage, size).
        output_dir: Path to data/raw/ directory.

    Returns:
        List of raw job dicts saved to disk.
    """
    name = company.get("name", "")
    domain = company.get("domain", "")
    known_url = company.get("careers_url", "") or None
    company_slug = slugify(name)

    logger.info("Starting scrape for %s (%s)", name, domain)

    # Detect / verify the careers URL
    careers_url = await detect_careers_url(
        domain=domain,
        known_url=known_url,
        company_name=name,
    )
    if not careers_url:
        logger.warning("No careers URL found for %s — skipping", name)
        return []

    # Check robots.txt for the careers page itself
    if not respect_robots(careers_url):
        logger.info("Careers page blocked by robots.txt for %s — skipping", name)
        return []

    # Determine ATS type
    ats_type = detect_ats_type(careers_url)
    logger.info("%s: ATS type = %s, URL = %s", name, ats_type or "generic", careers_url)

    saved_jobs: list[dict] = []

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        try:
            # Dispatch to the appropriate ATS scraper
            if ats_type == "greenhouse":
                job_listings = await _scrape_greenhouse(page, careers_url)
            elif ats_type == "lever":
                job_listings = await _scrape_lever(page, careers_url)
            elif ats_type == "ashby":
                job_listings = await _scrape_ashby(page, careers_url)
            else:
                job_listings = await _scrape_generic(page, careers_url)

            logger.info("%s: found %d job listings, fetching details...", name, len(job_listings))

            # For listings that already have HTML content (from API), save directly.
            # For those without, visit the page to get full HTML.
            for job in job_listings:
                if job.get("html"):
                    # Already have content (e.g. from Greenhouse/Lever API)
                    job_id = _job_hash(job["url"])
                    raw_data = {
                        "company": name,
                        "url": job["url"],
                        "html": job["html"],
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                    }
                    # Validate through schema
                    RawJobData(**raw_data)
                    save_raw_job(company_slug, job_id, raw_data)
                    saved_jobs.append(raw_data)
                else:
                    # Need to visit the page for full HTML
                    detail = await _fetch_job_detail(page, job)
                    if detail and detail.get("html"):
                        job_id = _job_hash(detail["url"])
                        raw_data = {
                            "company": name,
                            "url": detail["url"],
                            "html": detail["html"],
                            "scraped_at": datetime.now(timezone.utc).isoformat(),
                        }
                        RawJobData(**raw_data)
                        save_raw_job(company_slug, job_id, raw_data)
                        saved_jobs.append(raw_data)

        except Exception as exc:
            logger.error("Error scraping %s: %s", name, exc, exc_info=True)
        finally:
            await context.close()
            await browser.close()

    logger.info("%s: saved %d jobs to %s", name, len(saved_jobs), output_dir / company_slug)
    return saved_jobs


# ─── Main entry point ──────────────────────────────────────────


async def run_scraper(
    companies_csv: Path,
    output_dir: Path,
    max_companies: int | None = None,
) -> list[dict]:
    """Main entry point — scrape all companies from the CSV.

    Args:
        companies_csv: Path to companies.csv.
        output_dir: Path to data/raw/ directory.
        max_companies: If set, only scrape the first N companies.

    Returns:
        Flat list of all raw job dicts scraped.
    """
    companies = load_companies(companies_csv)
    if max_companies:
        companies = companies[:max_companies]

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for company in tqdm(companies, desc="Scraping companies"):
        try:
            jobs = await scrape_company(company, output_dir)
            results.extend(jobs)
            logging.info("%s: %d jobs found", company["name"], len(jobs))
        except Exception as exc:
            logging.error("%s: %s", company["name"], exc)

    logging.info("Total: %d jobs scraped from %d companies", len(results), len(companies))
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    csv_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else get_base_dir() / "data" / "companies.csv"
    )
    raw_dir = get_base_dir() / "data" / "raw"
    max_co = int(sys.argv[2]) if len(sys.argv) > 2 else None

    asyncio.run(run_scraper(csv_path, raw_dir, max_companies=max_co))
