"""Scraper package — crawls company career pages for job postings.

Public API:
    run_scraper(companies_csv, output_dir, max_companies)
    scrape_company(company, output_dir)
    detect_careers_url(domain, known_url, company_name)
"""

from scraper.crawler import run_scraper, scrape_company
from scraper.job_page_detector import detect_careers_url
from scraper.utils import load_companies, slugify, respect_robots, save_raw_job

__all__ = [
    "run_scraper",
    "scrape_company",
    "detect_careers_url",
    "load_companies",
    "slugify",
    "respect_robots",
    "save_raw_job",
]
