#!/usr/bin/env bash
set -euo pipefail

# Run the job scraper on all companies in data/companies.csv
# Usage: bash scripts/run_scraper.sh [max_companies]
# Example: bash scripts/run_scraper.sh 10  # scrape first 10 only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT"

MAX_COMPANIES="${1:-}"

if [ -n "$MAX_COMPANIES" ]; then
    python -m scraper.crawler --max-companies "$MAX_COMPANIES"
else
    python -m scraper.crawler
fi

echo "Scraping complete. Raw data saved to data/raw/"
