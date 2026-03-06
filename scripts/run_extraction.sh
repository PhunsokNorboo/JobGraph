#!/usr/bin/env bash
set -euo pipefail

# Extract structured job data from raw HTML using LLM (Ollama by default)
# Usage: bash scripts/run_extraction.sh
# Set LLM_PROVIDER=openai in .env to use GPT-4o instead

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

# Load .env if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

cd "$PROJECT_ROOT"
python -m extraction.llm_agent

echo "Extraction complete. Output: data/processed/jobs.parquet"
