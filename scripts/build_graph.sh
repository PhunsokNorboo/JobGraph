#!/usr/bin/env bash
set -euo pipefail

# Build the PyG heterogeneous knowledge graph from extracted job data
# Usage: bash scripts/build_graph.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT"
python -m graph.builder

echo "Graph built. Output: data/graph/hetero_data.pt"
