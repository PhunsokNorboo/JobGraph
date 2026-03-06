#!/usr/bin/env bash
set -euo pipefail

# Train the HGT model and generate embeddings
# Usage: bash scripts/train_model.sh [epochs]
# Example: bash scripts/train_model.sh 100

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT"

EPOCHS="${1:-200}"

echo "Training HGT model for $EPOCHS epochs..."
python -m model.train

echo "Generating embeddings..."
python -m model.embed

echo "Building FAISS index..."
python -m retrieval.index

echo "Training pipeline complete."
echo "  Model: data/graph/best_model.pt"
echo "  Embeddings: data/graph/job_embeddings.npy"
echo "  FAISS index: data/graph/faiss.index"
