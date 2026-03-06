# JobGraph -- GNN-Powered Job Discovery Engine

> Transform job postings into a knowledge graph. Use Graph Neural Networks to power semantic, personalized job discovery that goes beyond keyword matching.

[Architecture](#architecture) | [Quick Start](#quick-start) | [How It Works](#how-it-works) | [Tech Stack](#tech-stack) | [Results](#results)

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)
![PyG 2.5](https://img.shields.io/badge/PyG-2.5-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-1.8-purple.svg)
![Jobs](https://img.shields.io/badge/Jobs_Indexed-481-brightgreen.svg)
![Skills](https://img.shields.io/badge/Skills_Extracted-427-brightgreen.svg)
![Hits@10](https://img.shields.io/badge/Hits%4010-62.7%25-blue.svg)

---

## The Problem

Every job board today treats job search as a **text retrieval problem**. But job-seeker fit is fundamentally a **graph problem**:

- Jobs relate to skills
- Skills relate to each other (Python -> NumPy -> data analysis)
- Companies cluster by industry, stage, and tech stack
- Users have latent preferences encoded across all of these dimensions

JobGraph makes that graph **explicit, learnable, and queryable**.

---

## Architecture

```
+-----------------------------------------------------------------+
|                    DATA INGESTION                                |
|                                                                  |
|  151 Companies (CSV) ──> Playwright + ATS API Parsers            |
|       │                  (Greenhouse, Lever, Ashby)              |
|       v                                                          |
|  5,112 raw job postings (HTML / JSON)                            |
|       │                                                          |
|       v                                                          |
|  Ollama + llama3.2 (local LLM, 96% extraction rate)             |
|       │  Extracts: title, skills, seniority,                     |
|       │  salary, location, company meta                          |
|       v                                                          |
|  481 Structured Job Records (Parquet)                            |
+-----------------------------+------------------------------------+
                              |
+-----------------------------v------------------------------------+
|                GRAPH CONSTRUCTION (PyG)                           |
|                                                                  |
|  Nodes: 481 Job + 427 Skill + 8 Company = 916 total             |
|                                                                  |
|  Edges:                                                          |
|    1,668 Job ──requires──> Skill                                 |
|      481 Job ──at────────> Company                               |
|      520 Skill ──cooccurs─> Skill                                |
|    ─────────────────────────────────                             |
|    2,669 total edges                                             |
+-----------------------------+------------------------------------+
                              |
+-----------------------------v------------------------------------+
|             GNN TRAINING (HGT)                                   |
|                                                                  |
|  - Heterogeneous Graph Transformer, 2 HGTConv layers             |
|  - Task: link prediction on job <-> skill edges (BCE loss)       |
|  - 100 epochs on Apple MPS                                       |
|  - Output: 256-dim embedding per node                            |
+-----------------------------+------------------------------------+
                              |
+-----------------------------v------------------------------------+
|            RETRIEVAL LAYER (FAISS + BM25)                        |
|                                                                  |
|  - FAISS IndexFlatIP: 481 L2-normalized vectors (256-dim)        |
|  - Hybrid: 0.7 x cosine similarity + 0.3 x BM25                |
|  - Query: encode free-text -> ANN search -> re-rank              |
+-----------------------------+------------------------------------+
                              |
+-----------------------------v------------------------------------+
|                  DEMO UI (Streamlit)                              |
|                                                                  |
|  - Free-text query or paste resume                               |
|  - Top-10 ranked jobs with skill overlap explanation             |
|  - Interactive skill match visualization                         |
+-----------------------------------------------------------------+
```

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/PhunsokNorboo/jobgraph
cd jobgraph
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# Install Ollama (free, local LLM -- no API key needed)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Configure environment
cp .env.example .env
# Edit .env if you want to use OpenAI instead of Ollama

# Run the full pipeline
bash scripts/run_scraper.sh 10       # scrape 10 companies (quick test)
bash scripts/run_extraction.sh       # extract structured data via LLM
bash scripts/build_graph.sh          # build heterogeneous knowledge graph
bash scripts/train_model.sh          # train HGT + build FAISS index

# Launch the demo
streamlit run app/app.py
```

---

## How It Works

### 1. Data Ingestion

An async Playwright scraper crawls public career pages across **151 companies** (65 via Greenhouse API, the rest via Lever, Ashby, and generic HTML parsing), collecting **5,112 raw job postings**.

A 500-job subset is then processed by a **local LLM extraction agent** (Ollama + llama3.2) that transforms unstructured postings into clean, structured records with a **96% extraction success rate** (481 of 500 jobs). Every record passes through Pydantic validation and skill normalization before storage as Parquet.

The LLM provider is swappable with a single environment variable -- set `LLM_PROVIDER=openai` to use GPT-4o instead, with zero code changes.

### 2. Knowledge Graph

Structured records are assembled into a **heterogeneous graph** using PyTorch Geometric (PyG):

| Node Type    | Count | Features                                       |
|-------------|-------|------------------------------------------------|
| **Job**     | 481   | 256-dim (384-dim text embedding + metadata, projected) |
| **Skill**   | 427   | 256-dim (384-dim text embedding + frequency, projected) |
| **Company** | 8     | 256-dim (384-dim text embedding, projected)    |

| Edge Type              | Count | Meaning                                 |
|-----------------------|-------|------------------------------------------|
| Job --requires--> Skill    | 1,668 | Job listing requires this skill      |
| Job --at--> Company        | 481   | Job belongs to this company          |
| Skill --cooccurs--> Skill  | 520   | Skills appear together frequently    |

Text embeddings produced by **all-MiniLM-L6-v2** (sentence-transformers), then projected to 256-dim with metadata features for the GNN input.

### 3. GNN Embeddings

A **Heterogeneous Graph Transformer (HGT)** with 2 HGTConv layers learns rich 256-dimensional embeddings for every node in the graph.

- **Training task**: Link prediction on job-skill edges (BCE loss)
- **Training**: 100 epochs on Apple MPS (MacBook), best validation MRR: 0.486
- **Why this works**: The model learns what skills "belong" to what kind of jobs -- exactly the knowledge needed for recommendation
- **Test performance**: MRR 0.290, Hits@10 62.7% -- the model ranks the correct skill in the top 10 nearly two-thirds of the time
- **Output**: Embeddings that capture skill relatedness, company similarity, and job clustering -- all in a single 256-dim vector space

### 4. Semantic Search

The retrieval layer combines graph-learned representations with lexical precision:

1. **Encode** the user's free-text query into the shared 256-dim embedding space (via sentence-transformers)
2. **ANN search** via FAISS IndexFlatIP over 481 L2-normalized job embeddings
3. **Hybrid re-ranking**: `0.7 * cosine_similarity + 0.3 * BM25_score` for lexical precision on titles and skills
4. **Skill overlap analysis** for each result -- not just "these jobs match" but "here is WHY they match"

---

## Results

### Pipeline Statistics

| Stage | Metric | Value |
|-------|--------|-------|
| Scraping | Companies crawled | 151 (65 via Greenhouse API) |
| Scraping | Raw job postings collected | 5,112 |
| Extraction | Jobs processed (subset) | 500 |
| Extraction | Successful extractions | 481 (96% success rate) |
| Extraction | LLM used | Ollama llama3.2 (local, free) |

### Graph Statistics

| Statistic | Value |
|-----------|-------|
| Job nodes | 481 |
| Skill nodes | 427 |
| Company nodes | 8 |
| **Total nodes** | **916** |
| Job --requires--> Skill edges | 1,668 |
| Job --at--> Company edges | 481 |
| Skill --cooccurs--> Skill edges | 520 |
| **Total edges** | **2,669** |
| Node feature dim | 256 |
| Avg skills per job | 3.5 |

### Model Performance

| Metric | Value |
|--------|-------|
| Model | HGT (2 HGTConv layers) |
| Training task | Link prediction (job-skill, BCE loss) |
| Training epochs | 100 |
| Hardware | Apple MPS (MacBook) |
| Best validation MRR | 0.486 |
| **Test MRR** | **0.290** |
| **Test Hits@10** | **62.7%** |

### Search Configuration

| Parameter | Value |
|-----------|-------|
| Index type | FAISS IndexFlatIP |
| Vectors indexed | 481 (L2-normalized, 256-dim) |
| Hybrid ranking | 0.7 x cosine + 0.3 x BM25 |
| Text encoder | all-MiniLM-L6-v2 (sentence-transformers) |

---

## Example Queries

**Query 1**: *"Senior backend engineer with Python and distributed systems experience"*

```
  Rank  Title                              Company       Skills Matched                         Score
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
   1    Senior Software Engineer           Stripe        Python, Distributed Systems, AWS        0.91
   2    Backend Engineer, Platform         Datadog       Python, Microservices, Kubernetes        0.88
   3    Staff Engineer, Infrastructure     Cloudflare    Python, Go, Distributed Systems          0.85
```

**Query 2**: *"Machine learning engineer, PyTorch, computer vision"*

```
  Rank  Title                              Company       Skills Matched                         Score
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
   1    ML Engineer, Vision                Scale AI      PyTorch, Computer Vision, Python         0.93
   2    Applied ML Engineer                Figma         PyTorch, Deep Learning, ML Ops           0.87
   3    Research Engineer                  Anthropic     PyTorch, Machine Learning, Python         0.84
```

**Query 3**: *"Frontend developer React TypeScript"*

```
  Rank  Title                              Company       Skills Matched                         Score
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
   1    Frontend Engineer                  Vercel        React, TypeScript, Next.js               0.92
   2    Senior Frontend Developer          Stripe        React, TypeScript, CSS                   0.89
   3    UI Engineer                        Figma         React, TypeScript, GraphQL               0.86
```

**Query 4**: *"DevOps engineer Kubernetes AWS CI/CD"*

```
  Rank  Title                              Company       Skills Matched                         Score
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
   1    Site Reliability Engineer          Datadog       Kubernetes, AWS, Terraform               0.90
   2    Platform Engineer                  Cloudflare    Kubernetes, CI/CD, Docker                0.87
   3    Infrastructure Engineer            Scale AI      AWS, Kubernetes, Linux                   0.83
```

*Scores are hybrid-ranked: 0.7 x GNN cosine similarity + 0.3 x BM25 lexical match. Skills shown are the top-3 overlapping skills between query and job posting.*

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Scraping | Playwright + ATS API parsers | Handles JS-rendered pages + Greenhouse/Lever/Ashby APIs |
| LLM Extraction | Ollama + llama3.2 | Free, local, no API key needed, 96% extraction rate |
| LLM Extraction (alt) | OpenAI GPT-4o | Drop-in swap via `LLM_PROVIDER=openai` env var |
| Data Validation | Pydantic v2 | Strict schema enforcement on every extracted record |
| Data Storage | Parquet (PyArrow) | Columnar, fast reads, portable |
| Graph Library | PyG (torch-geometric) | Best-in-class heterogeneous graph support |
| GNN Model | HGT (2x HGTConv layers) | Multi-type node/edge attention for job-skill-company |
| Text Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 384-dim embeddings for titles, skills, queries |
| Vector Index | FAISS (IndexFlatIP) | Exact cosine search over 256-dim embeddings |
| Hybrid Search | rank-bm25 | Lexical re-ranking for keyword precision |
| Demo UI | Streamlit | Fast to ship, interactive interface |

---

## Project Structure

```
jobgraph/
|
+-- README.md                    # This file
+-- requirements.txt             # All dependencies
+-- .env.example                 # Environment config template
+-- .gitignore
|
+-- data/
|   +-- companies.csv            # Seed list of 151 companies
|   +-- raw/                     # Raw scraped HTML/JSON
|   +-- processed/               # Structured job records (parquet)
|   +-- graph/                   # Serialized PyG graph + FAISS index
|
+-- scraper/
|   +-- __init__.py
|   +-- crawler.py               # Playwright async crawler + ATS API parsers
|   +-- job_page_detector.py     # Detects career page URL per company
|   +-- utils.py
|
+-- extraction/
|   +-- __init__.py
|   +-- schema.py                # Pydantic models (single source of truth)
|   +-- llm_agent.py             # LLM extraction with retry logic
|   +-- llm_client.py            # Provider abstraction (Ollama / OpenAI)
|   +-- prompts.py               # Extraction prompt templates
|   +-- postprocess.py           # Skill normalization + deduplication
|
+-- graph/
|   +-- __init__.py
|   +-- builder.py               # Constructs PyG HeteroData object
|   +-- features.py              # Node feature engineering (256-dim)
|   +-- visualize.py             # Graph visualization (networkx)
|
+-- model/
|   +-- __init__.py
|   +-- hgt.py                   # HGT model definition (2x HGTConv)
|   +-- train.py                 # Training loop (link prediction, BCE)
|   +-- evaluate.py              # MRR, Hits@K evaluation
|   +-- embed.py                 # Generate + save all embeddings
|
+-- retrieval/
|   +-- __init__.py
|   +-- index.py                 # Build FAISS IndexFlatIP from embeddings
|   +-- search.py                # Hybrid query pipeline (FAISS + BM25)
|   +-- resume_parser.py         # Parse resume PDF -> query vector
|
+-- app/
|   +-- __init__.py
|   +-- app.py                   # Streamlit demo app
|   +-- components/
|       +-- __init__.py
|       +-- job_card.py          # Job result card component
|       +-- skill_chart.py       # Skill overlap visualization
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb
|   +-- 02_graph_analysis.ipynb
|   +-- 03_embedding_visualization.ipynb
|
+-- scripts/
    +-- run_scraper.sh
    +-- run_extraction.sh
    +-- build_graph.sh
    +-- train_model.sh
```

---

## Setup Guide

### Prerequisites

- Python 3.11+
- Node 18+ (for Playwright browser automation)
- Ollama (free, local LLM) -- or an OpenAI API key

### Installation

```bash
git clone https://github.com/PhunsokNorboo/jobgraph
cd jobgraph

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium

# Set up environment
cp .env.example .env
```

### LLM Setup

**Option A: Ollama (free, default)**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3.2 (3B parameters, ~2GB download)
ollama pull llama3.2

# Ollama runs at http://localhost:11434 -- no API key needed
```

**Option B: OpenAI GPT-4o (paid, higher extraction quality)**

Edit `.env`:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

No code changes required -- the provider abstraction handles routing.

### Environment Variables

```bash
# .env.example
LLM_PROVIDER=ollama              # "ollama" or "openai"
OLLAMA_MODEL=llama3.2            # Model name for Ollama
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=                  # Only needed if LLM_PROVIDER=openai
```

### Running the Pipeline

```bash
# 1. Scrape job postings from public career pages
bash scripts/run_scraper.sh [num_companies]

# 2. Extract structured data from raw HTML using LLM
bash scripts/run_extraction.sh

# 3. Build the heterogeneous knowledge graph
bash scripts/build_graph.sh

# 4. Train HGT model and generate embeddings + FAISS index
bash scripts/train_model.sh

# 5. Launch the Streamlit demo
streamlit run app/app.py
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **PyG for graphs** | PyTorch Geometric is the standard for heterogeneous GNNs, co-authored by Jure Leskovec |
| **HGT over GraphSAGE** | HGT handles multi-type nodes/edges with attention -- critical for job-skill-company relationships |
| **Link prediction training** | Forces the model to learn skill-job affinity -- the core signal for recommendation |
| **FAISS + BM25 hybrid** | Vector search captures semantic similarity; BM25 catches exact keyword matches |
| **Ollama + llama3.2 as default** | Zero cost, runs locally, 96% extraction rate, no API key friction |
| **Pydantic validation** | Every extracted record is validated before entering the pipeline -- data quality by construction |

---

## License

MIT

---

*Built as a portfolio project for the HiringCafe Founding Engineer role, demonstrating end-to-end AI search and graph-based recommender system engineering.*
*All job data scraped from public career pages. No private or proprietary data used.*
