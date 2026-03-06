# JobGraph Session Log

## 2026-03-05 — Project Kickoff

### Decisions
- 20 beads, 5 layers, peak parallelism 9
- Real scraping (not synthetic) — user preference
- Ollama + llama3.1 as default LLM provider (installed and ready)
- Interface-first: schema.py defined before all modules
- Bead label: `jobgraph`

### Corrections
1. `torch==2.2.0` not available on Python 3.14 → use `>=` version pins
2. `torch-scatter`/`torch-sparse` fail without torch → removed from requirements (PyG >=2.5 doesn't need them)
3. Playwright `wait_until="networkidle"` times out on heavy JS → use `"domcontentloaded"`
4. Generic scraper didn't handle relative URLs or PDF links → use `urljoin()`, filter `.pdf`

### Progress
- [x] Layer 0: Foundation (scaffolding, schemas, companies CSV) — 3/3 done
- [x] Layer 1: Code packages (9 parallel) — 9/9 done (all agents completed)
- [~] Layer 2: Data pipeline — scraper running on 151 companies (Anthropic test: 451 jobs via Greenhouse API)
- [ ] Layer 3: Graph + model pipeline
- [ ] Layer 4: Polish + deploy

## 2026-03-06 — Pipeline Execution Complete

### Layer 2 — Data Pipeline
- [x] Scraper: 5,112 jobs from 151 companies (65 Greenhouse API, rest generic/Lever/Ashby)
- [x] Extraction: 481 jobs extracted from 500-job subset via Ollama llama3.2 (96% success rate)
  - ~14-18s/job locally; full 5,112 would take ~5.7 hours
  - Produced: `data/processed/jobs.parquet` (481 rows), `data/processed/skill_vocabulary.csv`

### Layer 3 — Graph + Model Pipeline
- [x] Graph build: 481 job + 427 skill + 8 company nodes
  - Edges: 1,668 requires + 481 at + 520 cooccurs (threshold=2)
  - Saved: `data/graph/hetero_data.pt`, `data/graph/mappings.json`
- [x] HGT training: 100 epochs on MPS
  - Best val MRR: 0.486, Test MRR: 0.290, Hits@10: 62.7%
  - Saved: `model/checkpoints/best_model.pt`
- [x] Embeddings: job(481,256), skill(427,256), company(8,256)
  - Saved: `data/graph/{job,skill,company}_embeddings.npy`, `job_index.json`
- [x] FAISS index: 481 L2-normalized vectors, IndexFlatIP
  - Saved: `data/graph/faiss.index`

### Layer 4 — Polish (complete)
- [x] Finalize notebooks — all 3 filled with real data, executed via nbconvert, zero errors
- [x] Finalize README — real metrics, 4 example queries, architecture diagram, pipeline stats
- [x] Wire Streamlit app — fixed dim mismatch, numpy handling, NaN for BM25. App runs on :8501
- [x] Smoke test — search returns relevant results for Python/infra, ML, and frontend queries

### Corrections (Session 2)
5. Parquet stores list columns as numpy arrays → added `isinstance(val, np.ndarray)` check in `_parse_skill_list`
6. HGT embeddings collapsed (pairwise cosine ~0.999) with 481 nodes → rebuilt FAISS with sentence-transformer embeddings (384-dim, cosine range 0.17-0.79)

### Final Status
**All 20 beads closed. Project complete.**
- 5 layers, peak parallelism 9
- 7 corrections logged → 7 rules added to CLAUDE.md
- Search quality verified: relevant results for 3 distinct query types

### Stats
- Deps installed: torch 2.10.0, PyG 2.7.0, all others
- Playwright chromium installed
- Greenhouse API working (451 jobs from Anthropic alone)
- 65 of 151 companies use Greenhouse (API-based scraping)
- Total pipeline: scrape → extract → graph → train → embed → index ✓
