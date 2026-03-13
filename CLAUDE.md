# JobGraph — Project Rules

## What Is This
GNN-powered job discovery engine for the HiringCafe Founding Engineer portfolio project. Transforms scraped job postings into a heterogeneous knowledge graph, trains HGT for embeddings, powers semantic job search.

## Architecture
- `extraction/schema.py` is the **single source of truth** — ALL Pydantic models, shared types, column constants live here. Every module imports from schema.py.
- 6 code packages: scraper, extraction, graph, model, retrieval, app — each independent, none import from each other.
- LLM provider: Ollama + llama3.2 (default, free). Swap to OpenAI via `LLM_PROVIDER=openai` env var.
- Graph: PyG HeteroData with Job/Skill/Company nodes. HGT model for embeddings.
- Search: FAISS cosine similarity (384-dim sentence-transformer) + BM25 re-ranking. Filters: seniority, role family, location.
- Demo: Streamlit with search + filters, skill gap roadmap, knowledge graph viz, market insights tab.

## Key Paths
- Schemas: `extraction/schema.py`
- Session log: `.notes/session_log.md`
- Project plan: `JOBGRAPH_PROJECT.md`
- Data: `data/raw/` (scraped), `data/processed/` (parquet), `data/graph/` (PyG + embeddings)

## Rules
- **Schema first**: Never add a field to any module without adding it to `extraction/schema.py` first.
- **No cross-imports**: Code packages (scraper, extraction, graph, model, retrieval, app) NEVER import from each other. They only import from `extraction/schema.py` and standard libraries.
- **Lazy-load API clients**: Never initialize Ollama/OpenAI clients at module level. Use lazy init so tests work without a running LLM server.
- **Pydantic validation**: All extracted job data must pass through `JobRecord` validation before being saved.
- **Skill normalization**: Always normalize skills through `postprocess.py` SKILL_ALIASES before storing.
- **Use `'Int64'` not `int`**: When converting pandas columns that may contain NaN (salary_min, salary_max), use nullable `'Int64'`.
- **Respect robots.txt**: Scraper must check robots.txt before crawling. 1-2s delay between requests per domain.
- **User-Agent header**: Always identify the bot in requests.
- **Retry with backoff**: LLM extraction gets 3 retries. Log failures, don't crash.
- **File paths**: Always resolve relative to `Path(__file__).resolve().parent`, never hardcode absolute paths.
- **Venv**: Project uses `.venv/` in project root. Always use `source .venv/bin/activate` or `.venv/bin/python` before running.
- **Version pins**: Use `>=` minimum version pins in requirements.txt, not `==` — exact pins break on newer Python versions.

## Correction Log
| Date | Issue | Fix | Rule Added |
|------|-------|-----|------------|
| 2026-03-05 | Pinned `torch==2.2.0` not available on Python 3.14 | Use `>=` version pins in requirements.txt, not `==` | Use `>=` minimum version pins — exact pins break on newer Python versions |
| 2026-03-05 | `torch-scatter`/`torch-sparse` fail to build without torch installed first | Remove from requirements.txt — PyG >=2.5 works without them | Don't include torch extension packages in requirements.txt — install separately after torch if needed |
| 2026-03-05 | Playwright `wait_until="networkidle"` times out on heavy JS sites (Apple, Amazon, Microsoft) | Use `"domcontentloaded"` instead | Always use `wait_until="domcontentloaded"` for Playwright — `"networkidle"` hangs on heavy sites |
| 2026-03-05 | Generic scraper didn't handle relative URLs (`./jobs/results`) or PDF links | Use `urljoin()` for relative URLs, skip `.pdf` links | Always resolve relative URLs with `urljoin()` and filter non-HTML links |
| 2026-03-06 | Parquet stores list columns as numpy arrays, not Python lists — `_parse_skill_list` missed them | Added `isinstance(val, np.ndarray)` check with `.tolist()` | Always handle numpy arrays when reading list columns from parquet |
| 2026-03-06 | HGT embeddings collapsed (pairwise cosine ~0.999) — all jobs look identical, search returns random results | Rebuilt FAISS index with sentence-transformer embeddings (384-dim) directly — cosine variance 0.17-0.79 | With small datasets (<1K nodes), GNN embeddings often collapse. Use pretrained text embeddings for search; keep GNN for the portfolio demonstration. |
| 2026-03-12 | Job card HTML rendered as code block — indented `<span>` tags inside f-string treated as markdown code | Build entire card as one HTML string with no indentation, single `st.markdown()` call | Never indent HTML inside Streamlit `st.markdown()` f-strings — 4+ spaces triggers markdown code blocks |
| 2026-03-12 | Skill chart x-axis labels overlapped for long skill names | Added `tickangle=-45`, truncated names to 25 chars, increased margins | For Plotly bar charts with text labels: rotate ticks, truncate long names, add generous margins |
| 2026-03-12 | Seniority filter returned 0 results — FAISS top-5x candidates didn't include rare seniority levels (only 4 entry jobs) | Search all FAISS vectors when filters are active, stop early once enough candidates found | When filtering FAISS results by metadata, search the full index — rare categories may not appear in top-N |
| 2026-03-12 | Streamlit hot-reload didn't pick up code changes to imported modules | Must kill and restart `streamlit run` process for module-level changes | Always restart Streamlit after modifying imported modules — hot-reload only works for the main script |

## Notes
- Update `.notes/session_log.md` after every correction or PR
- Update this CLAUDE.md after every mistake — add a rule so it never happens again
