# Changelog

All notable changes, architectural decisions, feature implementations, and evaluation metrics for **ShopRank** are documented in this file.

---

## [Milestone M6 / Day 2] - 2026-09-01

### Added
- **Dimension Ablation Script (`scripts/run_dim_ablation.py`)**: Automates the extraction, truncation (1024 → 768 → 512), re-indexing, and evaluation of BGE-M3 Matryoshka embeddings.
- **RESTful API Surface (`app/routes/`)**:
  - `GET /api/examples`: Returns the list of 8 hard-coded preset demo queries and their descriptions.
  - `GET /api/search`: Dedicated cache-hit endpoint for preset queries. Fetches precomputed multi-stage search results from `demo_cache` with zero LLM/compute overhead.
  - `POST /api/search`: The main free-form query endpoint. Parses user configuration (`ShopRankPipelineConfig`), connects to Neon PG, executes BM25/Dense/Fusion/Rerank, and returns Explainable hits. Includes a daily token quota circuit breaker.
  - `GET /api/ablation`: Returns the raw evaluation metrics (JSON) for the current dimensions.
- **Demo Cache Layer (`demo_cache` table)**: Precomputes and stores full search result JSONs for all 8 preset queries across 6 different pipeline configs to bypass CPU reranker latency on Cloud Run and avoid free-tier quota exhaustion.
- **Angular Frontend (`web/`)**:
  - Upgraded to modern **Angular 21 (v21.2.22)** with TypeScript 5.9.3 and `@angular/build:application`.
  - Built `SearchPageComponent` for interactive query submission.
  - Built `ResultRowComponent` with expandable `ScoreBreakdownComponent` exposing BM25, Dense, RRF, and Rerank score provenance.
  - Configured `environment.prod.ts` via build-time injection for dynamic API routing.
- **Local Dev Orchestrator (`start_local.ps1`)**: Single PowerShell launcher to concurrently start both the FastAPI backend (`uvicorn`) and the Angular 21 dev server (`ng serve`).
- **Cloudflare Pages Deployment**: Added `cloudflare/pages-action` to `.github/workflows/deploy.yml` for serving the static Angular bundle globally.
- **Rich Product Metadata & UI Presentation (`app/routes/search.py`, `web/`)**:
  - Enriched both cached preset and free-form search results with `title`, `description` snippet, and `product_text`.
  - Upgraded Angular UI result cards to display human-readable product titles, formatted ASIN badges, context snippets, and full expandable catalog descriptions alongside the score provenance breakdown.
- **Demo Scenarios Documentation (`docs/DEMO-SCENARIOS.md`)**: Documents the rationale behind the 8 preset queries (e.g. Exact Match, Long-tail, Misspelling, Cross-lingual).

### Changed
- **Settings & Config (`app/settings.py`)**: Added CORS origin allow-lists (no `*`), `DAILY_TOKEN_QUOTA`, and API Key header validation.
- **Schema (`scripts/schema.sql`)**: Appended the `demo_cache` table structure.
- **README & DECISIONS**: 
  - Logged the findings of the Matryoshka truncation ablation.
  - 768-dim was officially chosen as the production baseline (achieving NDCG@10 of 0.5179 while keeping DB size at 418 MB, safely below the 500 MB limit).

### Fixed
- Neon DB Idle connection drop during the 45-minute vector caching phase of ablation by deferring `asyncpg.connect()` until immediately before DB writes.
- **PostgreSQL TOAST Bloat Mitigation**: Reclaimed ~176 MB of dead vector tuple space left by sequential 512/1024/768 ablation passes via `VACUUM FULL products;`, shrinking active database footprint from 430 MB (480 MB transient peak) to **254 MB** (documented in `docs/DIAGNOSTICS.md`).

---

## [Milestone M4 + M5] - Reranking, Hard Negative Mining & Multi-Lingual Evaluation (Current Phase)

### Added
- **Cross-Encoder Reranker (`core/rerankers/cross_encoder.py`)**:
  - Integrated `BAAI/bge-reranker-v2-m3` as a singleton cross-encoder for second-stage candidate reranking on Top-30 hits.
  - Full preservation of `ScoreBreakdown` data contracts: attaches `rerank_score` and records `rank_before_rerank` without mutating prior retrieval scores.
- **LLM Provider & Translation Layer (`providers/`)**:
  - `providers/deepseek.py`: Implemented primary translation client via DeepSeek API (`deepseek-chat`).
  - `providers/alibaba.py`: Implemented fallback translation client via Alibaba DashScope (`qwen-plus`).
  - `providers/translation.py`: Multi-provider router with automatic failover and exponential backoff retry.
- **Multi-Lingual Query Generation & Ingestion**:
  - `scripts/translate_queries.py`: Translated 300 dev queries to Chinese (`zh`) and French (`fr`) with strict `asyncio.Semaphore(20)` rate-limiting.
  - `scripts/ingest_translations.py`: Ingested 585 translated queries into Neon PostgreSQL with locale metadata.
- **Hard Negative Mining Engine (`scripts/mine_hard_negatives.py`)**:
  - Automated mining of top-ranked misclassifications across Substitute (S in Top 1), Complement (C in Top 5), and Irrelevant (I in Top 10) categories.
  - Extracted 292 hard negatives (53 S, 69 C, 170 I) and documented root causes in `docs/DIAGNOSTICS.md`.
- **RRF Parameter Diagnostics (`evals/ablation.py --rrf-diagnostic`)**:
  - Automated sweeps across `k=10, 60, 200` to evaluate rank decay impact on recall and NDCG.
- **Automated Test Suite**:
  - Added comprehensive unit tests in `tests/test_reranker.py`, `tests/test_translation.py`, and `tests/test_fusion.py` (total 34 tests passing).

### Changed & Refactored
- **Architecture & Configuration Guardrails**:
  - Removed all illegal `os.environ` / `load_dotenv` calls across `scripts/` and `core/`; enforced strict dependency injection via `app/settings.py`.
  - Aligned `.env.example` to strictly match the typed `Settings` model.
  - Updated `README.md` with full Mermaid pipeline flowcharts, 5-row ablation metrics, cross-tab results, and system benchmarks.

### Key Decisions (`docs/DECISIONS.md`)
- **CPU Reranker Latency Budget**:
  - Measured `BAAI/bge-reranker-v2-m3` CPU latency: Top-50 = ~9,652ms (p95), Top-30 = ~6,057ms (p95).
  - *Decision*: Online synchronous free-form queries route through Hybrid retrieval (~364ms) to stay within the <800ms SLA. Cross-encoder reranking is reserved for CDN-cached pre-computed examples or GPU/ONNX deployments.
- **Multi-Lingual Model Routing**:
  - Employed low-cost tier (DeepSeek with Qwen fallback) for batch translation to preserve project infrastructure budget.

### Metrics Achieved
- **5-Row Ablation Table (Dev Split)**:
  - BM25 Baseline: NDCG@10 = `0.4174`, Recall@50 = `0.5097`, MRR@10 = `0.6616`, p95 = `172.4ms`
  - +Dense (BGE-M3): NDCG@10 = `0.4937`, Recall@50 = **`0.6350`** (+12.5% recall lift), MRR@10 = `0.7234`, p95 = `275.2ms`
  - +Hybrid (RRF): NDCG@10 = `0.5132`, Recall@50 = `0.6323`, MRR@10 = `0.7463`, p95 = `364.0ms`
  - +Rerank: NDCG@10 = **`0.5810`** (+16.4% over BM25), Recall@50 = `0.5706`, MRR@10 = **`0.7939`**, p95 = `59,302ms`
  - +Weighted: NDCG@10 = `0.5777`, Recall@50 = `0.5617`, MRR@10 = `0.7896`, p95 = `58,651ms`
- **Cross-Lingual NDCG@10 (3 Locales × 4 Configs)**:
  - EN: `0.4174` (BM25) → `0.4937` (Dense) → `0.5132` (Hybrid) → **`0.5810`** (Hybrid+Rerank)
  - ZH: `0.0155` (BM25) → `0.3013` (Dense) → `0.2998` (Hybrid) → **`0.3821`** (Hybrid+Rerank)
  - FR: `0.0975` (BM25) → `0.3585` (Dense) → `0.2710` (Hybrid) → **`0.3982`** (Hybrid+Rerank)

---

## [Milestone M3a & M3b] - Dense Embeddings, Indexing & Hybrid Rank Fusion

### Added
- **Embedding Pipeline (`core/embeddings.py`, `scripts/build_index.py`)**:
  - Implemented BGE-M3 embedding generation with Matryoshka dimension truncation to 768-dim.
  - Disk-backed embedding cache to guarantee idempotent, zero-cost rebuilds.
- **Dense Vector Store (`core/retrievers/dense.py`)**:
  - PostgreSQL `pgvector` HNSW index with cosine distance operator `<=>` and configurable `ef_search`.
- **Hybrid Fusion Engine (`core/fusion.py`)**:
  - Reciprocal Rank Fusion (RRF) and min-max normalized weighted linear fusion.
- **Dual-Retriever Pipeline (`core/pipeline.py`)**:
  - Orchestrated parallel async execution of BM25 and Dense retrievers.

### Key Decisions
- **Dimension Truncation**: Truncated 1024-dim BGE-M3 embeddings to 768-dim to preserve Neon storage quota (<500MB).
- **GPU Batching**: Determined optimal batch size of 32 on RTX 2060 SUPER (88.7 items/s, peak VRAM 7,283 MiB).

### Metrics Achieved
- Index build time: 2,811s for 26,487 unique products.
- Storage footprint: 266 MB (well within 0.5 GB quota).
- Dense Recall@50: **0.6350** (+12.5% over BM25 0.5097).
- Hybrid NDCG@10: **0.5132** (p95 latency = 364ms).

---

## [Milestone M2] - Evaluation Harness & BM25 Baseline

### Added
- **Evaluation Runner (`evals/runner.py`)**:
  - Structured evaluation execution over stratified dataset splits with latency tracking and metric computation.
- **Deterministic Metrics Suite (`evals/metrics.py`)**:
  - Implementations for NDCG@k, Recall@k, and MRR@k adhering strictly to ESCI gain mapping: E=3, S=2, C=1, I=0.
- **PostgreSQL BM25 Lexical Retriever (`core/retrievers/bm25.py`)**:
  - `tsvector` English text search on title + description.

### Key Decisions
- **Relevance Gain Convention**: Hardcoded linear mapping `Exact=3, Substitute=2, Complement=1, Irrelevant=0` (gain > 0 defined as relevant).
- **Unlabelled Queries**: Queries with empty qrels skipped rather than scored as 0 to prevent uniform score degradation.

### Metrics Achieved
- BM25 Baseline (Dev): NDCG@10 = `0.4174`, Recall@50 = `0.5097`, MRR@10 = `0.6616`, p95 latency = `172.4ms`.

---

## [Milestone M1] - Data Stratification & Corpus Ingestion

### Added
- **Stratified Dataset Sampling (`scripts/sample_dataset.py`)**:
  - Sampled 1,500 queries from Amazon ESCI (`small_version=1`, `product_locale='us'`) across 5 dimensions: query length, difficulty, complement presence, candidate size, and language.
  - Split: 900 train / 300 dev / 300 test with fixed random seed 42.
- **Database Schema (`scripts/schema.sql`)**:
  - PostgreSQL schema for `products`, `queries`, and `qrels` with pgvector extension.

### Key Decisions
- **Corpus Cap**: Bound corpus to 30,000 products (~19.3 unique products per query) to ensure entire dataset fits comfortably within Neon free tier.
- **Full-Corpus Setting**: Evaluation conducted against all 30k products rather than isolated candidate pools for realistic production simulation.

---

## [Milestone M0a & M0b] - Architecture Skeleton & Deployment Topology

### Added
- **Domain Models (`core/models.py`)**: Pydantic v2 domain definitions (`Query`, `Product`, `ScoredHit`, `SearchResponse`, `ScoreBreakdown`, `PipelineConfig`).
- **Application & Routing (`app/main.py`, `app/routes/`)**: FastAPI application with `/healthz` and `/api/search` endpoints.
- **Configuration Hub (`app/settings.py`)**: Centralized, typed `Settings` instance.
- **Infrastructure & Containerization**: Multi-stage `Dockerfile`, `docker-compose.yml`, and GitHub Actions CI workflow.

### Key Decisions
- **Driver**: `asyncpg` adopted for async-first PostgreSQL performance.
- **Deployment Topology**: Frontend hosted on Cloudflare Pages (`https://shoprank.vectorlab.me`), backend on GCP Cloud Run (`*.run.app`).
- **Liveness Safety**: `/healthz` strictly reports memory/process status without touching the database to prevent Neon compute wakeups.
