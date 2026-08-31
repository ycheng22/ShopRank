# Architecture & Technical Decisions

This document records key technical decisions, environment configurations, and dependency trade-offs.

## Local Hardware & Deep Learning Runtime Environment

- **GPU Model**: NVIDIA GeForce RTX 2060 SUPER
- **VRAM**: 8192 MiB (8 GB)
- **Driver Version**: 595.95
- **CUDA Version**: CUDA 12.4 (Driver supports up to CUDA 13.2)
- **PyTorch Version**: 2.6.0+cu124
- **vLLM Version**: N/A (Official wheels do not support native Windows; recommended to use WSL2 or local serving via Ollama/llama.cpp)


## Model providers
- OpenAI API 
- Gemeni 
- DeepSeek 
- Alibaba Model 

## Batch size decision
```json
{
  "date": "2026-08-30",
  "device": "NVIDIA GeForce RTX 2060 SUPER",
  "max_len": 512,
  "total_products": 50000,
  "results": [
    {
      "batch_size": 32,
      "items_per_s": 88.7,
      "peak_gib": 1.74
    },
    {
      "batch_size": 64,
      "items_per_s": 71.0,
      "peak_gib": 2.24
    },
    {
      "batch_size": 128,
      "items_per_s": 70.4,
      "peak_gib": 3.33
    },
    {
      "batch_size": 256,
      "items_per_s": 69.6,
      "peak_gib": 5.58
    },
    {
      "batch_size": 512,
      "items_per_s": 13.7,
      "peak_gib": 10.1
    }
  ],
  "chosen": 32,
  "note": "Throughput reached its plateau (knee of the curve). Chosen value naturally leaves a memory safety buffer."
}
```

## Cloud region decision
- All cloud resources should prioritize us-east region.
- In case of resource shortage, use us-west region.

## Local deployment vs Docker deployment
Local deployment is preferred for development and testing, while Docker deployment is preferred for production.

## GCP artifactory registry path
- us-east1-docker.pkg.dev/gcp-share-507118/containers

## Split search into cacheable 
| 2026-09-02 | Split search into cacheable `GET` (preset examples) and `POST` (free-form, BYO key); CORS allow-list from Settings, never `*` | Frontend on Cloudflare Pages and API on Cloud Run are different origins. A cross-origin POST always triggers a preflight, doubling latency on a cold start — which is exactly the first impression a recruiter gets. GET keeps preset queries CORS-simple and CDN-cacheable, so they render even while the service is cold or down. | Applies to SPEC 22.5 |

## Dataset Version
- **Dataset**: Amazon ESCI Dataset (shopping_queries_dataset)
- **Version/Commit**: `7916cdf6ab75a462e77f20ab40428a10923998d5`
- **Note**: Every evaluation result must log this commit hash in the `dataset_version` field of the `eval_runs` table to ensure metrics comparability.

## ESCI variant frozen
| 2026-09-XX | ESCI variant frozen: small_version=1 AND product_locale='us' | 29,844 queries in pool |
| queries = 1,500 (900/300/300) | dedup is ~nil (19.3 unique products/query), so 1,500 queries ≈ 29k products, just under the 30k cap. Chosen so the cap never binds — a binding cap would drop whole queries and distort the strata. |
| difficulty_bin: hard iff E-fraction <= 0.312 | 40th pct measured on this variant. HARDCODED — do not recompute from the sample. Note: the same percentile on the full (non-small_version) US set is 0.688, which is why the variant must be frozen. |
| candidate_size_bin: large iff candidates >= 16 (median), NOT 20 | only 19.5% of queries have >=20 candidates, making the 40/60 target unreachable at the original threshold |
| has_complement: pool has 25.8%, target >=20% reachable by mild oversampling |

## Database Driver
- **Driver**: `asyncpg`
- **Reason**: The project enforces "async-first" execution. While `psycopg2` was present, it is synchronous. `asyncpg` is the standard, high-performance async driver for Postgres in modern Python. Added for Milestone M2 to support `BM25Retriever`.

## Indexing Engineering Metrics (M3a/M3b)
| 2026-08-31 | Index Build Time: 2811s \| Peak GPU Memory: 7283 MiB \| Table Size: 266 MB | Confirmed that processing 26,487 queries fits comfortably within the 0.5 GB Neon quota limit. No dimensionality reduction to 512 is required. Re-indexing leverages local disk `.cache` allowing the build script to finish in seconds without rewriting embeddings when vectors are present. |

## Dense Retrieval & Fusion Ablation (M3b)
- **Dense vs BM25**: Pure Dense retrieval explicitly dominates BM25, driving Recall@50 up by a massive **+12.7%** (from 50.9% to 63.6%). The candidate net is successfully catching half of the relevant items BM25 missed.
- **RRF Hybrid Superiority**: Fusing the BM25 exact-match signals with the Dense semantic signals using Reciprocal Rank Fusion yielded the best overall pipeline. Hybrid pushed **NDCG@10 to 0.5180** and **MRR@10 to 0.7538** without sacrificing the 63.3% recall, proving semantic matching effectively re-ordered the top 10 results. The pipeline concurrently executes in **~279ms (p95)**, satisfying the 800ms budget constraint.

