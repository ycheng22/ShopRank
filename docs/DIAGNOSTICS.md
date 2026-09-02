# Diagnostics & Failure Analysis

This document tracks known failure modes, hard negatives, and diagnostic analyses of the ShopRank retrieval pipeline.

## Known Hard Negatives (M4)

The following are top-ranked hard negatives (Rank=1) mined from the `us` locale dev set across three relevance categories (S/C/I). These examples represent cases where the BM25 + Dense hybrid retrieval currently struggles or misranks items.

### Substitute (S) Misrank
* **Query:** `the not a cat cat`
* **High-ranked Negative:** `I'm Here Live I'm Not A Cat: I'm Here Live I'm Not A Cat Fun`
* **Analysis:** The top result is a parody/merchandise item referencing a meme, rather than a genuine cat-related product. The exact word overlap ("Not A Cat") strongly triggers lexical retrieval.

### Complement (C) Misrank
* **Query:** `under a white sky: the nature of the future`
* **High-ranked Negative:** `Summary and Analysis of Under a White Sky: The Nature of the`
* **Analysis:** The top result is a study guide/summary rather than the original book. Because the titles are nearly identical, lexical and dense retrieval both score this extremely high.

### Irrelevant (I) Misrank
* **Query:** `swan pillow`
* **High-ranked Negative:** `ROYAL SWAN Bed Pillows for Sleeping 2 Pack,Best Neck Support`
* **Analysis:** The brand name contains "SWAN", which causes a lexical hit for "swan", but the product is a standard neck pillow, not a pillow shaped like a swan.

---

## Operational Incident (2026-09-01): PostgreSQL TOAST Bloat & Transient Storage Surge

### Summary
During the Milestone M6 dimension ablation study (`scripts/run_dim_ablation.py`), project storage surged from an initial ~266 MB baseline to a peak of **~480 MB**, dangerously close to Neon's 500 MB hard project quota.

### Root Cause Analysis (RCA)
1. **PostgreSQL MVCC & TOAST Dead Tuples**:
   - `pgvector` float arrays (512 to 1024 dimensions = 2,048 to 4,096 bytes/row) exceed inline page tuple limits and are stored in PostgreSQL's out-of-line **TOAST** (`The Oversized-Attribute Storage Technique`) relation.
   - The ablation script sequentially populated 26,487 products across three vector dimensions:
     - Iteration 1: 512-dim vectors + HNSW index
     - Iteration 2: 1024-dim vectors + HNSW index (peaked at 486 MB)
     - Iteration 3: 768-dim vectors + HNSW index
   - Under PostgreSQL's MVCC model, dropping or altering the vector column and rewriting values does **not** free physical disk space on the filesystem. The old TOAST chunks from iterations 1 and 2 remained as dead tuples.
   - Querying `pg_class` revealed that the `products` TOAST table had swollen to **252 MB**, harboring over **145 MB of dead version bloat** (active 768-dim vectors strictly require ~81 MB).
2. **Serverless Write-Ahead Log (WAL) Accounting**:
   - Neon's serverless storage architecture meters active storage across Pageservers and Safekeepers including WAL write spikes within the Point-In-Time Restore retention window, causing the web console to reflect this transient surge.

### Remediation
Executed a physical table and index compaction:
```sql
VACUUM FULL products;
```

**Outcome**:
| Metric | Before VACUUM FULL | After VACUUM FULL | Delta |
| --- | --- | --- | --- |
| **Products TOAST Table** | 252 MB | 107 MB | **-145 MB (-57.5%)** |
| **Products Total Size** | 418 MB | 243 MB | **-175 MB (-41.9%)** |
| **Entire Database Size** | 430 MB (480 MB peak) | **254 MB** | **-176 MB (-40.9%)** |

### Permanent Preventative Actions
1. **Post-Migration Compaction**: Any batch re-indexing script that executes multiple vector updates must append `VACUUM FULL <table_name>;` upon completion.
2. **Quota Headroom Assurance**: At 254 MB total database size (including 103 MB HNSW index, 15 MB GIN text index, and 48 demo cache records), ShopRank operates safely within 50.8% of the 500 MB free quota, leaving ~246 MB of headroom.

