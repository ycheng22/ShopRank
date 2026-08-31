import os
import json
import time
import uuid
import asyncio
import asyncpg
import numpy as np
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel
from retrieval_core.models import PipelineConfig, Query
from core.retrievers.bm25 import BM25Retriever
from evals.metrics import ndcg_at_k, recall_at_k, mrr_at_k
from app.settings import get_settings

class EvalResult(BaseModel):
    run_id: str
    config: PipelineConfig
    dataset_version: str
    split: str
    ndcg_10: float
    recall_50: float
    mrr_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost: float
    skipped_count: int

async def load_split_queries(pool: asyncpg.Pool, split: str, limit: int = None) -> list[tuple[str, Query]]:
    sql = "SELECT query_id, text, locale FROM queries WHERE split = $1 ORDER BY query_id"
    if limit:
        sql += f" LIMIT {limit}"
    rows = await pool.fetch(sql, split)
    return [(r["query_id"], Query(text=r["text"])) for r in rows]

async def load_qrels(pool: asyncpg.Pool) -> dict[str, dict[str, str]]:
    rows = await pool.fetch("SELECT query_id, product_id, esci_label FROM qrels")
    qrels = defaultdict(dict)
    for r in rows:
        qrels[r["query_id"]][r["product_id"]] = r["esci_label"]
    return dict(qrels)

async def init_gatemark_db(pool: asyncpg.Pool):
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            config JSONB,
            dataset_version TEXT,
            split TEXT,
            ndcg_10 FLOAT,
            recall_50 FLOAT,
            mrr_10 FLOAT,
            latency_p50_ms FLOAT,
            latency_p95_ms FLOAT,
            cost FLOAT,
            skipped_count INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

async def run(config: PipelineConfig, split: str = "dev", dataset_version: str = "unknown", allow_test: bool = False, limit: int = None) -> EvalResult:
    if split == "test" and not allow_test:
        raise ValueError("Reading the 'test' split requires allow_test=True")
        
    settings = get_settings()
    print("Connecting to Shoprank pool...")
    shoprank_pool = await asyncpg.create_pool(settings.database_url)
    
    gatemark_pool = None
    if settings.gatemark_database_url:
        print("Connecting to Gatemark pool...")
        gatemark_pool = await asyncpg.create_pool(settings.gatemark_database_url)
        await init_gatemark_db(gatemark_pool)
        
    try:
        print("Loading queries...")
        queries = await load_split_queries(shoprank_pool, split, limit)
        print(f"Loaded {len(queries)} queries. Loading qrels...")
        qrels_all = await load_qrels(shoprank_pool)
        
        print("Initializing retriever...")
        bm25_retriever = BM25Retriever(settings.database_url)
        await bm25_retriever._get_pool()
        
        print("Running evaluation loop...")
        
        latencies = []
        ndcg_scores = []
        recall_scores = []
        mrr_scores = []
        total_skipped = 0
        total_cost = 0.0 
        
        for q_id, q in queries:
            start_t = time.perf_counter()
            
            if config.use_bm25 and not config.use_dense:
                hits = await bm25_retriever.retrieve(q, top_k=config.top_k)
            else:
                hits = []
                
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)
            
            result_ids = [hit.product_id for hit in hits]
            q_qrels = qrels_all.get(q_id, {})
            
            ndcg, skipped = ndcg_at_k(q_qrels, result_ids, k=10)
            if skipped:
                total_skipped += 1
                continue
                
            recall, _ = recall_at_k(q_qrels, result_ids, k=50)
            mrr, _ = mrr_at_k(q_qrels, result_ids, k=10)
            
            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
            
        await bm25_retriever.close()
            
        avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        avg_mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        
        run_id = str(uuid.uuid4())
        
        res = EvalResult(
            run_id=run_id,
            config=config,
            dataset_version=dataset_version,
            split=split,
            ndcg_10=avg_ndcg,
            recall_50=avg_recall,
            mrr_10=avg_mrr,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            cost=total_cost,
            skipped_count=total_skipped
        )
        
        out_dir = Path("evals/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{run_id}.json", "w") as f:
            f.write(res.model_dump_json(indent=2))
            
        if gatemark_pool:
            await gatemark_pool.execute("""
                INSERT INTO eval_runs 
                (run_id, config, dataset_version, split, ndcg_10, recall_50, mrr_10, latency_p50_ms, latency_p95_ms, cost, skipped_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 
            res.run_id, 
            res.config.model_dump_json(), 
            res.dataset_version, 
            res.split, 
            res.ndcg_10, 
            res.recall_50, 
            res.mrr_10, 
            res.latency_p50_ms, 
            res.latency_p95_ms, 
            res.cost, 
            res.skipped_count)
            
        return res
        
    finally:
        await shoprank_pool.close()
        if gatemark_pool:
            await gatemark_pool.close()
