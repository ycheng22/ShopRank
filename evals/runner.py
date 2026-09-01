"""Evaluation runner — runs the pipeline on a split and computes metrics.

Results are written to both local JSON files and the gatemark eval_runs table.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from pathlib import Path

import asyncpg
import numpy as np
from pydantic import BaseModel
from retrieval_core.models import Query

from app.settings import get_settings
from core.pipeline import ShopRankPipelineConfig, close_pipeline, search
from evals.metrics import mrr_at_k, ndcg_at_k, recall_at_k


class EvalResult(BaseModel):
    run_id: str
    config: ShopRankPipelineConfig
    config_label: str
    dataset_version: str
    split: str
    locale: str
    ndcg_10: float
    recall_50: float
    mrr_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost: float
    skipped_count: int


async def load_split_queries(
    pool: asyncpg.Pool,
    split: str,
    limit: int | None = None,
    locale: str | None = None,
) -> list[tuple[str, Query]]:
    """Load queries for a split, optionally filtered by locale."""
    if locale:
        sql = "SELECT query_id, text, locale FROM queries WHERE split = $1 AND locale = $2 ORDER BY query_id"
        params: list[object] = [split, locale]
    else:
        sql = "SELECT query_id, text, locale FROM queries WHERE split = $1 ORDER BY query_id"
        params = [split]

    if limit:
        sql += f" LIMIT {limit}"

    rows = await pool.fetch(sql, *params)
    return [(r["query_id"], Query(text=r["text"])) for r in rows]


async def load_qrels(pool: asyncpg.Pool) -> dict[str, dict[str, str]]:
    rows = await pool.fetch("SELECT query_id, product_id, esci_label FROM qrels")
    qrels: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        qrels[r["query_id"]][r["product_id"]] = r["esci_label"]
    return dict(qrels)


async def init_gatemark_db(pool: asyncpg.Pool) -> None:
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            config JSONB,
            config_label TEXT,
            dataset_version TEXT,
            split TEXT,
            locale TEXT DEFAULT 'en',
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


def _strip_locale_suffix(query_id: str) -> str:
    """Strip locale suffix from translated query IDs.

    Translated queries have IDs like 'Q123_zh' or 'Q123_fr'.
    The original qrels use 'Q123'. This function returns the base ID.
    """
    for suffix in ("_zh", "_fr", "_de", "_ja", "_es"):
        if query_id.endswith(suffix):
            return query_id[: -len(suffix)]
    return query_id


async def run(
    config: ShopRankPipelineConfig,
    split: str = "dev",
    dataset_version: str = "unknown",
    allow_test: bool = False,
    limit: int | None = None,
    config_label: str = "",
    locale: str = "en",
) -> EvalResult:
    if split == "test" and not allow_test:
        raise ValueError("Reading the 'test' split requires allow_test=True")

    settings = get_settings()
    db_url = settings.database_url
    print("Connecting to Shoprank pool...")
    shoprank_pool = await asyncpg.create_pool(db_url)

    gatemark_pool = None
    if settings.gatemark_database_url:
        print("Connecting to Gatemark pool...")
        gatemark_pool = await asyncpg.create_pool(settings.gatemark_database_url)
        await init_gatemark_db(gatemark_pool)

    try:
        print(f"Loading queries (locale={locale})...")
        queries = await load_split_queries(
            shoprank_pool, split, limit, locale=locale if locale != "en" else "us"
        )
        print(f"Loaded {len(queries)} queries. Loading qrels...")
        qrels_all = await load_qrels(shoprank_pool)

        print("Running evaluation loop...")

        latencies: list[float] = []
        ndcg_scores: list[float] = []
        recall_scores: list[float] = []
        mrr_scores: list[float] = []
        total_skipped = 0
        total_cost = 0.0

        # Override config locale
        eval_config = config.model_copy(update={"locale": locale})

        from tqdm.asyncio import tqdm
        for q_id, q in tqdm(queries, desc=f"Evaluating {config_label}"):
            start_t = time.perf_counter()

            response = await search(q, eval_config, db_url=db_url)
            hits = response.hits

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)

            result_ids = [hit.product_id for hit in hits]

            # For translated queries, look up qrels using the base query ID
            qrel_key = _strip_locale_suffix(q_id)
            q_qrels = qrels_all.get(qrel_key, {})

            ndcg, skipped = ndcg_at_k(q_qrels, result_ids, k=10)
            if skipped:
                total_skipped += 1
                continue

            recall, _ = recall_at_k(q_qrels, result_ids, k=50)
            mrr, _ = mrr_at_k(q_qrels, result_ids, k=10)

            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
            mrr_scores.append(mrr)

        await close_pipeline()

        avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        avg_mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

        run_id = str(uuid.uuid4())

        res = EvalResult(
            run_id=run_id,
            config=config,
            config_label=config_label,
            dataset_version=dataset_version,
            split=split,
            locale=locale,
            ndcg_10=avg_ndcg,
            recall_50=avg_recall,
            mrr_10=avg_mrr,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            cost=total_cost,
            skipped_count=total_skipped,
        )

        out_dir = Path("evals/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{run_id}.json").write_text(
            res.model_dump_json(indent=2), encoding="utf-8"
        )

        if gatemark_pool:
            await gatemark_pool.execute(
                """
                INSERT INTO eval_runs
                (run_id, config, config_label, dataset_version, split, locale,
                 ndcg_10, recall_50, mrr_10, latency_p50_ms, latency_p95_ms, cost, skipped_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                res.run_id,
                res.config.model_dump_json(),
                res.config_label,
                res.dataset_version,
                res.split,
                res.locale,
                res.ndcg_10,
                res.recall_50,
                res.mrr_10,
                res.latency_p50_ms,
                res.latency_p95_ms,
                res.cost,
                res.skipped_count,
            )

        return res

    finally:
        await shoprank_pool.close()
        if gatemark_pool:
            await gatemark_pool.close()
