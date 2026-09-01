"""Pipeline orchestrator — assembles retrieval stages per PipelineConfig.

This module is the single entry point for running the search pipeline.
It does NOT read the environment; the database URL and all configuration
are passed in explicitly.
"""

from __future__ import annotations

import asyncio

import asyncpg
from pydantic import Field as PydanticField
from retrieval_core.models import PipelineConfig, Query, ScoredHit, SearchResponse

from core.fusion import fuse
from core.rerankers.cross_encoder import rerank
from core.retrievers.bm25 import BM25Retriever
from core.retrievers.dense import DenseRetriever


class ShopRankPipelineConfig(PipelineConfig):
    """Extended pipeline config with ShopRank-specific knobs."""

    ef_search: int = 40
    rerank_depth: int = 50
    rrf_k: int = 60
    fusion_weights: dict[str, float] = PydanticField(
        default_factory=lambda: {"bm25": 0.5, "dense": 0.5}
    )
    locale: str = "en"


_retrievers: dict[str, BM25Retriever | DenseRetriever] = {}


def _get_bm25(db_url: str, locale: str = "en") -> BM25Retriever:
    key = f"bm25_{locale}"
    if key not in _retrievers:
        _retrievers[key] = BM25Retriever(db_url, locale=locale)
    return _retrievers[key]  # type: ignore[return-value]


def _get_dense(db_url: str, embed_dim: int, ef_search: int) -> DenseRetriever:
    key = f"dense_{embed_dim}_{ef_search}"
    if key not in _retrievers:
        _retrievers[key] = DenseRetriever(db_url, embed_dim, ef_search)
    return _retrievers[key]  # type: ignore[return-value]


async def _fetch_product_texts(
    db_url: str, product_ids: list[str]
) -> dict[str, str]:
    """Fetch product texts for reranking from the database."""
    if not product_ids:
        return {}

    pool = await asyncpg.create_pool(db_url)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT product_id, product_text FROM products WHERE product_id = ANY($1)",
                product_ids,
            )
            return {r["product_id"]: r["product_text"] for r in rows}
    finally:
        await pool.close()


async def search(
    query: Query,
    config: ShopRankPipelineConfig,
    *,
    db_url: str,
) -> SearchResponse:
    """Run the full search pipeline.

    Args:
        query: The user query.
        config: Pipeline configuration controlling which stages run.
        db_url: Database connection URL (injected, not read from env).

    Returns:
        SearchResponse with hits and score breakdowns.
    """
    tasks: dict[str, asyncio.Task[list[ScoredHit]]] = {}

    if config.use_bm25:
        bm25 = _get_bm25(db_url, locale=config.locale)
        tasks["bm25"] = asyncio.create_task(
            bm25.retrieve(query, top_k=config.top_k)
        )

    if config.use_dense:
        dense = _get_dense(db_url, config.embed_dim, config.ef_search)
        tasks["dense"] = asyncio.create_task(
            dense.retrieve(query, top_k=config.top_k)
        )

    runs: dict[str, list[ScoredHit]] = {}
    if tasks:
        results = await asyncio.gather(*tasks.values())
        for name, result in zip(tasks.keys(), results):
            runs[name] = result

    # Fusion / pass-through
    if config.use_bm25 and config.use_dense:
        fusion_params: dict[str, object] = {}
        if config.fusion_method == "rrf":
            fusion_params["k"] = config.rrf_k
        elif config.fusion_method == "weighted":
            fusion_params["weights"] = config.fusion_weights
        fused_hits = fuse(runs, method=config.fusion_method, params=fusion_params)
    elif config.use_bm25:
        fused_hits = runs.get("bm25", [])
        for hit in fused_hits:
            hit.breakdown.bm25_score = hit.raw_score
    elif config.use_dense:
        fused_hits = runs.get("dense", [])
        for hit in fused_hits:
            hit.breakdown.dense_score = hit.raw_score
    else:
        fused_hits = []

    # Reranking
    if config.use_rerank and fused_hits:
        candidates = fused_hits[: config.rerank_depth]
        product_ids = [h.product_id for h in candidates]
        product_texts = await _fetch_product_texts(db_url, product_ids)
        reranked = rerank(
            query, candidates, top_k=config.top_k, product_texts=product_texts
        )
        fused_hits = reranked

    return SearchResponse(
        hits=fused_hits[: config.top_k],
        total_found=len(fused_hits),
    )


async def close_pipeline() -> None:
    """Close all cached retriever connection pools."""
    for ret in _retrievers.values():
        if hasattr(ret, "close"):
            await ret.close()
    _retrievers.clear()
