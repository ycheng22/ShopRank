import asyncio

from retrieval_core.models import PipelineConfig, Query, SearchResponse

from app.settings import get_settings
from core.fusion import fuse
from core.retrievers.bm25 import BM25Retriever
from core.retrievers.dense import DenseRetriever


class ShopRankPipelineConfig(PipelineConfig):
    ef_search: int = 40

_retrievers = {}

def _get_bm25(db_url: str) -> BM25Retriever:
    if "bm25" not in _retrievers:
        _retrievers["bm25"] = BM25Retriever(db_url)
    return _retrievers["bm25"]

def _get_dense(db_url: str, embed_dim: int, ef_search: int) -> DenseRetriever:
    key = f"dense_{embed_dim}_{ef_search}"
    if key not in _retrievers:
        _retrievers[key] = DenseRetriever(db_url, embed_dim, ef_search)
    return _retrievers[key]

async def search(query: Query, config: ShopRankPipelineConfig) -> SearchResponse:
    settings = get_settings()
    
    tasks = {}
    if config.use_bm25:
        bm25 = _get_bm25(settings.database_url)
        tasks["bm25"] = asyncio.create_task(bm25.retrieve(query, top_k=config.top_k))
        
    if config.use_dense:
        dense = _get_dense(settings.database_url, config.embed_dim, config.ef_search)
        tasks["dense"] = asyncio.create_task(dense.retrieve(query, top_k=config.top_k))
        
    runs = {}
    if tasks:
        # Await all concurrently
        results = await asyncio.gather(*tasks.values())
        for name, result in zip(tasks.keys(), results):
            runs[name] = result
            
    # Process results
    if config.use_bm25 and config.use_dense:
        # Both enabled, apply fusion
        fused_hits = fuse(runs, method=config.fusion_method)
    elif config.use_bm25:
        fused_hits = runs.get("bm25", [])
        # Ensure ScoreBreakdown is set properly
        for hit in fused_hits:
            hit.breakdown.bm25_score = hit.raw_score
    elif config.use_dense:
        fused_hits = runs.get("dense", [])
        # Ensure ScoreBreakdown is set properly
        for hit in fused_hits:
            hit.breakdown.dense_score = hit.raw_score
    else:
        fused_hits = []

    # Apply reranker later (M4)
    if config.use_rerank:
        pass
        
    return SearchResponse(
        hits=fused_hits[:config.top_k],
        total_found=len(fused_hits)
    )

async def close_pipeline():
    for ret in _retrievers.values():
        if hasattr(ret, "close"):
            await ret.close()
    _retrievers.clear()
