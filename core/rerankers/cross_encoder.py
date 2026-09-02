"""Cross-encoder reranker using bge-reranker-v2-m3 on CPU.

Loads the model once at process start (singleton). Every incoming ScoredHit
is preserved with all its original breakdown fields; this module only *adds*
rerank_score and rank_before_rerank — it never overwrites existing fields.
"""

from __future__ import annotations

import logging
import time

from retrieval_core.models import Query, ScoredHit

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once on first use.
_model = None
_model_name: str = ""


def _get_default_cache_dir() -> str | None:
    import os
    import sys

    if sys.platform == "win32" and os.path.exists("D:\\"):
        return "D:/huggingface_cache/hub"
    return None


def _load_model(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    cache_dir: str | None = None,
) -> None:
    """Load the cross-encoder model once, globally."""
    global _model, _model_name

    if _model is not None:
        return

    if cache_dir is None:
        cache_dir = _get_default_cache_dir()

    logger.info("Loading cross-encoder model %s on CPU …", model_name)
    start = time.perf_counter()

    from sentence_transformers import CrossEncoder

    _model = CrossEncoder(model_name, device="cpu", cache_folder=cache_dir)
    _model_name = model_name

    elapsed = time.perf_counter() - start
    logger.info("Cross-encoder loaded in %.2f s", elapsed)


def rerank(
    query: Query,
    hits: list[ScoredHit],
    top_k: int,
    product_texts: dict[str, str],
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    cache_dir: str | None = None,
) -> list[ScoredHit]:
    """Rerank hits using the cross-encoder.

    Args:
        query: The search query.
        hits: Fused (or single-retriever) hits to rerank.
        top_k: Number of results to return after reranking.
        product_texts: Mapping of product_id → product text used for scoring.
        model_name: HuggingFace model identifier.
        cache_dir: Directory for cached model weights.

    Returns:
        Top-k hits sorted by rerank_score, with all prior breakdown fields
        preserved and ``rerank_score`` / ``rank_before_rerank`` added.
    """
    _load_model(model_name=model_name, cache_dir=cache_dir)
    assert _model is not None  # for type-checker

    if not hits:
        return []

    # Build (query, passage) pairs — skip hits without product text
    pairs: list[tuple[str, str]] = []
    valid_hits: list[ScoredHit] = []
    for hit in hits:
        text = product_texts.get(hit.product_id, "")
        if text:
            pairs.append((query.text, text))
            valid_hits.append(hit)
        else:
            logger.warning(
                "No product text for %s — excluding from rerank", hit.product_id
            )

    if not pairs:
        return hits[:top_k]

    scores = _model.predict(pairs)

    # Attach scores and record pre-rerank rank
    reranked: list[ScoredHit] = []
    for hit, score in zip(valid_hits, scores):
        # Deep copy to avoid mutating the shared fusion output
        new_hit = hit.model_copy(deep=True)
        new_hit.breakdown.rank_before_rerank = hit.rank
        new_hit.breakdown.rerank_score = float(score)
        reranked.append(new_hit)

    # Sort descending by rerank_score
    reranked.sort(key=lambda h: h.breakdown.rerank_score, reverse=True)

    # Re-assign ranks
    for i, hit in enumerate(reranked):
        hit.rank = i + 1

    return reranked[:top_k]
