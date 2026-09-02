import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import asyncpg
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel
from retrieval_core.models import Query

from app.settings import Settings, get_settings
from core.pipeline import ShopRankPipelineConfig, search

router = APIRouter()
logger = logging.getLogger(__name__)

from app.limiter import limiter

# Simple in-memory token quota tracking
_daily_tokens = 0
_last_reset_day = -1


def _check_and_update_quota(tokens: int, limit: int) -> bool:
    global _daily_tokens, _last_reset_day
    current_day = datetime.now(UTC).timetuple().tm_yday
    if current_day != _last_reset_day:
        _daily_tokens = 0
        _last_reset_day = current_day

    if _daily_tokens + tokens > limit:
        return False
    _daily_tokens += tokens
    return True


def _clean_text(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def _extract_title(text: str) -> str:
    clean = _clean_text(text)
    if not clean:
        return ""
    if len(clean) <= 120:
        return clean
    truncated = clean[:120].rsplit(" ", 1)[0]
    return truncated + "..."


def _extract_description(text: str) -> str:
    clean = _clean_text(text)
    if not clean:
        return ""
    if len(clean) <= 120:
        return ""
    return clean[len(clean[:120].rsplit(" ", 1)[0]) :].strip()


async def _enrich_hits_with_product_info(
    conn: asyncpg.Connection, hits: list[dict[str, Any]]
) -> None:
    if not hits:
        return
    pids = [h["product_id"] for h in hits if "product_id" in h]
    if not pids:
        return
    rows = await conn.fetch(
        "SELECT product_id, product_text FROM products WHERE product_id = ANY($1)",
        pids,
    )
    text_map = {r["product_id"]: r["product_text"] for r in rows}
    for h in hits:
        pid = h.get("product_id")
        ptext = text_map.get(pid, "")
        h["product_text"] = ptext
        h["title"] = _extract_title(ptext)
        h["description"] = _extract_description(ptext)


class SearchRequest(BaseModel):
    query: str
    locale: str = "en"
    config: ShopRankPipelineConfig


def get_config_hash(config: ShopRankPipelineConfig) -> str:
    d: dict[str, Any] = {
        "use_dense": config.use_dense,
        "use_rerank": config.use_rerank,
        "fusion_method": config.fusion_method,
    }
    if config.fusion_method == "weighted":
        d["fusion_weights"] = config.fusion_weights

    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@router.get("/search")
async def search_preset(
    q: str,
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    locale: str = "en",
    use_dense: bool = True,
    use_rerank: bool = False,
    fusion_method: Literal["rrf", "weighted", "none"] = "rrf",
) -> dict[str, Any]:
    """GET endpoint for preset queries. Served from cache."""
    response.headers["Cache-Control"] = "public, max-age=300, s-maxage=86400"

    config = ShopRankPipelineConfig(
        use_bm25=True,
        use_dense=use_dense,
        use_rerank=use_rerank,
        fusion_method=fusion_method,
        embed_dim=768,
        ef_search=40,
        locale=locale,
        top_k=100,
    )
    chash = get_config_hash(config)

    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                "SELECT response_json FROM demo_cache WHERE query_text = $1 AND locale = $2 AND config_hash = $3",
                q,
                locale,
                chash,
            )
            if row:
                data: dict[str, Any] = json.loads(row["response_json"])
                hits = data.get("hits", [])
                if hits and not hits[0].get("title"):
                    await _enrich_hits_with_product_info(conn, hits)
                    # Opportunistically update cache so future reads have details cached
                    try:
                        await conn.execute(
                            "UPDATE demo_cache SET response_json = $1 WHERE query_text = $2 AND locale = $3 AND config_hash = $4",
                            json.dumps(data),
                            q,
                            locale,
                            chash,
                        )
                    except (asyncpg.PostgresError, OSError):
                        pass
                return data
            else:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Preset not found in cache. Use POST for free-form queries.",
                        "is_preset_error": True,
                    },
                )
        finally:
            await conn.close()
    except HTTPException:
        raise
    except (asyncpg.PostgresError, OSError) as e:
        logger.error(f"DB Error in GET /search: {e}")
        raise HTTPException(status_code=500, detail="Database error")


@router.post("/search")
@limiter.limit("5/minute")
async def search_freeform(
    request: Request,
    req: SearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """POST endpoint for free-form queries. Rate limited to 5/minute per IP."""

    cost = len(req.query) + 100
    if req.config.use_rerank:
        cost += 5000

    quota_ok = _check_and_update_quota(cost, settings.daily_token_quota)
    if not quota_ok:
        req.config.use_rerank = False

    try:
        res = await search(Query(text=req.query), req.config, db_url=settings.database_url)
        degraded_to_bm25 = False
    except Exception as e:
        if isinstance(e, asyncpg.PostgresError) or isinstance(e, HTTPException):
            raise
        logger.warning(f"Inference failed (missing torch/model), degrading to BM25: {e}")
        req.config.use_dense = False
        req.config.use_rerank = False
        res = await search(Query(text=req.query), req.config, db_url=settings.database_url)
        degraded_to_bm25 = True

    out: dict[str, Any] = res.model_dump()
    if not quota_ok:
        out["quota_exhausted"] = True
    if degraded_to_bm25:
        out["degraded_to_bm25"] = True

    hits = out.get("hits", [])
    if hits:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await _enrich_hits_with_product_info(conn, hits)
        finally:
            await conn.close()

    return out
