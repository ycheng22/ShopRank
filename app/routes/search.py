import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel
from retrieval_core.models import Query

from app.settings import Settings, get_settings
from core.pipeline import ShopRankPipelineConfig, search

router = APIRouter()
logger = logging.getLogger(__name__)

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


class SearchRequest(BaseModel):
    query: str
    locale: str = "en"
    config: ShopRankPipelineConfig


def get_config_hash(config: ShopRankPipelineConfig) -> str:
    d = {
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
    locale: str = "en",
    use_dense: bool = True,
    use_rerank: bool = False,
    fusion_method: str = "rrf",
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> dict:
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
                return json.loads(row["response_json"])
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
async def search_freeform(
    req: SearchRequest,
    x_api_key: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> dict:
    """POST endpoint for free-form queries. Requires API key."""
    if not x_api_key or x_api_key.strip() == "":
        return {
            "error": "demo_mode",
            "message": "Free-form search requires an API key. Try one of the preset examples.",
            "examples_url": "/api/examples",
        }

    cost = len(req.query) + 100
    if req.config.use_rerank:
        cost += 5000

    quota_ok = _check_and_update_quota(cost, settings.daily_token_quota)
    if not quota_ok:
        req.config.use_rerank = False

    res = await search(Query(text=req.query), req.config, db_url=settings.database_url)

    out = res.model_dump()
    if not quota_ok:
        out["quota_exhausted"] = True

    return out
