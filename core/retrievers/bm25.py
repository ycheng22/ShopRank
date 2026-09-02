"""BM25 retriever using PostgreSQL full-text search.

Supports locale-aware text search configuration:
- 'en' → 'english' (stemming, stop words)
- 'zh', 'fr', other → 'simple' (no stemming — intentional degradation for
  non-English locales to demonstrate BM25's weakness on CJK text)
"""

from __future__ import annotations

import asyncpg
from retrieval_core.models import Query, ScoredHit

# Locale to Postgres tsconfig mapping.
# Using 'simple' for zh/fr is intentional — it shows BM25 degradation on
# non-English text without adding a tokenizer dependency (pg_jieba etc.)
# that would mask the phenomenon we want to demonstrate.
_TSCONFIG_MAP: dict[str, str] = {
    "en": "english",
}


class BM25Retriever:
    def __init__(self, database_url: str, locale: str = "en"):
        self.database_url = database_url
        self.locale = locale
        self.tsconfig = _TSCONFIG_MAP.get(locale, "simple")
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.database_url)
        return self._pool

    async def retrieve(self, query: Query, top_k: int) -> list[ScoredHit]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            sql = """
            SELECT
                product_id,
                ts_rank(tsv, websearch_to_tsquery($3, replace($1, ' ', ' OR '))) as raw_score
            FROM products
            WHERE tsv @@ websearch_to_tsquery($3, replace($1, ' ', ' OR '))
            ORDER BY raw_score DESC
            LIMIT $2
            """
            rows = await conn.fetch(sql, query.text, top_k, self.tsconfig)

            results = []
            for rank_0_indexed, row in enumerate(rows):
                results.append(
                    ScoredHit(
                        product_id=row["product_id"],
                        raw_score=float(row["raw_score"]),
                        retriever_name="bm25",
                        rank=rank_0_indexed + 1,
                    )
                )
            return results

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
