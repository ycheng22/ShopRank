import asyncpg
from retrieval_core.models import Query, ScoredHit


class BM25Retriever:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.database_url)
        return self._pool

    async def retrieve(self, query: Query, top_k: int) -> list[ScoredHit]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            sql = """
            SELECT 
                product_id, 
                ts_rank(tsv, websearch_to_tsquery('english', replace($1, ' ', ' OR '))) as raw_score 
            FROM products 
            WHERE tsv @@ websearch_to_tsquery('english', replace($1, ' ', ' OR '))
            ORDER BY raw_score DESC 
            LIMIT $2
            """
            rows = await conn.fetch(sql, query.text, top_k)
            
            results = []
            for rank_0_indexed, row in enumerate(rows):
                results.append(
                    ScoredHit(
                        product_id=row['product_id'],
                        raw_score=float(row['raw_score']),
                        retriever_name="bm25",
                        rank=rank_0_indexed + 1
                    )
                )
            return results
            
    async def close(self):
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
