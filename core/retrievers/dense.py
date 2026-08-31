import asyncpg
from retrieval_core.models import Query, ScoredHit

from core.embeddings import embed_query


class DenseRetriever:
    def __init__(self, database_url: str, embed_dim: int, ef_search: int = 40):
        self.database_url = database_url
        self.embed_dim = embed_dim
        self.ef_search = ef_search
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.database_url)
        return self._pool

    async def retrieve(self, query: Query, top_k: int) -> list[ScoredHit]:
        # Embed the query using the configured dimensionality
        q_emb = await embed_query(query.text, dim=self.embed_dim)
        
        # asyncpg expects the string representation of a list for pgvector type casting if using $1::vector
        q_emb_str = str(q_emb.tolist())
        
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {self.ef_search}")
                
                # Using cosine distance <=> for pgvector
                # Cosine similarity is 1 - distance
                sql = """
                SELECT 
                    product_id, 
                    1 - (embedding <=> $1::vector) AS raw_score 
                FROM products 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector 
                LIMIT $2
                """
                
                rows = await conn.fetch(sql, q_emb_str, top_k)
                
                results = []
                for rank_0_indexed, row in enumerate(rows):
                    results.append(
                        ScoredHit(
                            product_id=row['product_id'],
                            raw_score=float(row['raw_score']),
                            retriever_name="dense",
                            rank=rank_0_indexed + 1
                        )
                    )
                return results

    async def close(self):
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
