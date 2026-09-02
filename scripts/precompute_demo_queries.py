import argparse
import asyncio
import json
import logging

import asyncpg
from retrieval_core.models import Query

from app.routes.examples import DEMO_QUERIES
from app.routes.search import get_config_hash
from app.settings import get_settings
from core.pipeline import ShopRankPipelineConfig, close_pipeline, search

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def precompute(args, settings):
    db_url = settings.database_url

    # We want to precompute across the configs exposed by the UI:
    # use_dense: True/False
    # fusion_method: none, rrf, weighted
    # use_rerank: True/False

    configs = []

    for use_dense in [True, False]:
        for fusion in ["none", "rrf", "weighted"]:
            for use_rerank in [True, False]:
                if not use_dense and fusion != "none":
                    continue  # BM25 only doesn't need fusion
                if use_dense and fusion == "none":
                    continue  # For this demo, dense is always fused with BM25

                c = ShopRankPipelineConfig(
                    use_bm25=True,
                    use_dense=use_dense,
                    use_rerank=use_rerank,
                    fusion_method=fusion,
                    embed_dim=768,
                    ef_search=40,
                    top_k=100,
                )
                configs.append(c)

    # Remove duplicates
    unique_configs = {}
    for c in configs:
        h = get_config_hash(c)
        if h not in unique_configs:
            unique_configs[h] = c

    logger.info(f"Will precompute {len(unique_configs)} configs per query.")

    conn = await asyncpg.connect(db_url)

    with open("scripts/schema.sql", "r") as f:  # noqa: ASYNC230
        schema_sql = f.read()
    await conn.execute(schema_sql)

    records = []

    for dq in DEMO_QUERIES:
        for chash, config in unique_configs.items():
            config.locale = dq.locale

            logger.info(
                f"Precomputing {dq.query} (locale: {dq.locale}, config: {chash})"
            )

            if not args.dry_run:
                try:
                    res = await search(Query(text=dq.query), config, db_url=db_url)
                    json_str = json.dumps(res.model_dump())
                    records.append((dq.query, dq.locale, chash, json_str))
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to precompute {dq.query}: {e}")

    if not args.dry_run and records:
        logger.info(f"Writing {len(records)} records to demo_cache...")
        await conn.executemany(
            """
            INSERT INTO demo_cache (query_text, locale, config_hash, response_json)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (query_text, locale, config_hash) 
            DO UPDATE SET response_json = EXCLUDED.response_json, created_at = NOW();
        """,
            records,
        )

    await conn.close()
    await close_pipeline()
    logger.info("Done.")


if __name__ == "__main__":
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(precompute(args, settings))
