import asyncio
import logging
import os
import subprocess
import time

import asyncpg
import numpy as np
import pandas as pd

from app.settings import get_settings
from core.embeddings import get_cache_key, get_provider
from core.pipeline import ShopRankPipelineConfig
from evals.runner import run

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _truncate_and_normalize(vector: np.ndarray, dim: int) -> np.ndarray:
    vec = vector.astype(np.float32)
    if len(vec) > dim:
        vec = vec[:dim]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


async def get_peak_gpu_memory():
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        mem = max(
            [
                int(x.strip())
                for x in result.stdout.strip().split("\n")
                if x.strip().isdigit()
            ]
        )
        return f"{mem} MiB"
    except Exception:  # noqa: BLE001
        return "N/A (No local GPU)"


def truncate_text(text, max_len):
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) > max_len:
        return " ".join(words[:max_len])
    return text


async def prepare_1024_cache(df, batch_size=32, max_len=512):
    cache_dir = ".cache/embeddings"
    os.makedirs(cache_dir, exist_ok=True)
    provider = get_provider()

    total_rows = len(df)
    logger.info(f"Checking 1024-dim cache for {total_rows} products...")

    missing_texts = []
    missing_hashes = []

    for i, row in df.iterrows():
        text = truncate_text(row["product_text"], max_len)
        h = get_cache_key(text, 1024).split("_")[-1]
        cache_path = os.path.join(cache_dir, f"bge-m3_1024_{h}.npy")
        if not os.path.exists(cache_path):
            missing_texts.append(text)
            missing_hashes.append(h)

    if not missing_texts:
        logger.info("All 1024-dim embeddings are cached.")
        return

    logger.info(
        f"Need to compute {len(missing_texts)} embeddings at 1024-dim. This may take a while..."
    )

    start_time = time.time()
    processed = 0

    for i in range(0, len(missing_texts), batch_size):
        batch_texts = missing_texts[i : i + batch_size]
        batch_hashes = missing_hashes[i : i + batch_size]

        raw_embeddings = await provider.embed(batch_texts, dim=1024)
        for h, raw_emb in zip(batch_hashes, raw_embeddings):
            emb = np.array(raw_emb, dtype=np.float32)
            # Normalize to 1024
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            cache_path = os.path.join(cache_dir, f"bge-m3_1024_{h}.npy")
            np.save(cache_path, emb)

        processed += len(batch_texts)
        if processed % 1000 < batch_size or processed == len(missing_texts):
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (len(missing_texts) - processed) / rate if rate > 0 else 0
            logger.info(f"Computed {processed}/{len(missing_texts)}. ETA: {eta:.2f}s")


async def run_ablation(settings):
    db_url = settings.database_url
    if not db_url:
        logger.error("DATABASE_URL is missing in Settings")
        return

    parquet_path = "data/products.parquet"
    if not os.path.exists(parquet_path) and os.path.exists(
        "data/run2/products.parquet"
    ):
        parquet_path = "data/run2/products.parquet"

    df = pd.read_parquet(parquet_path)

    await prepare_1024_cache(df)

    # Run 512, then 1024, then 768 to restore production state at the end
    dims = [512, 1024, 768]
    results = {}

    # Connect AFTER the 45 minute cache preparation to avoid idle timeout
    conn = await asyncpg.connect(db_url)

    for dim in dims:
        logger.info(f"\n{'=' * 40}\nRunning ablation for dim={dim}\n{'=' * 40}")

        logger.info(f"Loading and truncating to {dim}-dim...")
        
        records = []
        for i, row in df.iterrows():
            pid = row["product_id"]
            text = truncate_text(row["product_text"], 512)
            h = get_cache_key(text, 1024).split("_")[-1]
            cache_path = os.path.join(".cache/embeddings", f"bge-m3_1024_{h}.npy")

            vec_1024 = np.load(cache_path)
            vec_target = _truncate_and_normalize(vec_1024, dim)
            records.append((pid, str(vec_target.tolist())))

        logger.info(f"Updating database table (embedding vector({dim}))...")
        await conn.execute("DROP INDEX IF EXISTS products_embedding_idx;")
        await conn.execute("ALTER TABLE products DROP COLUMN IF EXISTS embedding;")
        await conn.execute(f"ALTER TABLE products ADD COLUMN embedding vector({dim});")

        # Batch insert/update using executemany with a prepared statement
        logger.info("Writing to DB...")
        await conn.executemany(
            """
            UPDATE products SET embedding = $2::vector WHERE product_id = $1;
        """,
            records,
        )

        logger.info(f"Building HNSW index for dim={dim}...")
        idx_start = time.time()
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS products_embedding_idx 
            ON products USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        idx_time = time.time() - idx_start
        logger.info(f"Index built in {idx_time:.2f}s")

        size_res = await conn.fetchval(
            "SELECT pg_size_pretty(pg_total_relation_size('products'));"
        )
        logger.info(f"Table size: {size_res}")

        logger.info("Running evaluation...")
        config = ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=False,
            top_k=100,
            fusion_method="rrf",
            embed_dim=dim,
            ef_search=40,
        )

        dataset_version = "7916cdf6ab75a462e77f20ab40428a10923998d5"
        eval_result = await run(
            config,
            split="dev",
            dataset_version=dataset_version,
            config_label=f"dim_{dim}",
            locale="en",
        )

        logger.info(
            f"NDCG@10={eval_result.ndcg_10:.4f}, Recall@50={eval_result.recall_50:.4f}"
        )

        results[dim] = {
            "ndcg_10": eval_result.ndcg_10,
            "recall_50": eval_result.recall_50,
            "table_size": size_res,
            "idx_time": idx_time,
        }

    await conn.close()

    print("\n\n=== Dimension Ablation Results ===")
    print("| Dimension | NDCG@10 | Recall@50 | Table Size | Index Build Time |")
    print("| --- | --- | --- | --- | --- |")
    for dim in [1024, 768, 512]:
        res = results[dim]
        print(
            f"| {dim} | {res['ndcg_10']:.4f} | {res['recall_50']:.4f} | {res['table_size']} | {res['idx_time']:.0f}s |"
        )


if __name__ == "__main__":
    settings = get_settings()
    asyncio.run(run_ablation(settings))
