import argparse
import asyncio
import time
import logging
import subprocess

import asyncpg
import pandas as pd

from app.settings import get_settings
from core.embeddings import embed_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def get_peak_gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, check=True
        )
        mem = max([int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip().isdigit()])
        return f"{mem} MiB"
    except Exception:
        return "N/A (No local GPU or nvidia-smi failed)"

async def build_index(args, settings):
    logger.info("Connecting to database...")
    # In Neon, you might need pooling, but for script unpooled is fine if available
    db_url = settings.database_url
    if not db_url:
        logger.error("DATABASE_URL is missing in Settings")
        return
        
    conn = await asyncpg.connect(db_url)
    
    logger.info("Loading products from parquet...")
    parquet_path = "data/products.parquet"
    import os
    if not os.path.exists(parquet_path) and os.path.exists("data/run2/products.parquet"):
        parquet_path = "data/run2/products.parquet"
        
    df = pd.read_parquet(parquet_path)
    if args.limit:
        df = df.head(args.limit)
    
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} products.")
    
    batch_size = args.batch_size
    dim = args.dim
    
    start_time = time.time()
    
    # Truncate text to max_len tokens (approximated by words for simplicity)
    def truncate_text(text, max_len):
        if not isinstance(text, str):
            return ""
        words = text.split()
        if len(words) > max_len:
            return " ".join(words[:max_len])
        return text
        
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        
        texts = batch["product_text"].apply(lambda t: truncate_text(t, args.max_len)).tolist()
        product_ids = batch["product_id"].tolist()
        
        embeddings = await embed_documents(texts, dim=dim)
        
        records = [(pid, emb.tolist()) for pid, emb in zip(product_ids, embeddings)]
        await conn.executemany("""
            INSERT INTO products (product_id, embedding)
            VALUES ($1, $2::vector)
            ON CONFLICT (product_id) DO UPDATE SET embedding = EXCLUDED.embedding;
        """, records)
        
        processed = min(i + batch_size, total_rows)
        if processed % 1000 < batch_size or processed == total_rows:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (total_rows - processed) / rate if rate > 0 else 0
            logger.info(f"Processed {processed}/{total_rows}. ETA: {eta:.2f}s")
    
    if not args.skip_index:
        logger.info("Building HNSW index...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS products_embedding_idx 
            ON products USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        logger.info("HNSW index built.")
    else:
        logger.info("Skipping HNSW index creation.")
        
    end_time = time.time()
    logger.info(f"Final build time: {end_time - start_time:.2f}s")
    
    gpu_mem = await get_peak_gpu_memory()
    logger.info(f"Peak GPU memory: {gpu_mem}")
    
    size_res = await conn.fetchval("SELECT pg_size_pretty(pg_total_relation_size('products'));")
    logger.info(f"Resulting table size: {size_res}")
    
    await conn.close()

def main():
    settings = get_settings()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=getattr(settings, "batch_size", 32))
    parser.add_argument("--max-len", type=int, default=getattr(settings, "max_len", 512))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()
    
    asyncio.run(build_index(args, settings))

if __name__ == "__main__":
    main()
