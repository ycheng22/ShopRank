"""Mine hard negatives from the current pipeline.

For every dev query, runs the pipeline and collects candidates labelled S/C/I
that rank above configurable thresholds. These are cases where the pipeline
is wrong — exactly the material for error analysis.

Usage:
    python scripts/mine_hard_negatives.py --fusion-method rrf
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import asyncpg
import pandas as pd
from retrieval_core.models import Query

from app.settings import get_settings
from core.pipeline import ShopRankPipelineConfig, close_pipeline, search

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


async def mine(args: argparse.Namespace) -> None:
    settings = get_settings()
    pool = await asyncpg.create_pool(settings.database_url)

    try:
        # Load dev queries
        rows = await pool.fetch(
            "SELECT query_id, text, locale FROM queries WHERE split = 'dev' AND locale = 'us' ORDER BY query_id"
        )
        logger.info("Loaded %d dev queries", len(rows))

        # Load qrels
        qrel_rows = await pool.fetch("SELECT query_id, product_id, esci_label FROM qrels")
        qrels: dict[str, dict[str, str]] = defaultdict(dict)
        for r in qrel_rows:
            qrels[r["query_id"]][r["product_id"]] = r["esci_label"]

        # Load product titles for display
        prod_rows = await pool.fetch("SELECT product_id, product_text FROM products")
        product_texts = {r["product_id"]: r["product_text"][:120] for r in prod_rows}

        config = ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=args.use_rerank,
            top_k=100,
            fusion_method=args.fusion_method,
            embed_dim=768,
            ef_search=40,
            rerank_depth=args.rerank_depth,
            locale="en",
        )

        hard_negatives: list[dict[str, object]] = []
        label_counts: dict[str, int] = defaultdict(int)
        label_ranks: dict[str, list[int]] = defaultdict(list)

        for i, row in enumerate(rows):
            q_id = row["query_id"]
            q_text = row["text"]

            response = await search(
                Query(text=q_text),
                config,
                db_url=settings.database_url,
            )

            q_qrels = qrels.get(q_id, {})

            for hit in response.hits:
                label = q_qrels.get(hit.product_id)
                if label is None:
                    continue  # Unjudged — not useful for hard neg mining

                # Check thresholds
                is_hard_neg = False
                if label in ("I", "Irrelevant") and hit.rank <= args.threshold_i:
                    is_hard_neg = True
                elif label in ("C", "Complement") and hit.rank <= args.threshold_c:
                    is_hard_neg = True
                elif label in ("S", "Substitute") and hit.rank <= args.threshold_s:
                    is_hard_neg = True

                if is_hard_neg:
                    breakdown_dict = hit.breakdown.model_dump()
                    hard_negatives.append({
                        "query_id": q_id,
                        "query_text": q_text,
                        "product_id": hit.product_id,
                        "product_title": product_texts.get(hit.product_id, ""),
                        "label": label,
                        "rank": hit.rank,
                        "score_breakdown": json.dumps(breakdown_dict),
                    })
                    label_counts[label] += 1
                    label_ranks[label].append(hit.rank)

            if (i + 1) % 50 == 0:
                logger.info("Processed %d/%d queries, found %d hard negatives so far",
                            i + 1, len(rows), len(hard_negatives))

        await close_pipeline()

    finally:
        await pool.close()

    # Report
    logger.info("=" * 60)
    logger.info("Hard Negative Mining Results")
    logger.info("=" * 60)
    logger.info("Total hard negatives found: %d", len(hard_negatives))
    for label, count in sorted(label_counts.items()):
        ranks = label_ranks[label]
        mean_rank = sum(ranks) / len(ranks) if ranks else 0
        logger.info("  %s: %d (mean rank %.1f)", label, count, mean_rank)

    if not hard_negatives:
        logger.info("No hard negatives found with current thresholds.")
        return

    # Save parquet
    df = pd.DataFrame(hard_negatives)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved to %s", out_path)

    # Print worst 20
    df_sorted = df.sort_values("rank")
    worst = df_sorted.head(20)

    logger.info("")
    logger.info("Worst 20 hard negatives (lowest rank = most egregious):")
    logger.info("-" * 80)
    for _, row in worst.iterrows():
        logger.info(
            "  rank=%d label=%s query='%s' product='%s'",
            row["rank"],
            row["label"],
            str(row["query_text"])[:60],
            str(row["product_title"])[:60],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives from the current pipeline")
    parser.add_argument("--threshold-i", type=int, default=10, help="Max rank for Irrelevant to be considered a hard negative")
    parser.add_argument("--threshold-c", type=int, default=5, help="Max rank for Complement to be considered a hard negative")
    parser.add_argument("--threshold-s", type=int, default=1, help="Max rank for Substitute to be considered a hard negative")
    parser.add_argument("--fusion-method", default="rrf", help="Fusion method to use")
    parser.add_argument("--use-rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--rerank-depth", type=int, default=50, help="Rerank depth")
    parser.add_argument("--output", default="data/hard_negatives.parquet", help="Output parquet path")
    args = parser.parse_args()

    asyncio.run(mine(args))


if __name__ == "__main__":
    main()
