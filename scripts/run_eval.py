import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from evals.runner import run
from retrieval_core.models import PipelineConfig
from evals.ablation import render_ablation_table

async def main():
    config = PipelineConfig(
        use_bm25=True,
        use_dense=False,
        use_rerank=False,
        top_k=100,
        fusion_method="rrf",
        embed_dim=768
    )
    
    print("Running evaluation on dev split...")
    res = await run(config=config, split="dev", dataset_version="7916cdf6ab75a462e77f20ab40428a10923998d5", limit=None)
    
    print(f"Skipped queries: {res.skipped_count}")
    print(f"NDCG@10: {res.ndcg_10:.4f}")
    
    print("\nAblation Table:")
    await render_ablation_table()

if __name__ == "__main__":
    asyncio.run(main())
