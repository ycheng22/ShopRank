import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
from pathlib import Path
import asyncpg
from retrieval_core.models import PipelineConfig
from app.settings import get_settings

CONFIGS = [
    (
        "BM25 Baseline", 
        PipelineConfig(
            use_bm25=True,
            use_dense=False,
            use_rerank=False,
            top_k=100,
            fusion_method="rrf",
            embed_dim=768
        )
    )
]

async def render_ablation_table():
    settings = get_settings()
    pool = None
    if settings.gatemark_database_url:
        pool = await asyncpg.create_pool(settings.gatemark_database_url)
    
    lines = []
    lines.append("| Configuration | NDCG@10 | Recall@50 | MRR@10 | p95 Latency (ms) |")
    lines.append("| --- | --- | --- | --- | --- |")
    
    for name, config in CONFIGS:
        row = None
        if pool:
            sql = "SELECT * FROM eval_runs ORDER BY created_at DESC"
            rows = await pool.fetch(sql)
            for r in rows:
                c = PipelineConfig.model_validate_json(r["config"])
                if c == config:
                    row = r
                    break
        
        # Fallback to local json files if not found in DB
        if not row:
            results_dir = Path("evals/results")
            if results_dir.exists():
                latest_mtime = 0
                for f in results_dir.glob("*.json"):
                    data = json.loads(f.read_text())
                    c = PipelineConfig.model_validate(data["config"])
                    if c == config:
                        mtime = f.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            row = data

        if row:
            lines.append(f"| {name} | {row['ndcg_10']:.4f} | {row['recall_50']:.4f} | {row['mrr_10']:.4f} | {row['latency_p95_ms']:.1f} |")
        else:
            lines.append(f"| {name} | TBD | TBD | TBD | TBD |")

    if pool:
        await pool.close()

    table_md = "\n".join(lines)
    print(table_md)

    # Update README.md
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        import re
        new_content = re.sub(
            r"<!-- ABLATION_START -->.*?<!-- ABLATION_END -->",
            f"<!-- ABLATION_START -->\n{table_md}\n<!-- ABLATION_END -->",
            content,
            flags=re.DOTALL
        )
        readme_path.write_text(new_content, encoding="utf-8")
        print("Updated README.md")

if __name__ == "__main__":
    asyncio.run(render_ablation_table())
