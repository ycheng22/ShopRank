"""Ablation table generator and cross-lingual evaluation renderer.

Generates:
1. Five-row ablation table (BM25 → +dense → +hybrid → +rerank → +weighted)
2. RRF k diagnostic (k=10, 60, 200)
3. Cross-tab: 3 locales × 4 configs of NDCG@10
4. Bad-case dumps for non-English locales

Usage:
    python evals/ablation.py                    # Render ablation table
    python evals/ablation.py --cross-tab        # Render cross-lingual table
    python evals/ablation.py --rrf-diagnostic   # RRF k sweep
    python evals/ablation.py --bad-cases        # Dump worst queries per locale
    python evals/ablation.py --run-all          # Run all eval configs
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


from app.settings import get_settings
from core.pipeline import ShopRankPipelineConfig

# ─── Ablation configs (5 rows) ──────────────────────────────────────────────

CONFIGS: list[tuple[str, ShopRankPipelineConfig]] = [
    (
        "BM25 Baseline",
        ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=False,
            use_rerank=False,
            top_k=100,
            fusion_method="none",
            embed_dim=768,
            ef_search=40,
        ),
    ),
    (
        "+dense",
        ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=False,
            top_k=100,
            fusion_method="none",
            embed_dim=768,
            ef_search=40,
        ),
    ),
    (
        "+hybrid",
        ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=False,
            top_k=100,
            fusion_method="rrf",
            embed_dim=768,
            ef_search=40,
        ),
    ),
    (
        "+rerank",
        ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=True,
            top_k=100,
            fusion_method="rrf",
            embed_dim=768,
            ef_search=40,
            rerank_depth=30,
        ),
    ),
    (
        "+weighted",
        ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=True,
            top_k=100,
            fusion_method="weighted",
            embed_dim=768,
            ef_search=40,
            rerank_depth=30,
            fusion_weights={"bm25": 0.4, "dense": 0.6},
        ),
    ),
]

# ─── Cross-tab configs (4 configs × 3 locales) ─────────────────────────────

CROSS_TAB_CONFIGS: list[tuple[str, ShopRankPipelineConfig]] = [
    (
        "bm25",
        ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=False,
            use_rerank=False,
            top_k=100,
            fusion_method="none",
            embed_dim=768,
            ef_search=40,
        ),
    ),
    (
        "dense",
        ShopRankPipelineConfig(
            use_bm25=False,
            use_dense=True,
            use_rerank=False,
            top_k=100,
            fusion_method="none",
            embed_dim=768,
            ef_search=40,
        ),
    ),
    (
        "hybrid",
        ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=False,
            top_k=100,
            fusion_method="rrf",
            embed_dim=768,
            ef_search=40,
        ),
    ),
    (
        "hybrid+rerank",
        ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=True,
            top_k=100,
            fusion_method="rrf",
            embed_dim=768,
            ef_search=40,
            rerank_depth=30,
        ),
    ),
]

LOCALES = ["en", "zh", "fr"]


# ─── Table rendering ────────────────────────────────────────────────────────


def _find_result_for_config(
    config: ShopRankPipelineConfig,
    results_dir: Path,
    *,
    locale: str = "en",
    config_label: str = "",
) -> dict | None:  # type: ignore[type-arg]
    """Find the most recent eval result matching the given config."""
    if not results_dir.exists():
        return None

    latest_mtime = 0.0
    best: dict | None = None  # type: ignore[type-arg]

    for f in results_dir.glob("*.json"):
        data = json.loads(f.read_text())
        c = ShopRankPipelineConfig.model_validate(data["config"])

        # Match config (ignoring locale since we set it at eval time)
        config_match = c.model_copy(update={"locale": "en"})
        target_match = config.model_copy(update={"locale": "en"})

        result_locale = data.get("locale", "en")

        if config_match == target_match and result_locale == locale:
            mtime = f.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                best = data

    return best


async def render_ablation_table() -> None:
    """Render the 5-row ablation table and update README."""
    results_dir = Path("evals/results")

    lines = [
        "| Configuration | NDCG@10 | Recall@50 | MRR@10 | p95 Latency (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for name, config in CONFIGS:
        row = _find_result_for_config(config, results_dir)

        if row:
            lines.append(
                f"| {name} | {row['ndcg_10']:.4f} | {row['recall_50']:.4f} "
                f"| {row['mrr_10']:.4f} | {row['latency_p95_ms']:.1f} |"
            )
        else:
            lines.append(f"| {name} | TBD | TBD | TBD | TBD |")

    table_md = "\n".join(lines)
    print(table_md)

    # Update README.md
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        new_content = re.sub(
            r"<!-- ABLATION_START -->.*?<!-- ABLATION_END -->",
            f"<!-- ABLATION_START -->\n{table_md}\n<!-- ABLATION_END -->",
            content,
            flags=re.DOTALL,
        )
        readme_path.write_text(new_content, encoding="utf-8")
        print("Updated README.md ablation table")


async def render_cross_tab() -> None:
    """Render the 3-locale × 4-config cross-tab of NDCG@10."""
    results_dir = Path("evals/results")

    # Header
    header = "| Locale |"
    separator = "| --- |"
    for name, _ in CROSS_TAB_CONFIGS:
        header += f" {name} |"
        separator += " --- |"

    lines = [header, separator]

    for locale in LOCALES:
        row_line = f"| {locale} |"
        for _, config in CROSS_TAB_CONFIGS:
            result = _find_result_for_config(config, results_dir, locale=locale)
            if result:
                row_line += f" {result['ndcg_10']:.4f} |"
            else:
                row_line += " TBD |"
        lines.append(row_line)

    table_md = "\n".join(lines)
    print("\nCross-lingual NDCG@10:")
    print(table_md)

    # Update README with cross-tab
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        marker_start = "<!-- CROSSTAB_START -->"
        marker_end = "<!-- CROSSTAB_END -->"
        if marker_start in content:
            new_content = re.sub(
                rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}",
                f"{marker_start}\n{table_md}\n{marker_end}",
                content,
                flags=re.DOTALL,
            )
            readme_path.write_text(new_content, encoding="utf-8")
            print("Updated README.md cross-tab")


async def run_all_evals() -> None:
    """Run all ablation configs and cross-tab configs."""
    from evals.runner import run

    dataset_version = "7916cdf6ab75a462e77f20ab40428a10923998d5"

    # Run 5-row ablation (English only)
    print("=" * 60)
    print("Running 5-row ablation table (English, dev)")
    print("=" * 60)
    for name, config in CONFIGS:
        print(f"\n--- {name} ---")
        result = await run(
            config,
            split="dev",
            dataset_version=dataset_version,
            config_label=name,
            locale="en",
        )
        print(
            f"  NDCG@10={result.ndcg_10:.4f}  recall@50={result.recall_50:.4f}  "
            f"MRR@10={result.mrr_10:.4f}  p95={result.latency_p95_ms:.1f}ms  "
            f"skipped={result.skipped_count}"
        )

    # Run cross-tab (all locales × 4 configs)
    print("\n" + "=" * 60)
    print("Running cross-tab (3 locales × 4 configs)")
    print("=" * 60)
    for locale in LOCALES:
        for name, config in CROSS_TAB_CONFIGS:
            label = f"{name}_{locale}"
            print(f"\n--- {label} ---")
            result = await run(
                config,
                split="dev",
                dataset_version=dataset_version,
                config_label=label,
                locale=locale,
            )
            print(
                f"  NDCG@10={result.ndcg_10:.4f}  recall@50={result.recall_50:.4f}  "
                f"MRR@10={result.mrr_10:.4f}  p95={result.latency_p95_ms:.1f}ms"
            )


async def run_rrf_diagnostic() -> None:
    """Run RRF with k=10, 60, 200 to diagnose recall anomaly."""
    from evals.runner import run

    dataset_version = "7916cdf6ab75a462e77f20ab40428a10923998d5"

    print("=" * 60)
    print("RRF k Diagnostic")
    print("=" * 60)

    for k_val in [10, 60, 200]:
        config = ShopRankPipelineConfig(
            use_bm25=True,
            use_dense=True,
            use_rerank=False,
            top_k=100,
            fusion_method="rrf",
            embed_dim=768,
            ef_search=40,
            rrf_k=k_val,
        )
        print(f"\n--- RRF k={k_val} ---")
        result = await run(
            config,
            split="dev",
            dataset_version=dataset_version,
            config_label=f"rrf_k{k_val}",
            locale="en",
        )
        print(
            f"  NDCG@10={result.ndcg_10:.4f}  recall@50={result.recall_50:.4f}  "
            f"MRR@10={result.mrr_10:.4f}  p95={result.latency_p95_ms:.1f}ms"
        )

    # Also test asymmetric top_k
    print("\n--- Asymmetric top_k (BM25=50, Dense=100) diagnostic ---")
    print("  (Note: both retrievers use the same top_k from config;")
    print("   the recall anomaly is likely due to RRF penalizing")
    print("   items only in one list.)")


async def dump_bad_cases() -> None:
    """Dump the 20 worst-scoring queries per non-English locale."""
    from collections import defaultdict

    import asyncpg as apg
    from retrieval_core.models import Query

    from core.pipeline import close_pipeline, search
    from evals.metrics import ndcg_at_k

    settings = get_settings()
    pool = await apg.create_pool(settings.database_url)

    try:
        qrel_rows = await pool.fetch("SELECT query_id, product_id, esci_label FROM qrels")
        qrels: dict[str, dict[str, str]] = defaultdict(dict)
        for r in qrel_rows:
            qrels[r["query_id"]][r["product_id"]] = r["esci_label"]

        prod_rows = await pool.fetch("SELECT product_id, product_text FROM products")
        product_texts = {r["product_id"]: r["product_text"][:120] for r in prod_rows}

        for locale in ["zh", "fr"]:
            q_rows = await pool.fetch(
                "SELECT query_id, text FROM queries WHERE split = 'dev' AND locale = $1 ORDER BY query_id",
                locale,
            )

            if not q_rows:
                print(f"No {locale} queries found. Run translate_queries.py first.")
                continue

            config = ShopRankPipelineConfig(
                use_bm25=True,
                use_dense=True,
                use_rerank=True,
                top_k=100,
                fusion_method="rrf",
                embed_dim=768,
                ef_search=40,
                rerank_depth=50,
                locale=locale,
            )

            scored_queries: list[tuple[float, str, str, list[tuple[str, str, str]]]] = []

            for row in q_rows:
                q_id = row["query_id"]
                q_text = row["text"]

                response = await search(
                    Query(text=q_text), config, db_url=settings.database_url
                )

                base_qid = q_id
                for suffix in ("_zh", "_fr"):
                    base_qid = base_qid.removesuffix(suffix)

                q_qrels = qrels.get(base_qid, {})
                result_ids = [h.product_id for h in response.hits]
                ndcg, skipped = ndcg_at_k(q_qrels, result_ids, k=10)

                if skipped:
                    continue

                top5_info = []
                for h in response.hits[:5]:
                    label = q_qrels.get(h.product_id, "unjudged")
                    title = product_texts.get(h.product_id, "???")
                    top5_info.append((h.product_id, title, label))

                scored_queries.append((ndcg, q_id, q_text, top5_info))

            await close_pipeline()

            # Sort by NDCG ascending (worst first)
            scored_queries.sort(key=lambda x: x[0])
            worst_20 = scored_queries[:20]

            # Write markdown
            out_path = Path(f"evals/results/bad_cases_{locale}.md")
            lines = [
                f"# Worst 20 queries — {locale.upper()}",
                "",
                f"Generated from dev split, locale={locale}, hybrid+rerank pipeline.",
                "",
            ]

            for i, (ndcg_val, q_id, q_text, top5) in enumerate(worst_20, 1):
                lines.append(f"## {i}. NDCG@10 = {ndcg_val:.4f}")
                lines.append(f"**Query:** `{q_text}`  (id: {q_id})")
                lines.append("")
                lines.append("| Rank | Product | Label |")
                lines.append("| --- | --- | --- |")
                for rank, (pid, title, label) in enumerate(top5, 1):
                    lines.append(f"| {rank} | {title} | {label} |")
                lines.append("")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Wrote {len(worst_20)} bad cases to {out_path}")

    finally:
        await pool.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation table and evaluation tools")
    parser.add_argument("--cross-tab", action="store_true", help="Render cross-lingual table")
    parser.add_argument("--rrf-diagnostic", action="store_true", help="Run RRF k diagnostic")
    parser.add_argument("--bad-cases", action="store_true", help="Dump worst queries per locale")
    parser.add_argument("--run-all", action="store_true", help="Run all eval configs")
    args = parser.parse_args()

    if args.run_all:
        asyncio.run(run_all_evals())
    elif args.rrf_diagnostic:
        asyncio.run(run_rrf_diagnostic())
    elif args.bad_cases:
        asyncio.run(dump_bad_cases())
    elif args.cross_tab:
        asyncio.run(render_cross_tab())
    else:
        asyncio.run(render_ablation_table())
